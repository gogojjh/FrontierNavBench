"""Interfaces exposed to methods (interface layer, frozen after T-I0).

Methods interact with simulation through these types: each step receives an
:class:`Observation` and returns an :class:`Action`. They contain no world-frame
truth: odometry is relative to the episode start, and the goal is an image.
Units are meters, radians and simulation seconds; quaternions use xyzw order.
Signatures follow plan section 6.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Protocol, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Intrinsics = Tuple[float, ...]
"""Camera intrinsics: a row-major 3x3 matrix flattened into nine values."""


class Pose2D(NamedTuple):
    x: float  # m
    y: float  # m
    yaw: float  # rad; counterclockwise is positive


@dataclass(frozen=True)
class OdomState:
    """Episode-relative odometry, starting at the spawn pose; simulation has no drift."""

    pose: Pose2D
    position_xyz: Vec3
    quat_xyzw: Quat
    v_body: Vec3  # Body-frame linear velocity, m/s
    w_body: Vec3  # Body-frame angular velocity, rad/s


@dataclass(frozen=True)
class Observation:
    """All information available to a method at one decision step."""

    step_index: int  # Decision index, starting at zero
    sim_time_s: float  # Elapsed episode simulation time, excluding spawn settling
    budget_left_s: float  # Remaining simulation budget; do not expose it to the VLM (plan section
    # 4-9)
    rgb: np.ndarray  # (480, 640, 3), uint8, RGB order
    rgb_jpeg: bytes  # Original simulation JPEG (quality 85), uploaded unchanged to the VLM
    depth_m: np.ndarray  # (480, 640), float32 meters; zero is invalid, maximum 8.0
    K: Intrinsics
    odom: OdomState
    frames: Tuple[Tuple[float, bytes], ...] = ()  # First-person JPEGs at 10 Hz during the previous
    # hold, for video output


@dataclass(frozen=True)
class Action:
    """Body-frame velocity and hold duration; see :mod:`vnav_benchmark.fairness` for limits."""

    vx: float
    vy: float
    wz: float
    hold_s: float  # Simulation seconds, quantized to 0.1 within [0.1, 2.0]
    done: bool = False  # Method declares arrival; the evaluator determines success using ground
    # truth
    note: str = ""
    debug: Mapping[str, Any] = field(default_factory=dict)
    # debug["latency"] keys: vlm_api_s / server_ms / retries / controller_s / loc_s; logs only


@dataclass(frozen=True)
class EpisodeSpec:
    """Runtime episode view with opaque identifiers, excluding truth and difficulty labels."""

    episode_token: str
    repeat_index: int
    scene: str
    goal_rgb: np.ndarray
    goal_rgb_jpeg: bytes
    goal_depth_m: np.ndarray
    goal_K: Intrinsics
    budget_s: float


@dataclass(frozen=True)
class LocResult:
    ok: bool
    pose_map: Optional[Pose2D]  # Body pose in the map frame
    kf_id: Optional[int]  # Nearest keyframe; ideal localization may return None
    reason: str = ""  # Failure reason when ok=False, such as "quota" or "no_map"
    floor: Optional[str] = None  # Floor ID; an unknown floor imposes no floor restriction


class Localizer(Protocol):
    """Localize a goal image (purpose="goal") or the robot ("self").

    Method-specific quotas follow plan section 6.4.
    """

    def localize(
        self, rgb_jpeg: bytes, depth_m: Optional[np.ndarray], K: Intrinsics, *, purpose: str
    ) -> LocResult: ...


class MapHandle(Protocol):
    """Sanitized read-only map; implemented in mapview.py."""

    map_id: str
    variant: str  # "dense" | "sparse"
    camera_forward_m: float
    sha256: str
    edge_type: str

    def topo_graph(self) -> Any: ...  # vnav_benchmark.topo.TopoGraph

    def keyframe_pose(self, kf_id: int) -> Pose2D: ...

    def keyframe_floor(self, kf_id: int) -> Optional[str]: ...

    def floor_at_z(self, z: float) -> Optional[str]: ...  # Gravity-aligned world/odometry height

    def keyframe_rgb_jpeg(self, kf_id: int) -> bytes: ...

    def keyframe_depth_m(self, kf_id: int) -> np.ndarray: ...


@dataclass(frozen=True)
class UsageRecord:
    """Usage for one paid VLM call; BudgetGuard accounts using billed_usd."""

    provider: str  # "openai" | "dashscope"
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    list_usd: float = 0.0  # Cost at list prices
    billed_usd: float = 0.0  # Actual billing tier, including discounts such as half-price Flex
    request_id: str = ""


class UsageSink(Protocol):
    def add(self, record: UsageRecord) -> None: ...

    def episode_billed_usd(self) -> float: ...


class MethodProfileLike(Protocol):
    """Read-only fairness.MethodProfile view, avoiding a reverse import from interfaces."""

    method_id: str
    map_variant: Optional[str]
    loc_mode: str
    planner: str
    vlm: Optional[str]
    progress_judge: str  # Subgoal progression judge: pose / vlm
    done_judge: str  # Arrival judge: pose / vlm


@dataclass(frozen=True)
class MethodContext:
    profile: MethodProfileLike
    config: Mapping[str, Any]  # Dictionary loaded from configs/methods/<method_id>.yaml
    map: Optional[MapHandle]  # None for nomap methods
    localizer: Optional[Localizer]  # None for noloc methods; otherwise quota-limited
    rng: np.random.Generator  # Derived from (seed, episode, repeat); use for all method randomness
    log: logging.Logger
    usage: UsageSink
    artifact_dir: Path  # Method-owned artifact directory (runs/.../method/)


class Method(Protocol):
    method_id: str
    uses_truth: bool  # True for unscored debugging methods

    def reset(self, spec: EpisodeSpec, obs: Observation, ctx: MethodContext) -> None: ...

    def act(self, obs: Observation) -> Action: ...

    def close(self) -> None: ...


class InfraError(Exception):
    """Retry unscored when simulation, the VLM provider or the SRU process is unavailable.

    Other exceptions count as scored method failures.
    """


class FatalInfraError(InfraError):
    """Unrecoverable infrastructure error, such as exhausted quota or authentication failure.

    Leave the episode unscored and stop the campaign.
    """
