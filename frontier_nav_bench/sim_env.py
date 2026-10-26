"""SimEnv converts world-frame wire truth into method-visible observations.

The interface layer is frozen after T-I0. Observations contain only episode-relative
odometry, using ``nav_sim.odom_reset.OdomOrigin`` with the same semantics as
``/sim_bridge/reset_odom``. World truth stays inside SimEnv for evaluation/localization.
Only executed holds consume budget; floor the last hold to the remaining 0.1 s
quantum. Computation time is excluded. Hold segments provide 10 Hz first-person
frames in ``Observation.frames`` for video output, alongside 10 Hz odometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np

from frontier_nav_bench import fairness
from frontier_nav_bench.interfaces import Action, Observation, OdomState, Pose2D
from frontier_nav_bench.sim import codecs as nav_codecs
from frontier_nav_bench.sim import ctrl_protocol as cp
from frontier_nav_bench.sim.frames import OdomOrigin, yaw_from_quat

DEPTH_MAX_M = 8.0  # Depth range; out-of-range and invalid values become zero


class SimLink(Protocol):
    """Client subset used by SimEnv, implemented by CtrlClient and InProcessClient."""

    def reset(self, x: float, y: float, yaw: float) -> cp.WireObs: ...

    def step(
        self,
        vx: float,
        vy: float,
        wz: float,
        hold_s: float,
        frames: bool = True,
        third: bool = False,
    ) -> cp.StepReply: ...


def pose2d_of(pose: cp.WirePose) -> Pose2D:
    """Convert a world-frame wire pose to planar pose for evaluation/localization, hidden from
    methods.
    """
    return Pose2D(
        float(pose.position_xyz[0]), float(pose.position_xyz[1]), yaw_from_quat(pose.quat_xyzw)
    )


def decode_obs_images(obs: cp.WireObs) -> Tuple[np.ndarray, np.ndarray]:
    """Decode JPEG to RGB uint8 and depth to float32 meters, where zero means invalid."""
    bgr = cv2.imdecode(np.frombuffer(obs.rgb_jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("RGB JPEG 解不开")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    depth_mm = nav_codecs.decode_depth(obs.depth, obs.width, obs.height, obs.depth_encoding)
    depth = nav_codecs.millimeters_to_meters(depth_mm).astype(np.float32)
    depth[(depth <= 0.0) | (depth > DEPTH_MAX_M) | ~np.isfinite(depth)] = 0.0
    return rgb, depth


@dataclass
class TruthTrace:
    """Episode world truth at 10 Hz plus decision times, accessible only to evaluation."""

    samples: List[Tuple[float, cp.WirePose]] = field(default_factory=list)

    def add(self, t_sim: float, pose: cp.WirePose) -> None:
        if self.samples and t_sim <= self.samples[-1][0] + 1e-9:
            return  # Record each timestamp once; observation time equals the final trajectory
            # sample
        self.samples.append((t_sim, pose))


class SimEnv:
    """One instance per episode: start creates observation zero; each step(action) advances one
    hold.
    """

    def __init__(self, link: SimLink, budget_s: float, want_third: bool = False) -> None:
        self.link = link
        self.budget_s = float(budget_s)
        self.want_third = want_third
        self.origin: Optional[OdomOrigin] = None
        self.truth = TruthTrace()
        self.sim_time = 0.0
        self.step_index = 0
        self.last_wire: Optional[cp.WireObs] = None
        self.third_frames: List[Tuple[float, bytes]] = []
        self.applied: List[Tuple[float, Tuple[float, float, float], float]] = []  # (t, cmd, hold)
        self.server_clamped = 0
        self.fallen_flags = 0

    # -- Ground truth for evaluation/localization only --
    def truth_pose(self) -> Pose2D:
        assert self.last_wire is not None, "先 start"
        return pose2d_of(self.last_wire.pose)

    def truth_wire(self) -> cp.WirePose:
        assert self.last_wire is not None, "先 start"
        return self.last_wire.pose

    @property
    def budget_left_s(self) -> float:
        return max(0.0, round(self.budget_s - self.sim_time, 6))

    @property
    def out_of_budget(self) -> bool:
        return self.budget_left_s < fairness.HOLD_MIN_S - 1e-9

    # -- Projection --
    def _odom(self, pose: cp.WirePose) -> OdomState:
        assert self.origin is not None
        position, quat = self.origin.to_local_pose(pose.position_xyz, pose.quat_xyzw)
        return OdomState(
            pose=Pose2D(float(position[0]), float(position[1]), yaw_from_quat(quat)),
            position_xyz=(float(position[0]), float(position[1]), float(position[2])),
            quat_xyzw=tuple(float(v) for v in quat),  # type: ignore[arg-type]
            v_body=tuple(float(v) for v in pose.v_body),  # type: ignore[arg-type]
            w_body=tuple(float(v) for v in pose.w_body),  # type: ignore[arg-type]
        )

    def _project(self, wire: cp.WireObs, frames: Tuple[Tuple[float, bytes], ...]) -> Observation:
        rgb, depth = decode_obs_images(wire)
        return Observation(
            step_index=self.step_index,
            sim_time_s=round(self.sim_time, 6),
            budget_left_s=self.budget_left_s,
            rgb=rgb,
            rgb_jpeg=wire.rgb_jpeg,
            depth_m=depth,
            K=tuple(float(v) for v in wire.K),
            odom=self._odom(wire.pose),
            frames=frames,
        )

    # -- Episode lifecycle --
    def start(self, start: Pose2D) -> Observation:
        """Settle without consuming budget, then anchor odometry at the body pose."""
        wire = self.link.reset(float(start.x), float(start.y), float(start.yaw))
        self.origin = OdomOrigin.from_pose(wire.pose.position_xyz, wire.pose.quat_xyzw)
        self.sim_time = 0.0
        self.step_index = 0
        self.last_wire = wire
        self.truth.add(0.0, wire.pose)
        return self._project(wire, ((0.0, wire.rgb_jpeg),))

    def clip_hold(self, hold_s: float) -> float:
        """Clamp the final hold to remaining budget, rounding down to a multiple of 0.1 s."""
        steps_left = int(math.floor(self.budget_left_s / fairness.HOLD_QUANTUM_S + 1e-6))
        return round(min(hold_s, steps_left * fairness.HOLD_QUANTUM_S), 6)

    def step(self, action: Action) -> Tuple[Observation, cp.StepReply]:
        """Execute an already-clamped action; calling after budget exhaustion is a programming
        error.
        """
        hold = self.clip_hold(action.hold_s)
        if hold < fairness.HOLD_MIN_S - 1e-9:
            raise RuntimeError("预算已用完,不能再 step")
        reply = self.link.step(
            action.vx, action.vy, action.wz, hold, frames=True, third=self.want_third
        )
        self.applied.append((round(self.sim_time, 6), reply.applied_cmd, hold))
        self.server_clamped += int(reply.clamped)
        self.fallen_flags += int(reply.fallen_now)
        for sample in reply.trace:
            self.truth.add(sample.t_sim, sample.pose)
        self.truth.add(reply.obs.t_sim, reply.obs.pose)
        self.third_frames.extend(reply.third)
        self.sim_time = round(self.sim_time + hold, 6)
        self.step_index += 1
        self.last_wire = reply.obs
        return self._project(reply.obs, tuple(reply.frames)), reply


def observation_field_names() -> Tuple[str, ...]:
    """Return all Observation/OdomState field names for tests excluding *_world fields."""
    import dataclasses

    names: List[str] = []
    for cls in (Observation, OdomState):
        names += [f.name for f in dataclasses.fields(cls)]
    return tuple(names)
