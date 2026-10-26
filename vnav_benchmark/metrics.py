"""Evaluation metrics using world-frame truth, hidden from methods (frozen after T-I0).

Both success criteria appear in the main table (plan sections 3 and 6.7). A method
must declare done within 0.5 m and 45 degrees for strict success, or 2 m and
90 degrees for relaxed success. Timeouts, falls and method exceptions fail.
SPL = S * l / max(p, l); PLR = p / l is defined only for successful episodes.
The shortest-path length l comes from the truth grid in oracle.py.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vnav_benchmark import oracle
from vnav_benchmark.interfaces import Pose2D

SUCCESS_DIST_M = 0.5  # Primary success distance
SUCCESS_YAW_DEG = 45.0  # Primary success yaw error
SUCCESS_DIST_LOOSE_M = 2.0  # Relaxed success distance; replaces the former SR@1.0 m criterion
SUCCESS_YAW_LOOSE_DEG = 90.0  # Relaxed success yaw error
PATH_HYSTERESIS_M = 0.05  # Path hysteresis: accumulate only after moving 5 cm from the anchor;
# ignore standing sway
FALL_UP_Z = 0.5  # Trunk up-axis threshold for inversion, matching go2_rig.FALLEN_UP_Z
FALL_HOLD_S = 1.0  # Continuous inverted duration required to declare a fall
STUCK_WINDOW_S = 15.0  # Time window for detecting a stuck robot
STUCK_MIN_MOVE_M = 0.3  # Minimum displacement in that window; counted without declaring failure
PROXIMITY_DEPTH_M = 0.3  # Central depth median below 0.3 m indicates obstacle proximity; counted
# only
PROXIMITY_CROP = (1 / 3, 2 / 3)  # Use the central third of both image dimensions

TERMINATIONS = ("done", "timeout", "fallen", "method_error", "act_timeout", "cost_guard")
SCORED_TERMINATIONS = TERMINATIONS[:4]
"""Scored terminations; act_timeout and cost_guard are recorded but unscored guard events."""


def wrap_angle(a: float) -> float:
    return math.remainder(a, 2.0 * math.pi)


def pose_error(pose: Pose2D, goal: Pose2D) -> Tuple[float, float]:
    """Return (distance in meters, absolute yaw error in degrees within 0-180)."""
    dist = math.hypot(pose.x - goal.x, pose.y - goal.y)
    return dist, abs(math.degrees(wrap_angle(pose.yaw - goal.yaw)))


def within_pose(pose: Pose2D, goal: Pose2D) -> bool:
    dist, yaw_deg = pose_error(pose, goal)
    return dist <= SUCCESS_DIST_M + 1e-9 and yaw_deg <= SUCCESS_YAW_DEG + 1e-9


def within_loose(pose: Pose2D, goal: Pose2D) -> bool:
    dist, yaw_deg = pose_error(pose, goal)
    return dist <= SUCCESS_DIST_LOOSE_M + 1e-9 and yaw_deg <= SUCCESS_YAW_LOOSE_DEG + 1e-9


def path_length_m(
    xy: Sequence[Tuple[float, float]], hysteresis_m: float = PATH_HYSTERESIS_M
) -> float:
    """Accumulate path length only after moving hysteresis distance from the anchor.

    Advance the anchor for each accepted point and include the final residual segment.
    """
    if len(xy) < 2:
        return 0.0
    total = 0.0
    anchor = xy[0]
    for point in xy[1:]:
        step = math.hypot(point[0] - anchor[0], point[1] - anchor[1])
        if step >= hysteresis_m:
            total += step
            anchor = point
    total += math.hypot(xy[-1][0] - anchor[0], xy[-1][1] - anchor[1])
    return total


def path_length_3d_m(points: Sequence[Tuple[float, float, float]]) -> float:
    """Sum consecutive world-space distances, including height, for multi-floor traces."""
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)
    return total


def spl(success: bool, shortest_m: float, path_m: float) -> float:
    if not success or shortest_m <= 0.0:
        return 0.0
    return shortest_m / max(path_m, shortest_m)


def plr(success: bool, shortest_m: float, path_m: float) -> Optional[float]:
    """Path-length ratio, defined only for success; values closer to one are better."""
    if not success or shortest_m <= 0.0:
        return None
    return path_m / shortest_m


class FallMonitor:
    """Declare a fall after up_z < 0.5 for at least 1 s; return the first detection sim time."""

    def __init__(self, up_z_min: float = FALL_UP_Z, hold_s: float = FALL_HOLD_S) -> None:
        self.up_z_min = up_z_min
        self.hold_s = hold_s
        self._down_since: Optional[float] = None
        self.fallen_at: Optional[float] = None

    def update(self, t_sim: float, up_z: float) -> bool:
        if up_z < self.up_z_min:
            if self._down_since is None:
                self._down_since = t_sim
            if self.fallen_at is None and t_sim - self._down_since >= self.hold_s - 1e-9:
                self.fallen_at = t_sim
        else:
            self._down_since = None
        return self.fallen_at is not None


class StuckMonitor:
    """Count displacement below 0.3 m over 15 s as stuck, then restart the nonoverlapping window."""

    def __init__(
        self, window_s: float = STUCK_WINDOW_S, min_move_m: float = STUCK_MIN_MOVE_M
    ) -> None:
        self.window_s = window_s
        self.min_move_m = min_move_m
        self._samples: List[Tuple[float, float, float]] = []
        self.events: List[float] = []

    def update(self, t_sim: float, x: float, y: float) -> None:
        self._samples.append((t_sim, x, y))
        while self._samples and t_sim - self._samples[0][0] > self.window_s + 1e-9:
            self._samples.pop(0)
        if not self._samples or t_sim - self._samples[0][0] < self.window_s - 1e-9:
            return
        moved = max(math.hypot(x - sx, y - sy) for _, sx, sy in self._samples)
        if moved < self.min_move_m:
            self.events.append(t_sim)
            self._samples = [(t_sim, x, y)]


def is_proximity(depth_m: np.ndarray) -> bool:
    """Check whether valid depth in the central image third has a median below 0.3 m."""
    h, w = depth_m.shape[:2]
    r0, r1 = int(h * PROXIMITY_CROP[0]), int(h * PROXIMITY_CROP[1])
    c0, c1 = int(w * PROXIMITY_CROP[0]), int(w * PROXIMITY_CROP[1])
    crop = depth_m[r0:r1, c0:c1]
    valid = crop[crop > 0.0]
    return bool(valid.size) and float(np.median(valid)) < PROXIMITY_DEPTH_M


def latency_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


@dataclass
class TraceRow:
    """One world-frame ground-truth sample for evaluation."""

    t_sim: float
    x: float
    y: float
    yaw: float
    up_z: float = 1.0
    z: float = 0.0  # World height; appended to preserve existing positional argument semantics


@dataclass
class EpisodeOutcome:
    termination: str
    success: bool  # Primary criterion: 0.5 m / 45 degrees
    success_loose: bool  # Relaxed criterion: 2 m / 90 degrees
    final_dist_m: Optional[float]  # Planar error is undefined across floors; record null
    final_yaw_err_deg: Optional[float]
    ever_within_pose: bool
    premature_done: bool  # Declared done outside the primary success tolerances
    falls: int
    fallen_at_s: Optional[float]
    stuck_events: int
    proximity_events: int
    path_length_m: float
    shortest_path_m: float
    spl: float
    plr: Optional[float]
    sim_time_s: float
    decisions: int
    act_wall_s: Dict[str, Optional[float]] = field(default_factory=dict)
    compute_per_sim_s: Optional[float] = None  # Equivalent wall/simulation ratio: total computation
    # wall time / simulation time

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def find_fall(trace: Sequence[TraceRow]) -> Optional[float]:
    monitor = FallMonitor()
    for row in trace:
        if monitor.update(row.t_sim, row.up_z):
            break
    return monitor.fallen_at


def evaluate_run(
    trace: Sequence[TraceRow],
    goal: Pose2D,
    shortest_m: float,
    declared_done: bool,
    termination: str,
    decisions: int,
    act_wall_s: Sequence[float] = (),
    proximity_events: int = 0,
    goal_scene: Optional[str] = None,
    goal_floor: Optional[str] = None,
) -> EpisodeOutcome:
    """Score an episode using its runner termination and truth trajectory.

    A detected fall overrides the termination to fallen. Truncate the trajectory
    at the fall time for both path length and checks of whether the goal was visited.
    goal_scene=None preserves the legacy criterion; multistory scenes require
    the episode goal_floor. Legacy single-floor episodes with an empty goal_floor
    use the scene's sole navigation floor. Cross-floor terminal errors are null.
    """
    if termination not in TERMINATIONS:
        raise ValueError(f"未知结局 {termination!r}")
    if not trace:
        raise ValueError("轨迹为空")
    if goal_scene == oracle.SCENE_ENV3 and not goal_floor:
        raise ValueError("多层评估需要 goal_floor")
    target_floor = (
        goal_floor or oracle.floor_of_z(goal_scene, 0.0) if goal_scene is not None else None
    )

    def same_floor(row: TraceRow) -> bool:
        return goal_scene is None or oracle.floor_of_z(goal_scene, row.z) == target_floor

    fallen_at = find_fall(trace)
    rows = [r for r in trace if fallen_at is None or r.t_sim <= fallen_at + 1e-9]
    if fallen_at is not None and termination in ("done", "timeout"):
        termination = "fallen"
    final = rows[-1]
    final_pose = Pose2D(final.x, final.y, final.yaw)
    dist, yaw_err = pose_error(final_pose, goal)
    final_same_floor = same_floor(final)
    done = declared_done and termination == "done"
    success = done and final_same_floor and within_pose(final_pose, goal)
    success_loose = done and final_same_floor and within_loose(final_pose, goal)
    stuck = StuckMonitor()
    for r in rows:
        stuck.update(r.t_sim, r.x, r.y)
    path = (
        path_length_3d_m([(r.x, r.y, r.z) for r in rows])
        if goal_scene == oracle.SCENE_ENV3
        else path_length_m([(r.x, r.y) for r in rows])
    )
    sim_time = rows[-1].t_sim - rows[0].t_sim
    compute = float(sum(act_wall_s)) / sim_time if act_wall_s and sim_time > 0 else None
    return EpisodeOutcome(
        termination=termination,
        success=success,
        success_loose=success_loose,
        final_dist_m=dist if final_same_floor else None,
        final_yaw_err_deg=yaw_err if final_same_floor else None,
        ever_within_pose=any(
            same_floor(r) and within_pose(Pose2D(r.x, r.y, r.yaw), goal) for r in rows
        ),
        premature_done=done and not success,
        falls=int(fallen_at is not None),
        fallen_at_s=fallen_at,
        stuck_events=len(stuck.events),
        proximity_events=int(proximity_events),
        path_length_m=path,
        shortest_path_m=shortest_m,
        spl=spl(success, shortest_m, path),
        plr=plr(success, shortest_m, path),
        sim_time_s=sim_time,
        decisions=int(decisions),
        act_wall_s=latency_stats(act_wall_s),
        compute_per_sim_s=compute,
    )


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Bootstrap mean confidence intervals by start-goal pair, for reports with one repeat."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
