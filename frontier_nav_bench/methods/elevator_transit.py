"""Experimental transit geometry, without semantic elevator or cabin guarantees.

Compact depth, a floor-changing edge and a route reversal identify a candidate.
Projected wall clearance only concerns observed points; it does not establish an
unseen cabin envelope or prove that the robot's whole body is inside a cabin.
"""

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from frontier_nav_bench.interfaces import MapHandle, Pose2D

# Image fractions select a horizontal wall band, away from the floor and ceiling.
BAND_TOP = 0.30
BAND_BOTTOM = 0.60
FRONT_LEFT = 0.35  # Central rays used to detect a visibly open exit.
FRONT_RIGHT = 0.65
COMPACT_VALID_RATIO = 0.50  # Strictly more than half the band must have valid depth.
COMPACT_MEDIAN_M = 1.0  # A nearby wall envelope is required for a compact candidate.
COMPACT_P90_M = 1.5  # Reject broad, open depth distributions such as hallways.
COMPACT_PERCENTILE = 90
MAX_FLOOR_EDGE_M = 4.0  # Long floor-changing planar edges are not transit candidates.
MIN_ROUTE_TURN_RAD = math.radians(110.0)  # Require a substantial approach/exit reversal.
BOARD_OFFSET_M = 0.40  # Continue along the observed route approach direction.
EXIT_OFFSET_M = 1.50  # Exit back along the same route-derived axis.
MIN_BOARD_CLEARANCE_M = 0.45  # Body-center clearance from every observed wall point.
MIN_DIRECTION_M = 1e-6  # Degenerate route vectors cannot define a safe direction.
K_TOLERANCE = 1e-6  # Tolerance for the homogeneous row of a pinhole calibration.
ACTIVATE_RADIUS_M = 1.60  # Intercept before the normal topological one-meter switch.
TARGET_RADIUS_M = 0.30  # Boarding and exit use a tighter position tolerance.
TURN_TOLERANCE_RAD = 0.15  # Face the exit before waiting for transport.
MIN_VERTICAL_TRAVEL_M = 0.50  # A floor label alone cannot establish vertical transport.
STABLE_HEIGHT_RANGE_M = 0.08  # Maximum vertical range throughout a stability window.
STABLE_DURATION_S = 1.0  # Do not release on a single arrival-height observation.
FRONT_CLEAR_M = 1.20  # Front depth must show open space rather than the closed wall.
FRONT_CLEAR_PERCENTILE = 20
FRONT_VALID_RATIO = 0.60  # Missing depth cannot establish an open exit.
CHECKPOINT_VERSION = 1
PHASES = ("idle", "board", "turn", "wait", "exit", "complete")


@dataclass(frozen=True)
class TransitPlan:
    """Route-derived motion targets and JSON-compatible evidence for one crossing."""

    anchor: Pose2D
    boarding: Pose2D
    exit: Pose2D
    source_floor: str
    target_floor: str
    anchor_kf: int
    consumed_kfs: Tuple[int, ...]
    evidence: Dict[str, Any]


@dataclass(frozen=True)
class TransitTarget:
    """One motion target or explicit hold, consumed by the method's controller."""

    phase: str
    target: Optional[Pose2D] = None
    yaw_error: float = 0.0
    hold: bool = False
    completed: bool = False


def _distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _finite_pose(pose: Pose2D) -> bool:
    return all(math.isfinite(value) for value in pose)


def _band(depth: np.ndarray, central: bool = False) -> np.ndarray:
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or not depth.size:
        return np.empty((0, 0))
    height, width = depth.shape
    left, right = (int(width * FRONT_LEFT), int(width * FRONT_RIGHT)) if central else (0, width)
    return depth[int(height * BAND_TOP):int(height * BAND_BOTTOM), left:right]


def _valid(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0.0)


def _compact_stats(depth: np.ndarray) -> Optional[Dict[str, float]]:
    band = _band(depth)
    if not band.size:
        return None
    valid = _valid(band)
    ratio = float(np.mean(valid))
    if ratio <= COMPACT_VALID_RATIO:
        return None
    values = band[valid]
    median = float(np.median(values))
    p90 = float(np.percentile(values, COMPACT_PERCENTILE))
    if median > COMPACT_MEDIAN_M or p90 > COMPACT_P90_M:
        return None
    return {"depth_median_m": median, "depth_p90_m": p90, "depth_valid_ratio": ratio}


def _intrinsics(K: Tuple[float, ...]) -> Optional[np.ndarray]:
    """Validate a supplied calibration, without substituting a default camera."""
    try:
        matrix = np.asarray(K, dtype=float).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        return None
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=K_TOLERANCE, rtol=0.0):
        return None
    if abs(float(np.linalg.det(matrix))) <= K_TOLERANCE:
        return None
    return matrix


def _boarding_clearance(
    depth: np.ndarray, K: np.ndarray, anchor: Pose2D, boarding: Pose2D, camera_forward_m: float,
) -> Optional[float]:
    """Project all valid band rays to the map plane and measure observed clearance.

    Camera optical +z is body forward and optical +x is body right, following
    the public map pose convention. Only planar yaw and camera forward offset
    are available; no unseen wall or cabin dimensions are reconstructed.
    """
    height, width = depth.shape
    if not (0.0 <= K[0, 2] < width and 0.0 <= K[1, 2] < height):
        return None
    band = _band(depth)
    valid = _valid(band)
    # A missing central sector must not disappear behind the whole-band ratio.
    center = _band(depth, central=True)
    if not center.size or float(np.mean(_valid(center))) <= COMPACT_VALID_RATIO:
        return None
    rows, cols = np.nonzero(valid)
    rows = rows + int(height * BAND_TOP)
    pixels = np.stack((cols, rows, np.ones_like(rows)))
    rays = np.linalg.solve(K, pixels)
    distances = band[valid]
    forward = distances + camera_forward_m
    left = -rays[0] * distances
    cos_yaw, sin_yaw = math.cos(anchor.yaw), math.sin(anchor.yaw)
    world_x = anchor.x + cos_yaw * forward - sin_yaw * left
    world_y = anchor.y + sin_yaw * forward + cos_yaw * left
    clearance = np.hypot(world_x - boarding.x, world_y - boarding.y)
    return float(np.min(clearance)) if clearance.size else None


def _candidate(
    map_handle: MapHandle, route: Tuple[int, ...], edge: int,
    start_map: Pose2D, K: np.ndarray,
) -> Optional[TransitPlan]:
    """Derive one candidate from the shallower compact endpoint of a floor edge."""
    endpoints = []
    for index in (edge, edge + 1):
        try:
            depth = map_handle.keyframe_depth_m(route[index])
        except Exception:
            # Public map implementations have their own depth-I/O exception types.
            # Unavailable evidence explicitly means no candidate, never a fallback.
            return None
        stats = _compact_stats(depth)
        if stats is not None:
            endpoints.append((stats["depth_median_m"], index, depth, stats))
    if not endpoints:
        return None
    _, index, depth, stats = min(endpoints, key=lambda item: item[0])
    if index + 1 >= len(route):
        return None
    anchor = map_handle.keyframe_pose(route[index])
    predecessor = map_handle.keyframe_pose(route[index - 1]) if index else start_map
    successor = map_handle.keyframe_pose(route[index + 1])
    if not all(_finite_pose(pose) for pose in (anchor, predecessor, successor)):
        return None
    incoming = np.array((anchor.x - predecessor.x, anchor.y - predecessor.y))
    outgoing = np.array((successor.x - anchor.x, successor.y - anchor.y))
    incoming_norm, outgoing_norm = float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))
    if min(incoming_norm, outgoing_norm) <= MIN_DIRECTION_M:
        return None
    incoming /= incoming_norm
    outgoing /= outgoing_norm
    turn = math.acos(float(np.clip(incoming @ outgoing, -1.0, 1.0)))
    if turn < MIN_ROUTE_TURN_RAD:
        return None
    dx, dy = float(incoming[0]), float(incoming[1])
    boarding = Pose2D(anchor.x + BOARD_OFFSET_M * dx, anchor.y + BOARD_OFFSET_M * dy,
                      math.atan2(dy, dx))
    exit_pose = Pose2D(anchor.x - EXIT_OFFSET_M * dx, anchor.y - EXIT_OFFSET_M * dy,
                       math.atan2(-dy, -dx))
    clearance = _boarding_clearance(depth, K, anchor, boarding, map_handle.camera_forward_m)
    if clearance is None or clearance < MIN_BOARD_CLEARANCE_M:
        return None
    evidence = dict(stats)
    evidence.update({"route_turn_rad": turn, "boarding_clearance_m": clearance,
                     "floor_edge": [route[edge], route[edge + 1]],
                     "classification": "depth_geometric_candidate"})
    return TransitPlan(anchor, boarding, exit_pose,
                       map_handle.keyframe_floor(route[edge]),
                       map_handle.keyframe_floor(route[edge + 1]),
                       route[index], route[:index + 1], evidence)


def plan_transit(
    map_handle: MapHandle, route_ids: Sequence[int], start_map: Pose2D, K: Tuple[float, ...],
) -> Optional[TransitPlan]:
    """Find exactly one identifiable geometric floor-crossing transit candidate.

    Args:
        map_handle: Public map poses, floor labels, metric depth and camera offset.
        route_ids: Ordered route keyframes, with approach and departure evidence.
        start_map: Current pose used as predecessor if the first route node is compact.
        K: Current observation's flattened pinhole calibration; never guessed.

    Returns:
        One plan with observed clearance evidence, or None for missing evidence,
        an ordinary route, or ambiguous candidates. No semantic guarantee is made.
    """
    matrix = _intrinsics(K)
    route = tuple(route_ids)
    if matrix is None or len(route) < 2 or not _finite_pose(start_map):
        return None
    keyframe_floor = getattr(map_handle, "keyframe_floor", None)
    if not callable(keyframe_floor):
        return None
    if len(set(route)) != len(route):
        return None
    graph = map_handle.topo_graph()
    if any(kf not in graph.poses for kf in route):
        return None
    try:
        floors = [keyframe_floor(kf) for kf in route]
        if any(not floor for floor in floors):
            return None
        crossings = [edge for edge in range(len(route) - 1) if floors[edge] != floors[edge + 1]]
        if len(crossings) != 1:
            return None
        camera_forward_m = getattr(map_handle, "camera_forward_m", None)
        if (camera_forward_m is None or not math.isfinite(camera_forward_m)
                or camera_forward_m < 0.0):
            return None
        edge = crossings[0]
        source_pose = map_handle.keyframe_pose(route[edge])
        target_pose = map_handle.keyframe_pose(route[edge + 1])
        if not _finite_pose(source_pose) or not _finite_pose(target_pose):
            return None
        if _distance(source_pose, target_pose) > MAX_FLOOR_EDGE_M:
            return None
        return _candidate(map_handle, route, edge, start_map, matrix)
    except (KeyError, OSError, ValueError, np.linalg.LinAlgError):
        # Missing map evidence never activates a partly constructed transit plan.
        return None


def _plan_fingerprint(plan: TransitPlan) -> str:
    """Bind checkpoint ownership to the rederived geometry and directed route."""
    identity = {
        "anchor": tuple(plan.anchor), "boarding": tuple(plan.boarding), "exit": tuple(plan.exit),
        "source_floor": plan.source_floor, "target_floor": plan.target_floor,
        "anchor_kf": plan.anchor_kf, "consumed_kfs": plan.consumed_kfs,
    }
    payload = json.dumps(identity, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _front_clear(depth: np.ndarray) -> bool:
    roi = _band(depth, central=True)
    if not roi.size:
        return False
    valid = _valid(roi)
    if float(np.mean(valid)) < FRONT_VALID_RATIO:
        return False
    return float(np.percentile(roi[valid], FRONT_CLEAR_PERCENTILE)) > FRONT_CLEAR_M


class ElevatorTransit:
    """Keep controller ownership through boarding, transport, and the observed exit."""

    def __init__(self, plan: TransitPlan, start_z: float) -> None:
        if not math.isfinite(start_z):
            raise ValueError("Transit requires a finite initial odometry height")
        self.plan = plan
        self._plan_fingerprint = _plan_fingerprint(plan)
        self._start_z = float(start_z)
        self._phase = "idle"
        self._started_t: Optional[float] = None
        self._last_t: Optional[float] = None
        self._stable_since: Optional[float] = None
        self._stable_min_z: Optional[float] = None
        self._stable_max_z: Optional[float] = None
        self.stats: Dict[str, Any] = {"phase_events": [], "steps": 0, "elapsed_s": 0.0}

    @property
    def phase(self) -> str:
        """Current phase; idle leaves ordinary route following in control."""
        return self._phase

    @property
    def completed(self) -> bool:
        """Whether the exit target was reached on the destination floor."""
        return self._phase == "complete"

    def _transition(self, phase: str, t: float) -> None:
        self._phase = phase
        self.stats["phase_events"].append({"phase": phase, "t": float(t)})

    def _reset_stability(self) -> None:
        self._stable_since = self._stable_min_z = self._stable_max_z = None

    def _stable(self, floor: Optional[str], z: float, t: float) -> bool:
        """Require continuous destination-floor height evidence over the full window."""
        if floor != self.plan.target_floor or abs(z - self._start_z) < MIN_VERTICAL_TRAVEL_M:
            self._reset_stability()
            return False
        if self._stable_since is None:
            self._stable_since, self._stable_min_z, self._stable_max_z = t, z, z
            return False
        low, high = min(self._stable_min_z, z), max(self._stable_max_z, z)
        if high - low > STABLE_HEIGHT_RANGE_M:
            self._stable_since, self._stable_min_z, self._stable_max_z = t, z, z
            return False
        self._stable_min_z, self._stable_max_z = low, high
        return t - self._stable_since >= STABLE_DURATION_S

    def _target(self, robot: Pose2D) -> TransitTarget:
        if self._phase == "board":
            return TransitTarget("board", target=self.plan.boarding)
        if self._phase == "turn":
            error = self.plan.exit.yaw - robot.yaw
            return TransitTarget("turn", yaw_error=math.atan2(math.sin(error), math.cos(error)))
        if self._phase == "exit":
            return TransitTarget("exit", target=self.plan.exit)
        return TransitTarget(self._phase, hold=True, completed=self.completed)

    def step(
        self, robot: Pose2D, floor: Optional[str], z: float, depth: np.ndarray, t: float,
    ) -> Optional[TransitTarget]:
        """Advance one phase using public pose, floor, odometry height and live depth.

        Invalid observations hold active motion. Completion consumes no route here;
        the owner may consume the planned prefix only after completed becomes True.
        """
        self.stats["steps"] += 1
        if self.completed:
            return TransitTarget("complete", hold=True, completed=True)
        if not math.isfinite(t) or not math.isfinite(z) or not _finite_pose(robot):
            self._reset_stability()
            return None if self._phase == "idle" else TransitTarget(self._phase, hold=True)
        if self._last_t is not None and t < self._last_t:
            self._reset_stability()
            return None if self._phase == "idle" else TransitTarget(self._phase, hold=True)
        self._last_t = t
        if self._started_t is not None and not self.completed:
            self.stats["elapsed_s"] = t - self._started_t
        if self._phase == "idle":
            if (floor != self.plan.source_floor
                    or _distance(robot, self.plan.anchor) > ACTIVATE_RADIUS_M):
                return None
            self._started_t = t
            # The route may approach this crossing long after episode initialization.
            self._start_z = float(z)
            self._transition("board", t)
        elif self._phase == "board":
            if _distance(robot, self.plan.boarding) <= TARGET_RADIUS_M:
                self._transition("turn", t)
        elif self._phase == "turn":
            if abs(self._target(robot).yaw_error) <= TURN_TOLERANCE_RAD:
                self._transition("wait", t)
        elif self._phase == "wait":
            stable = self._stable(floor, z, t)
            error = self.plan.exit.yaw - robot.yaw
            yaw_error = math.atan2(math.sin(error), math.cos(error))
            if abs(yaw_error) > TURN_TOLERANCE_RAD:
                # Keep the depth guard facing the exit as heading drifts during transport.
                return TransitTarget("wait", yaw_error=yaw_error)
            if stable and _front_clear(depth):
                self._transition("exit", t)
        elif self._phase == "exit":
            if floor != self.plan.target_floor:
                return TransitTarget("exit", hold=True)
            if (self._stable_min_z is None or self._stable_max_z is None
                    or max(self._stable_max_z, z) - min(self._stable_min_z, z)
                    > STABLE_HEIGHT_RANGE_M):
                self._reset_stability()
                self._transition("wait", t)
                self._stable(floor, z, t)
                return TransitTarget("wait", hold=True)
            if _distance(robot, self.plan.exit) <= TARGET_RADIUS_M:
                self._transition("complete", t)
        return self._target(robot)

    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible checkpoint state; the owner must rederive the plan."""
        return {
            "version": CHECKPOINT_VERSION, "phase": self._phase, "start_z": self._start_z,
            "plan_fingerprint": self._plan_fingerprint,
            "started_t": self._started_t, "last_t": self._last_t,
            "stable_since": self._stable_since, "stable_min_z": self._stable_min_z,
            "stable_max_z": self._stable_max_z, "stats": copy.deepcopy(self.stats),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore checkpoint state against the separately rederived public-map plan.

        Raises:
            ValueError: The checkpoint version or phase is unsupported.
        """
        if state.get("version") != CHECKPOINT_VERSION or state.get("phase") not in PHASES:
            raise ValueError("Unsupported transit checkpoint")
        if state.get("plan_fingerprint") != self._plan_fingerprint:
            raise ValueError("Transit checkpoint does not match the rederived plan")
        self._phase = state["phase"]
        self._start_z = float(state["start_z"])
        self._started_t = state["started_t"]
        self._last_t = state["last_t"]
        self._stable_since = state["stable_since"]
        self._stable_min_z = state["stable_min_z"]
        self._stable_max_z = state["stable_max_z"]
        self.stats = copy.deepcopy(state["stats"])
