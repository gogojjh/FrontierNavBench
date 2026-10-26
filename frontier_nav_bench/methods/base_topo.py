"""Shared topological navigation components for methods with maps.

- :class:`InitLocProcedure` localizes the robot once at startup. The v1 ideal
  localizer succeeds immediately; turning until real localization succeeds is
  reserved for v2. Image-based methods M2/M3 also use this procedure.
- :class:`ContLocTracker` combines startup localization with per-frame odometry
  to transform episode-relative poses into the map frame.
- :class:`TopoNavCore` localizes the goal image, selects its nearest keyframe and
  plans with ``topo.TopoPlanner``. Body poses advance subgoals; stalls can block
  the current edge and trigger replanning. After the route, approach the final
  goal pose, align in place and declare done within internal pose tolerances.
- :class:`TopoSubgoalFeed` exposes planned keyframe images and a subgoal index.

The map frame is the task frame in keyframe ``poses.txt``; the odometry frame
starts at the episode spawn pose.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from frontier_nav_bench.interfaces import (
    EpisodeSpec,
    Localizer,
    LocResult,
    MapHandle,
    Observation,
    OdomState,
    Pose2D,
)
from frontier_nav_bench.topo import (
    NoPathError,
    Route,
    TopoGraph,
    TopoGraphError,
    TopoPlanner,
    dijkstra,
)

# ---- Subgoal progression and arrival (plan section 6.6, D7) ----
REACH_M = 1.0  # Advance when the body is within 1.0 m of the current subgoal keyframe
FINAL_POS_TOL_M = 0.5  # Internal arrival distance, matching the 0.5 m evaluation threshold
FINAL_YAW_TOL_RAD = math.radians(30.0)  # Internal arrival yaw tolerance; evaluation allows 45
# degrees
ALIGN_EXIT_M = FINAL_POS_TOL_M + 0.15  # Leave alignment only beyond this distance, avoiding
# repeated phase switching

# ---- Stall handling ----
STALL_WINDOW_S = 10.0  # Simulation window without STALL_PROGRESS_M improvement toward the current
# goal
STALL_PROGRESS_M = 0.3
MAX_REPLANS = 5  # Maximum blocked-edge replans; afterward keep pursuing the subgoal without
# abandoning the map

class LocalizationFailed(RuntimeError):
    """Localization returned ok=False; unexpected in v1 ideal localization except quota/wiring
    errors.
    """


# ---------------------------------------------------------------- Planar pose helpers


def wrap_angle(angle: float) -> float:
    """Normalize an angle to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """Compose a with pose b expressed in frame a, returning b in the parent frame of a."""
    cos_a, sin_a = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(
        a.x + cos_a * b.x - sin_a * b.y,
        a.y + sin_a * b.x + cos_a * b.y,
        wrap_angle(a.yaw + b.yaw),
    )


def inverse(a: Pose2D) -> Pose2D:
    cos_a, sin_a = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(-cos_a * a.x - sin_a * a.y, sin_a * a.x - cos_a * a.y, -a.yaw)


def distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def floors_match(robot_floor: Optional[str], goal_floor: Optional[str]) -> bool:
    """Unknown floors preserve the legacy planar arrival rule."""
    return not robot_floor or not goal_floor or robot_floor == goal_floor


def pose_goal_reached(
    robot_map: Pose2D, goal_map: Pose2D, *,
    robot_floor: Optional[str] = None, goal_floor: Optional[str] = None,
) -> bool:
    """Shared B0/M1/M2/M3 arrival check: distance at most 0.5 m and yaw error at most 30 degrees."""
    return (
        floors_match(robot_floor, goal_floor)
        and distance(robot_map, goal_map) <= FINAL_POS_TOL_M
        and abs(wrap_angle(goal_map.yaw - robot_map.yaw)) <= FINAL_YAW_TOL_RAD
    )


# ---------------------------------------------------------------- Localization


def _require_ok(result: LocResult, what: str) -> Pose2D:
    if not result.ok or result.pose_map is None:
        raise LocalizationFailed(f"{what}定位失败: {result.reason}")
    return result.pose_map


class InitLocProcedure:
    """Localize the robot once at startup; the v1 ideal localizer succeeds immediately."""

    def __init__(self, localizer: Localizer) -> None:
        self._localizer = localizer
        self.floor: Optional[str] = None

    def localize_start(self, obs: Observation) -> Pose2D:
        """Return the current body pose in the map frame.

        Raises:
            LocalizationFailed: The localizer returned ok=False.
        """
        result = self._localizer.localize(obs.rgb_jpeg, obs.depth_m, obs.K, purpose="self")
        self.floor = result.floor
        return _require_ok(result, "开头自")


def localize_goal(localizer: Localizer, spec: EpisodeSpec) -> Pose2D:
    """Localize the body pose when the goal image was captured, with one call per episode."""
    return _require_ok(localize_goal_result(localizer, spec), "目标照片")


def localize_goal_result(localizer: Localizer, spec: EpisodeSpec) -> LocResult:
    """Localize once and retain the floor metadata for pose-driven methods."""
    result = localizer.localize(spec.goal_rgb_jpeg, spec.goal_depth_m, spec.goal_K, purpose="goal")
    _require_ok(result, "目标照片")
    return result


class ContLocTracker:
    """Track map pose as T_map_odom composed with odometry; fix T_map_odom at startup."""

    def __init__(
        self, start_map: Pose2D, start_odom: OdomState, *,
        map_handle: Optional[MapHandle] = None,
    ) -> None:
        self.map_from_odom = compose(start_map, inverse(start_odom.pose))
        self._map = map_handle

    def floor(self, odom: OdomState) -> Optional[str]:
        """OdomOrigin resets XY/yaw only, so odometry retains world height."""
        return self._map.floor_at_z(odom.position_xyz[2]) if self._map is not None else None

    def to_map(self, odom: OdomState) -> Pose2D:
        return compose(self.map_from_odom, odom.pose)

    def map_to_odom(self, pose_map: Pose2D) -> Pose2D:
        """Transform a map pose into odometry coordinates for a point-goal controller."""
        return compose(inverse(self.map_from_odom), pose_map)


# ---------------------------------------------------------------- Navigation core

PHASE_SUBGOAL = "subgoal"  # Follow the route toward a keyframe
PHASE_FINAL = "final"  # Route complete or degraded; approach the final goal pose directly
PHASE_ALIGN = "align"  # Near the goal; align yaw in place
PHASE_DONE = "done"  # Position and yaw meet internal tolerances


@dataclass(frozen=True)
class NavTarget:
    """Target for this frame; pose_map yaw matters only during align/done."""

    phase: str
    pose_map: Pose2D
    kf_id: Optional[int]
    dist_m: float  # Planar body distance to pose_map
    yaw_err_rad: float  # Target yaw minus body yaw; meaningful only during align/done


class TopoNavCore:
    """Shared B0/M1/M2/M3 episode state machine that supplies targets, not velocities.

    1. Plan with Dijkstra at startup. ``self._remaining`` stores unvisited
       keyframes with the current subgoal first.
    2. Each frame, find the nearest remaining keyframe k; within reach, choose
       its successor. Remove all preceding keyframes as visited.
    3. On subgoal changes, rerun Dijkstra from the current body pose and replace
       the remaining sequence. Remove initial points within reach and visited
       keyframes to avoid backtracking.
    4. After 10 s without 0.3 m progress, test whether blocking the preceding
       route edge preserves a route. If so, block/replan up to five times.
       Otherwise retain the edge, pursue the subgoal and restart stall timing.
       Keep pursuing after the replan limit; only a planning failure degrades
       to direct goal pursuit, without voluntarily abandoning the map.
    5. After the remaining sequence, approach the final pose; align within
       0.5 m and finish when :func:`pose_goal_reached` is true.
    """

    def __init__(self, map_handle: MapHandle, reach_m: float = REACH_M) -> None:
        self._map = map_handle
        self._graph = map_handle.topo_graph()
        self._planner = TopoPlanner(self._graph)
        self._reach_m = reach_m
        self._goal: Optional[Pose2D] = None
        self._goal_floor: Optional[str] = None
        self._goal_kf: Optional[int] = None
        self._aligning = False
        self._degraded = False
        self._remaining: List[int] = []  # Unvisited keyframes; index zero is the current subgoal
        self._visited: List[int] = []  # Removed/visited keyframes in traversal order
        self._last_kf: Optional[int] = None
        self._stall_ref: Optional[Tuple[Optional[int], float, float]] = None  # (keyframe ID,
        # closest distance, timestamp)
        self.stats: Dict[str, int] = {
            "replans": 0,  # Stall-triggered blocked-edge replans, with a limit
            "replans_on_switch": 0,  # Replans from the current pose after subgoal changes
            "subgoal_advances": 0,
            "blocked_edges": 0,
            "degraded": 0,
            "degraded_no_route": 0,  # Planning failures that degraded to direct goal pursuit
            "stall_kept_no_alt": 0,  # Stalls where blocking would disconnect the route, so pursuit
            # continues
            "stall_kept_after_cap": 0,  # Stalls after the blocked-edge replan limit, with continued
            # pursuit
        }

    @property
    def route(self) -> Optional[Route]:
        """Latest full Dijkstra route, including removed initial points; see remaining for pending
        points.
        """
        return self._planner.current_route

    @property
    def remaining(self) -> Tuple[int, ...]:
        return tuple(self._remaining)

    @property
    def visited(self) -> Tuple[int, ...]:
        return tuple(self._visited)

    @property
    def goal(self) -> Optional[Pose2D]:
        return self._goal

    @property
    def goal_floor(self) -> Optional[str]:
        return self._goal_floor

    @property
    def route_version(self) -> int:
        """Route revision count for stall/subgoal replans, used to detect reordered pose-based
        sequences.
        """
        return self.stats["replans"] + self.stats["replans_on_switch"]

    def start(
        self, robot_map: Pose2D, goal_map: Pose2D, *,
        robot_floor: Optional[str] = None, goal_floor: Optional[str] = None,
    ) -> Route:
        """Plan from the nearest current-body keyframe to the keyframe nearest the goal pose."""
        self._goal = goal_map
        self._goal_floor = goal_floor
        self._goal_kf = self._nearest(goal_map, goal_floor)
        route = self._planner.plan(
            robot_map, self._goal_kf, start_kf=self._nearest(robot_map, robot_floor)
        )
        self._remaining = list(route.kf_path)
        self._visited = []
        self._last_kf = None
        return route

    def _target(self, phase: str, pose: Pose2D, kf: Optional[int], robot: Pose2D) -> NavTarget:
        return NavTarget(phase, pose, kf, distance(robot, pose), wrap_angle(pose.yaw - robot.yaw))

    def consume_prefix(self, completed: Sequence[int]) -> None:
        """Resume after a method-level facility segment without changing normal reach rules."""
        covered = set(completed)
        while self._remaining and self._remaining[0] in covered:
            kf = self._remaining.pop(0)
            if kf not in self._visited:
                self._visited.append(kf)
        self._last_kf = None
        self._stall_ref = None

    def _nearest(self, pose: Pose2D, floor: Optional[str]) -> int:
        if not floor:
            return self._graph.nearest(pose.x, pose.y)
        candidates = [kf for kf in self._graph.poses
                      if floors_match(floor, self._map.keyframe_floor(kf))]
        if not candidates:
            raise TopoGraphError(f"楼层 {floor!r} 没有关键帧")
        return min(candidates, key=lambda kf: self._dist_kf(pose, kf, floor))

    def _dist_kf(self, robot: Pose2D, kf: int, floor: Optional[str] = None) -> float:
        if floor and not floors_match(floor, self._map.keyframe_floor(kf)):
            return math.inf
        pose = self._graph.poses[kf]
        return math.hypot(robot.x - pose.x, robot.y - pose.y)

    def _adopt(self, route: Route, robot: Pose2D, floor: Optional[str]) -> None:
        """Replace pending keyframes, dropping visited points and initial points within reach."""
        path = list(route.kf_path)
        visited = set(self._visited)
        while path and (path[0] in visited or self._dist_kf(robot, path[0], floor) < self._reach_m):
            path.pop(0)
        self._remaining = path

    def _replan(
        self, robot: Pose2D, block: Optional[Tuple[int, int]], floor: Optional[str] = None
    ) -> bool:
        """Replan from the current pose, optionally blocking an edge; return success or pursue the
        goal.
        """
        try:
            route = self._planner.replan(
                robot, block_edge=block, start_kf=self._nearest(robot, floor)
            )
        except TopoGraphError:
            self.stats["degraded_no_route"] += 1
            self._degrade()
            return False
        self._adopt(route, robot, floor)
        return True

    def _has_route_without(
        self, robot: Pose2D, block: Tuple[int, int], floor: Optional[str]
    ) -> bool:
        """Test reachability after blocking an edge without modifying planner state."""
        if self._goal_kf is None:
            return False
        blocked = set(self._planner.blocked) | {(min(block), max(block))}
        try:
            dijkstra(self._graph, self._nearest(robot, floor), self._goal_kf, blocked)
        except NoPathError:
            return False
        return True

    def _degrade(self) -> None:
        if not self._degraded:
            self._degraded = True
            self.stats["degraded"] = 1

    def _select(self, robot: Pose2D, floor: Optional[str]) -> Optional[int]:
        """Select the nearest pending keyframe or its successor within reach, removing preceding
        points.
        """
        if not self._remaining:
            return None
        # Do not skip an unvisited floor transition to a later same-floor node.
        candidates = []
        for kf in self._remaining:
            if floor and not floors_match(floor, self._map.keyframe_floor(kf)):
                break
            candidates.append(kf)
        if not candidates:
            # If all remaining candidates are on other floors, still permit one distance-based
            # advance.
            # Treat remaining[0] as reached when close enough; floor tracking catches up while
            # moving.
            # Advance one point without skipping intermediates, matching normal floor-filtered
            # progression.
            target = self._remaining[0]
            dist = self._dist_kf(robot, target, floor=None)  # Use planar distance without filtering
            # by floor
            i = 1 if dist < self._reach_m else 0
            self._visited.extend(self._remaining[:i])
            self._remaining = self._remaining[i:]
            return self._remaining[0] if self._remaining else None
        dists = [self._dist_kf(robot, kf, floor) for kf in candidates]
        i = min(range(len(dists)), key=dists.__getitem__)
        if dists[i] < self._reach_m:
            i += 1
        self._visited.extend(self._remaining[:i])
        self._remaining = self._remaining[i:]
        return self._remaining[0] if self._remaining else None

    def _stalled(self, kf: Optional[int], dist_m: float, sim_time_s: float) -> bool:
        ref = self._stall_ref
        if ref is None or ref[0] != kf or dist_m < ref[1] - STALL_PROGRESS_M:
            self._stall_ref = (kf, dist_m, sim_time_s)
            return False
        return sim_time_s - ref[2] >= STALL_WINDOW_S

    def update(
        self, robot_map: Pose2D, sim_time_s: float, *, robot_floor: Optional[str] = None
    ) -> NavTarget:
        """Advance state using the current map-frame body pose and return this frame target."""
        if self._goal is None:
            raise RuntimeError("先调 start()")
        goal = self._goal
        dist_goal = distance(robot_map, goal)
        if (floors_match(robot_floor, self._goal_floor)
                and dist_goal <= (ALIGN_EXIT_M if self._aligning else FINAL_POS_TOL_M)):
            self._aligning = True
            if pose_goal_reached(
                robot_map, goal, robot_floor=robot_floor, goal_floor=self._goal_floor
            ):
                return self._target(PHASE_DONE, goal, None, robot_map)
            return self._target(PHASE_ALIGN, goal, None, robot_map)
        self._aligning = False
        if self._degraded:
            return self._target(PHASE_FINAL, goal, None, robot_map)

        kf = self._select(robot_map, robot_floor)
        if kf is not None and kf != self._last_kf:
            if self._last_kf is not None:
                self.stats["subgoal_advances"] += 1
                self.stats["replans_on_switch"] += 1  # Rule 3: replan whenever the subgoal changes
                if self._replan(robot_map, None, robot_floor):
                    kf = self._remaining[0] if self._remaining else None
            self._last_kf = kf
            self._stall_ref = None
        if kf is None:
            return self._target(PHASE_FINAL, goal, None, robot_map)
        pose = self._graph.poses[kf]
        target = self._target(PHASE_SUBGOAL, Pose2D(pose.x, pose.y, pose.yaw), kf, robot_map)
        if self._stalled(kf, target.dist_m, sim_time_s):
            self._stall_ref = None  # Restart stall timing regardless of whether an edge was blocked
            if self.stats["replans"] >= MAX_REPLANS:
                self.stats["stall_kept_after_cap"] += 1
                return target
            path = self.route.kf_path if self.route is not None else ()
            at = path.index(kf) if kf in path else 0
            block = (path[at - 1], kf) if at > 0 else None  # Route edge from the immediate
            # predecessor to the subgoal
            if block is None or not self._has_route_without(robot_map, block, robot_floor):
                self.stats["stall_kept_no_alt"] += 1
                return target
            self.stats["replans"] += 1
            self.stats["blocked_edges"] += 1
            self._replan(robot_map, block, robot_floor)
            self._last_kf = self._remaining[0] if self._remaining else None
            return self.update(robot_map, sim_time_s, robot_floor=robot_floor)
        return target


def route_length_from(robot_map: Pose2D, goal_map: Pose2D, route: Route, graph: TopoGraph) -> float:
    """Total route length: body to first keyframe, graph edges, then final keyframe to goal pose."""
    first = graph.poses[route.kf_path[0]]
    last = graph.poses[route.kf_path[-1]]
    return (
        math.hypot(first.x - robot_map.x, first.y - robot_map.y)
        + route.length_m
        + math.hypot(goal_map.x - last.x, goal_map.y - last.y)
    )


# ---------------------------------------------------------------- Subgoal image sequence


@dataclass(frozen=True)
class Subgoal:
    index: int  # Zero-based position in kf_ids
    kf_id: int
    rgb_jpeg: bytes  # Image for this subgoal keyframe
    is_last: bool  # Last route image; only the final goal image remains afterward
    route_version: int = 0  # Pose-based route revision; changes indicate the subgoal sequence was
    # reordered


class TopoSubgoalFeed:
    """Planned keyframe-image sequence; VLM decisions advance its current image index.

    Use the startup route without later localization/replanning in initloc mode.
    Skip the starting keyframe if it is within ``skip_within_m`` of the body.
    """

    def __init__(
        self, map_handle: MapHandle, route: Route, start_map: Pose2D, skip_within_m: float = REACH_M
    ) -> None:
        graph = map_handle.topo_graph()
        kf_ids: List[int] = list(route.kf_path)
        while len(kf_ids) > 1:
            pose = graph.poses[kf_ids[0]]
            if math.hypot(pose.x - start_map.x, pose.y - start_map.y) >= skip_within_m:
                break
            kf_ids.pop(0)
        self._map = map_handle
        self.route = route
        self.kf_ids: Tuple[int, ...] = tuple(kf_ids)
        self._index = 0

    @classmethod
    def plan(cls, map_handle: MapHandle, start_map: Pose2D, goal_map: Pose2D) -> "TopoSubgoalFeed":
        """Plan from the body pose to the goal and construct the image sequence; see .route."""
        route = TopoNavCore(map_handle).start(start_map, goal_map)
        return cls(map_handle, route, start_map)

    @property
    def index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        """All subgoals have been traversed; only the final goal image remains."""
        return self._index >= len(self.kf_ids)

    def current(self) -> Optional[Subgoal]:
        if self.finished:
            return None
        kf = self.kf_ids[self._index]
        return Subgoal(
            self._index, kf, self._map.keyframe_rgb_jpeg(kf), self._index == len(self.kf_ids) - 1
        )

    def peek(self, ahead: int = 1) -> Optional[Subgoal]:
        """Look ahead by ahead images without advancing; return None beyond the sequence."""
        index = self._index + ahead
        if index >= len(self.kf_ids):
            return None
        kf = self.kf_ids[index]
        return Subgoal(index, kf, self._map.keyframe_rgb_jpeg(kf), index == len(self.kf_ids) - 1)

    def advance(self) -> Optional[Subgoal]:
        """Advance after the VLM reports arrival; return the new current subgoal or None at the end.
        """
        if not self.finished:
            self._index += 1
        return self.current()

    def state_dict(self) -> Dict[str, object]:
        return {"kf_ids": list(self.kf_ids), "index": self._index}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if list(state["kf_ids"]) != list(self.kf_ids):  # type: ignore[call-overload]
            raise ValueError("存档的路线和当前路线不一致")
        self._index = int(state["index"])  # type: ignore[call-overload]


class PoseSubgoalFeed:
    """Pose-driven subgoal images for M2/M3 ``*-contloc-imggoal`` methods.

    Share :class:`TopoNavCore` and its progression rules with B0/M1. Map poses
    advance subgoals each frame; a 10 s stall may block/replan an edge, and each
    subgoal change replans from the current pose. Increment Subgoal.route_version
    when reordering the sequence. After the route, or near the final pose,
    current_subgoal_image returns None so only the final goal image remains.
    phase distinguishes final, align and done.
    """

    def __init__(
        self, map_handle: MapHandle, start_map: Pose2D, goal_map: Pose2D, reach_m: float = REACH_M,
        *, start_floor: Optional[str] = None, goal_floor: Optional[str] = None,
    ) -> None:
        self._map = map_handle
        self.core = TopoNavCore(map_handle, reach_m)
        self.core.start(start_map, goal_map, robot_floor=start_floor, goal_floor=goal_floor)
        self.target: Optional[NavTarget] = None

    @property
    def phase(self) -> str:
        """Phase after the latest current_subgoal_image call: subgoal / final / align / done."""
        return self.target.phase if self.target is not None else PHASE_SUBGOAL

    @property
    def route(self) -> Optional[Route]:
        return self.core.route

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self.core.stats)

    def pose_goal_reached(self, pose_map: Pose2D, *, floor: Optional[str] = None) -> bool:
        """Check pose-based goal arrival using the same pose_goal_reached criterion as B0."""
        goal = self.core.goal
        return goal is not None and pose_goal_reached(
            pose_map, goal, robot_floor=floor, goal_floor=self.core.goal_floor
        )

    def current_subgoal_image(
        self, pose_map: Pose2D, sim_time_s: float, *, floor: Optional[str] = None
    ) -> Optional[Subgoal]:
        """Advance using the current map-frame body pose and return the subgoal image, or None."""
        self.target = self.core.update(pose_map, sim_time_s, robot_floor=floor)
        if self.target.phase != PHASE_SUBGOAL or self.target.kf_id is None:
            return None
        kf = self.target.kf_id
        return Subgoal(
            len(self.core.visited),  # Number of visited keyframes
            kf,
            self._map.keyframe_rgb_jpeg(kf),
            len(self.core.remaining) == 1,
            self.core.route_version,
        )
