"""B0/M1: dense/sparse topology, continuous localization and the SRU point-goal controller.

Both inherit :class:`~frontier_nav_bench.methods.nav_template.NavMethod` and follow
plan section 6.6:
1. setup localizes self and the goal image once, plans a graph route and starts
   the SRU sidecar.
2. Each act holds for 0.2 s, matching SRU at 5 Hz. subgoal propagates odometry to
   the map frame, then TopoNavCore advances within 1.0 m or handles stalls by
   testing alternate routes before blocking/replanning. decide transforms the
   subgoal/final target into odometry coordinates and asks SRU for velocity.
3. Within 0.5 m, align in place at least 0.5 rad/s because the walking policy
   barely turns more slowly. system_done_decision declares done within the
   internal 0.5 m / 30 degree pose tolerances.

B0/M1 differ only in the map selected by the first method-name field. Artifacts
under ctx.artifact_dir include route.json (initial plan), steps.jsonl (phase,
subgoal, velocity, latency), summary.json and sru/sidecar.log.
"""

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from frontier_nav_bench.fairness import WZ_MAX, parse_method_id
from frontier_nav_bench.interfaces import EpisodeSpec, MethodContext, Observation, Pose2D
from frontier_nav_bench.methods.base_topo import (
    PHASE_ALIGN,
    PHASE_DONE,
    ContLocTracker,
    InitLocProcedure,
    NavTarget,
    TopoNavCore,
    localize_goal_result,
    route_length_from,
)
from frontier_nav_bench.methods.nav_template import Decision, NavMethod, SubgoalInfo
from frontier_nav_bench.methods.sru.client import PointGoalController, SruSidecarController

HOLD_S = 0.2  # Hold each decision for 0.2 simulation seconds, matching SRU training/deployment at 5
# Hz
ALIGN_WZ_MIN = 0.5  # Minimum alignment yaw rate; MoE-CTS barely turns at <= 0.3 rad/s (plan section
# 4-16)
ALIGN_GAIN = 1.5  # Alignment yaw rate = gain * yaw error, clamped to [ALIGN_WZ_MIN, WZ_MAX]
CONTROLLERS = ("sru_sidecar",)  # Configurable point-goal controller; future ROS variants can be
# added here

ControllerFactory = Callable[[Path], PointGoalController]


def _pose_json(pose: Pose2D) -> List[float]:
    return [round(pose.x, 4), round(pose.y, 4), round(pose.yaw, 4)]


class PtGoalSruMethod(NavMethod):
    """B0/M1 method instance, newly constructed for every episode (plan section 6.8)."""

    uses_truth = False
    LOG_ENSURE_ASCII = True  # Preserve steps.jsonl / summary.json formats from before template
    # extraction
    SUMMARY_JSON_KW = {"indent": 1}

    def __init__(
        self, method_id: str, controller_factory: Optional[ControllerFactory] = None
    ) -> None:
        profile = parse_method_id(method_id)
        if profile.planner != "ptgoal" or profile.loc_mode != "contloc":
            raise ValueError(f"{method_id} 不是 *-contloc-ptgoal")
        self.method_id = method_id
        self._controller_factory = controller_factory or SruSidecarController
        self._controller: Optional[PointGoalController] = None
        self._core: Optional[TopoNavCore] = None
        self._dir: Optional[Path] = None
        self._summary: Dict[str, Any] = {}

    def setup(self, spec: EpisodeSpec, obs: Observation, ctx: MethodContext) -> None:
        if ctx.map is None or ctx.localizer is None:
            raise ValueError(f"{self.method_id} 需要地图和定位器")
        controller_name = str(ctx.config.get("controller", CONTROLLERS[0]))
        if controller_name not in CONTROLLERS:
            raise ValueError(f"未知的点目标规划器: {controller_name!r}")
        self._dir = Path(ctx.artifact_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.steps_log_path = self._dir / "steps.jsonl"
        self.summary_path = self._dir / "summary.json"
        initial = InitLocProcedure(ctx.localizer)
        start = initial.localize_start(obs)
        self._tracker = ContLocTracker(start, obs.odom, map_handle=ctx.map)
        goal_loc = localize_goal_result(ctx.localizer, spec)
        goal = goal_loc.pose_map
        assert goal is not None
        self._core = TopoNavCore(ctx.map)
        route = self._core.start(start, goal, robot_floor=initial.floor, goal_floor=goal_loc.floor)
        planned_m = route_length_from(start, goal, route, ctx.map.topo_graph())
        (self._dir / "route.json").write_text(
            json.dumps(
                {
                    "map_id": ctx.map.map_id,
                    "edge_type": ctx.map.edge_type,
                    "start_map": _pose_json(start),
                    "goal_map": _pose_json(goal),
                    "kf_path": list(route.kf_path),
                    "route_length_m": round(route.length_m, 3),
                    "planned_total_m": round(planned_m, 3),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        self._summary = {
            "method_id": self.method_id,
            "controller": controller_name,
            "route_kfs": len(route.kf_path),
            "planned_total_m": round(planned_m, 3),
            "decisions": 0,
            "align_decisions": 0,
            "sru_arrived_steps": 0,
            "controller_s_total": 0.0,
            "termination": None,
        }
        self._controller = self._controller_factory(self._dir / "sru" / "sidecar.log")
        self._controller.reset()
        self.configure_transit(ctx, route.kf_path, start, obs, self._controller)

    def transit_completed(self, consumed: Sequence[int]) -> None:
        assert self._core is not None
        self._core.consume_prefix(consumed)

    def subgoal(self, obs: Observation) -> SubgoalInfo:
        """Propagate odometry to map pose and advance subgoals; set system_done within pose
        tolerances.
        """
        assert self._controller is not None and self._core is not None, "先 reset"
        robot = self._tracker.to_map(obs.odom)
        target = self._core.update(robot, obs.sim_time_s, robot_floor=self._tracker.floor(obs.odom))
        self._summary["decisions"] += 1
        return SubgoalInfo(
            target.phase, system_done=(target.phase == PHASE_DONE), kf_id=target.kf_id,
            pose_map=target.pose_map, extra={"robot": robot, "target": target},
        )  # fmt: skip

    def system_done_decision(self, obs: Observation, sub: SubgoalInfo) -> Decision:
        self._summary["termination"] = "done"
        return self._decision(obs, sub, 0.0, 0.0, 0.0, {"loc_s": 0.0}, "pose within tolerance")

    def decide(self, obs: Observation, sub: SubgoalInfo) -> Decision:
        """Turn in place during alignment; otherwise pass depth and odometry-frame targets to SRU.
        """
        assert self._controller is not None
        target: NavTarget = sub.extra["target"]
        latency: Dict[str, float] = {"loc_s": 0.0}
        if target.phase == PHASE_ALIGN:
            self._summary["align_decisions"] += 1
            rate = min(WZ_MAX, max(ALIGN_WZ_MIN, ALIGN_GAIN * abs(target.yaw_err_rad)))
            wz = math.copysign(rate, target.yaw_err_rad)
            return self._decision(obs, sub, 0.0, 0.0, wz, latency)
        goal_odom = self._tracker.map_to_odom(target.pose_map)
        cmd = self._controller.step(obs.depth_m, obs.odom, (goal_odom.x, goal_odom.y))
        self._summary["sru_arrived_steps"] += int(cmd.arrived)
        self._summary["controller_s_total"] += cmd.controller_s
        latency["controller_s"] = cmd.controller_s
        return self._decision(obs, sub, cmd.vx, cmd.vy, cmd.wz, latency)

    def _decision(
        self, obs: Observation, sub: SubgoalInfo, vx: float, vy: float, wz: float,
        latency: Dict[str, float], note: str = "",
    ) -> Decision:  # fmt: skip
        """Build action, debug fields and a steps.jsonl row with unchanged pre-template fields."""
        robot: Pose2D = sub.extra["robot"]
        target: NavTarget = sub.extra["target"]
        done = target.phase == PHASE_DONE
        row = {
            "step": obs.step_index,
            "t": obs.sim_time_s,
            "phase": target.phase,
            "kf": target.kf_id,
            "robot_map": _pose_json(robot),
            "target_map": _pose_json(target.pose_map),
            "dist_m": round(target.dist_m, 3),
            "cmd": [round(vx, 4), round(vy, 4), round(wz, 4)],
            "done": done,
        }
        return Decision(
            vx, vy, wz, HOLD_S, done=done, note=note or target.phase,
            debug={"latency": latency, "phase": target.phase, "kf": target.kf_id}, log=row,
        )  # fmt: skip

    def summary(self) -> Dict[str, Any]:
        out = dict(self._summary)
        if out and self._core is not None:
            out.update(self._core.stats)
            out["controller_s_total"] = round(out["controller_s_total"], 4)
        out.update(self.transit_summary())
        return out

    def close_backend(self) -> None:
        if self._controller is not None:
            self._controller.close()
            self._controller = None


def make(method_id: str) -> PtGoalSruMethod:
    """Registry factory in methods/__init__.py; construct one method per episode."""
    return PtGoalSruMethod(method_id)
