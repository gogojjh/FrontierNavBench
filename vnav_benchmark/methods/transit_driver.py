"""Execute depth-based elevator candidates from permitted method inputs, without scene/evaluator
access.
"""

import math
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Sequence

from vnav_benchmark.fairness import WZ_MAX
from vnav_benchmark.interfaces import MethodContext, Observation, Pose2D
from vnav_benchmark.methods.base_topo import ContLocTracker
from vnav_benchmark.methods.elevator_transit import ElevatorTransit, plan_transit
from vnav_benchmark.methods.nav_template import Decision
from vnav_benchmark.methods.sru.client import PointGoalController, SruSidecarController

HOLD_S = 0.2  # SRU control cadence is 5 Hz; waiting also consumes episode budget
TURN_MIN_RAD_S = 0.5  # Match ordinary alignment: below this yaw rate the walking policy barely
# rotates
TURN_GAIN = 1.5  # Proportional yaw-error gain
METHOD_REVISION = "depth-transfer-v2"  # v1 was tested on two runs; v2 adds waiting-yaw retention
# and requires separate simulation tests


class TransitDriver:
    """Own transfer execution and an optional lazily created point-goal controller."""

    def __init__(
        self, transit: ElevatorTransit, tracker: ContLocTracker,
        on_complete: Callable[[Sequence[int]], None],
        controller: Optional[PointGoalController], factory: Callable[[], PointGoalController],
    ) -> None:
        self.transit = transit
        self._tracker = tracker
        self._on_complete = on_complete
        self._controller = controller
        self._factory = factory
        self._owned = controller is None
        self._controller_phase: Optional[str] = None
        self._resumed = False
        self._controller_s = 0.0
        self._takeover_decisions = 0

    @classmethod
    def create(
        cls, ctx: MethodContext, route_ids: Sequence[int], start: Pose2D,
        obs: Observation, on_complete: Callable[[Sequence[int]], None],
        controller: Optional[PointGoalController] = None,
    ) -> Optional["TransitDriver"]:
        """Derive a candidate from existing map data without another localization call."""
        if ctx.map is None:
            return None
        plan = plan_transit(ctx.map, route_ids, start, obs.K)
        if plan is None:
            return None
        return cls(
            ElevatorTransit(plan, obs.odom.position_xyz[2]),
            ContLocTracker(start, obs.odom, map_handle=ctx.map), on_complete, controller,
            lambda: SruSidecarController(ctx.artifact_dir / "transit_sru" / "sidecar.log"),
        )

    def act(self, obs: Observation) -> Optional[Decision]:
        """Pause normal progress while executing; return None to resume the original method."""
        if self._resumed:
            return None
        robot = self._tracker.to_map(obs.odom)
        step = self.transit.step(
            robot, self._tracker.floor(obs.odom), obs.odom.position_xyz[2],
            obs.depth_m, obs.sim_time_s,
        )
        if step is None:
            return None
        if step.completed:
            self._on_complete(self.transit.plan.consumed_kfs)
            self._resumed = True
            if self._controller is not None:
                if self._owned:
                    self.close()
                else:
                    self._controller.reset()
            return None

        self._takeover_decisions += 1
        vx = vy = wz = controller_s = 0.0
        if step.target is not None and not step.hold:
            if self._controller is None:
                self._controller = self._factory()
            if self._controller_phase != step.phase:
                self._controller.reset()
            goal = self._tracker.map_to_odom(step.target)
            cmd = self._controller.step(obs.depth_m, obs.odom, (goal.x, goal.y))
            vx, vy, wz, controller_s = cmd.vx, cmd.vy, cmd.wz, cmd.controller_s
            self._controller_s += controller_s
        elif not step.hold and step.yaw_error:
            rate = min(WZ_MAX, max(TURN_MIN_RAD_S, TURN_GAIN * abs(step.yaw_error)))
            wz = math.copysign(rate, step.yaw_error)
        self._controller_phase = step.phase
        phase = "elevator_" + step.phase
        row = {
            "step": obs.step_index, "t": obs.sim_time_s, "phase": phase,
            "method_revision": METHOD_REVISION, "robot_map": list(robot),
            "target_map": None if step.target is None else list(step.target),
            "odom_z": obs.odom.position_xyz[2], "cmd": [vx, vy, wz], "done": False,
        }
        return Decision(
            vx, vy, wz, HOLD_S, note=phase, log=row,
            debug={"phase": phase, "method_revision": METHOD_REVISION,
                   "latency": {"controller_s": controller_s}},
        )

    def summary(self) -> Dict[str, Any]:
        """Record the method revision, evidence and takeover separately from normal decisions."""
        return {"elevator_transit": {
            "method_revision": METHOD_REVISION, "plan": asdict(self.transit.plan),
            "stats": self.transit.stats, "resumed": self._resumed,
            "takeover_decisions": self._takeover_decisions,
            "control_mode": "depth_odom_sru_facility_hybrid",
            "controller_s_total": round(self._controller_s, 4),
        }}

    def state_dict(self) -> Dict[str, Any]:
        return {"transit": self.transit.state_dict(), "resumed": self._resumed,
                "controller_s": self._controller_s,
                "takeover_decisions": self._takeover_decisions}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.transit.load_state_dict(state["transit"])
        self._resumed = bool(state["resumed"])
        self._controller_s = float(state["controller_s"])
        self._takeover_decisions = int(state["takeover_decisions"])
        if self._resumed:
            # Pose feeds are rebuilt at reset; the covered prefix must stay covered on restore.
            self._on_complete(self.transit.plan.consumed_kfs)

    def close(self) -> None:
        """Close only the sidecar created by this driver, never the method's borrowed SRU."""
        if self._owned and self._controller is not None:
            self._controller.close()
            self._controller = None
