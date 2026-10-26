"""Depth geometry and observation-only elevator transit contracts."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from frontier_nav_bench.interfaces import Pose2D
from frontier_nav_bench.methods.elevator_transit import ElevatorTransit, plan_transit
from frontier_nav_bench.topo import TopoGraph

K = (80.0, 0.0, 49.5, 0.0, 80.0, 39.5, 0.0, 0.0, 1.0)
START = Pose2D(-3.0, 0.0, 0.0)
BLOCKED = np.full((80, 100), 0.9, dtype=np.float32)
CLEAR = np.full((80, 100), 3.0, dtype=np.float32)


class DepthMap:
    """In-memory public map surface; no simulator or semantic labels."""

    camera_forward_m = 0.15

    def __init__(self):
        self.poses = {
            0: START,
            1: Pose2D(0.0, 0.0, 0.0),
            2: Pose2D(-1.0, 0.0, math.pi),
            3: Pose2D(-4.0, 0.0, math.pi),
        }
        self.floors = {0: "lower", 1: "lower", 2: "upper", 3: "upper"}
        self.depths = {i: CLEAR.copy() for i in self.poses}
        self.depths[1] = BLOCKED.copy()

    def topo_graph(self):
        return TopoGraph(poses=self.poses)

    def keyframe_pose(self, kf_id):
        return self.poses[kf_id]

    def keyframe_floor(self, kf_id):
        return self.floors[kf_id]

    def keyframe_depth_m(self, kf_id):
        return self.depths[kf_id]


def make_plan(map_handle=None, route=(0, 1, 2, 3), start=START, intrinsics=K):
    return plan_transit(map_handle or DepthMap(), route, start, intrinsics)


def waiting_transit(start_z=0.0):
    plan = make_plan()
    transit = ElevatorTransit(plan, start_z=start_z)
    assert transit.step(plan.anchor, plan.source_floor, start_z, BLOCKED, 0.0).phase == "board"
    assert transit.step(plan.boarding, plan.source_floor, start_z, BLOCKED, 0.1).phase == "turn"
    facing_exit = Pose2D(plan.boarding.x, plan.boarding.y, plan.exit.yaw)
    assert transit.step(facing_exit, plan.source_floor, start_z, BLOCKED, 0.2).phase == "wait"
    return transit, plan, facing_exit


def test_candidate_is_derived_from_depth_route_and_calibrated_clearance():
    plan = make_plan()

    assert plan is not None
    assert plan.anchor_kf == 1
    assert plan.boarding == Pose2D(0.4, 0.0, 0.0)
    assert plan.exit.x == pytest.approx(-1.5)
    assert abs(plan.exit.yaw) == pytest.approx(math.pi)
    assert (plan.source_floor, plan.target_floor) == ("lower", "upper")
    assert plan.consumed_kfs == (0, 1)
    assert plan.evidence["boarding_clearance_m"] >= 0.45
    json.dumps(plan.evidence, allow_nan=False)


@pytest.mark.parametrize("case", ["same_floor", "stairs", "noncompact", "missing_floor"])
def test_non_candidates_fail_closed(case):
    handle = DepthMap()
    if case == "same_floor":
        handle.floors = dict.fromkeys(handle.poses, "lower")
    elif case == "stairs":
        handle.poses[2] = Pose2D(1.0, 0.0, 0.0)
        handle.poses[3] = Pose2D(4.0, 0.0, 0.0)
    elif case == "noncompact":
        handle.depths[1] = CLEAR
    else:
        handle.floors[2] = None

    assert make_plan(handle) is None


def test_legacy_map_without_facility_evidence_returns_no_plan():
    class LegacyMap:
        def topo_graph(self):
            return DepthMap().topo_graph()

        def keyframe_pose(self, kf_id):
            return DepthMap().keyframe_pose(kf_id)

    assert make_plan(LegacyMap()) is None


def test_ordinary_route_does_not_read_camera_offset():
    class OrdinaryMap(DepthMap):
        @property
        def camera_forward_m(self):
            raise AssertionError("An ordinary route does not need camera evidence")

    handle = OrdinaryMap()
    handle.floors = dict.fromkeys(handle.poses, "lower")

    assert make_plan(handle) is None


def test_crossing_without_camera_offset_returns_no_plan():
    class NoCameraMap(DepthMap):
        @property
        def camera_forward_m(self):
            raise AttributeError("Camera offset is unavailable")

    assert make_plan(NoCameraMap()) is None


@pytest.mark.parametrize("invalid", [(), (0.0,) * 9, (float("nan"),) * 9])
def test_missing_or_invalid_intrinsics_fail_closed(invalid):
    assert make_plan(intrinsics=invalid) is None


@pytest.mark.parametrize("depth", [np.zeros((80, 100)), np.full((80, 100), np.nan)])
def test_invalid_candidate_depth_fails_closed(depth):
    handle = DepthMap()
    handle.depths[1] = depth

    assert make_plan(handle) is None


def test_unobserved_boarding_clearance_is_not_invented():
    handle = DepthMap()
    handle.depths[1] = np.full((80, 100), 0.5)

    assert make_plan(handle) is None


def test_missing_approach_or_successor_fails_closed():
    assert make_plan(route=(1, 2, 3), start=Pose2D(0.0, 0.0, 0.0)) is None
    assert make_plan(route=(0, 1)) is None


def test_two_identifiable_crossings_are_ambiguous():
    handle = DepthMap()
    handle.poses.update({4: Pose2D(-1.0, 1.0, 0.0), 5: Pose2D(-4.0, 1.0, math.pi)})
    handle.floors.update({4: "upper", 5: "lower"})
    handle.depths.update({4: BLOCKED, 5: CLEAR})
    handle.poses[6] = Pose2D(-7.0, 1.0, math.pi)
    handle.floors[6] = "lower"
    handle.depths[6] = CLEAR

    assert make_plan(handle, route=(0, 1, 2, 3, 4, 5, 6)) is None


@pytest.mark.parametrize("last_floor", ["lower", None])
def test_second_crossing_or_unknown_floor_anywhere_rejects_the_plan(last_floor):
    handle = DepthMap()
    handle.floors[3] = last_floor
    assert make_plan(handle) is None


def test_map_depth_provider_error_fails_closed():
    class DepthUnavailable(Exception):
        pass

    class UnreadableMap(DepthMap):
        def keyframe_depth_m(self, kf_id):
            raise DepthUnavailable("missing depth image")

    assert make_plan(UnreadableMap()) is None


def test_destination_endpoint_and_reversed_floor_names_are_supported():
    handle = DepthMap()
    handle.floors[1] = "upper"
    handle.poses[1] = Pose2D(0.0, 0.0, math.pi)
    handle.depths[1] = np.full((80, 100), 0.5)

    plan = make_plan(handle)

    assert plan is not None
    assert plan.source_floor == "lower"
    assert plan.target_floor == "upper"
    assert plan.consumed_kfs == (0, 1)
    reversed_handle = DepthMap()
    reversed_handle.floors = {
        i: "upper" if floor == "lower" else "lower"
        for i, floor in reversed_handle.floors.items()
    }
    reverse_plan = make_plan(reversed_handle)
    assert (reverse_plan.source_floor, reverse_plan.target_floor) == ("upper", "lower")


def test_idle_requires_source_floor_and_approach_distance():
    plan = make_plan()
    transit = ElevatorTransit(plan, 0.0)

    assert transit.phase == "idle"
    assert transit.step(START, "lower", 0.0, BLOCKED, 0.0) is None
    assert transit.step(plan.anchor, None, 0.0, BLOCKED, 0.1) is None
    result = transit.step(Pose2D(-1.5, 0.0, 0.0), "lower", 0.0, BLOCKED, 0.2)
    assert result.phase == "board"
    assert result.target == plan.boarding


def test_board_does_not_advance_at_regular_one_meter_tolerance():
    plan = make_plan()
    transit = ElevatorTransit(plan, 0.0)
    transit.step(plan.anchor, "lower", 0.0, BLOCKED, 0.0)

    result = transit.step(Pose2D(-0.6, 0.0, 0.0), "lower", 0.0, BLOCKED, 0.1)

    assert result.phase == "board"
    assert result.target == plan.boarding
    assert not transit.completed


def test_turn_cannot_release_on_early_floor_flip():
    plan = make_plan()
    transit = ElevatorTransit(plan, 0.0)
    transit.step(plan.anchor, "lower", 0.0, BLOCKED, 0.0)

    result = transit.step(plan.boarding, "upper", 3.0, CLEAR, 10.0)

    assert result.phase == "turn"
    assert abs(result.yaw_error) == pytest.approx(math.pi)


def test_midfloor_label_or_stable_pause_does_not_release():
    transit, plan, robot = waiting_transit()
    for t, floor, z in [(0.3, "upper", 0.2), (2.0, "upper", 0.2),
                        (3.0, "lower", 1.0), (5.0, "lower", 1.0)]:
        result = transit.step(robot, floor, z, CLEAR, t)
        assert result.phase == "wait"
        assert result.hold


def test_exit_requires_destination_stable_height_and_front_clearance():
    transit, plan, robot = waiting_transit()
    assert transit.step(robot, "upper", 3.0, BLOCKED, 1.0).phase == "wait"
    assert transit.step(robot, "upper", 3.02, BLOCKED, 2.1).phase == "wait"

    result = transit.step(robot, "upper", 3.01, CLEAR, 2.2)

    assert result.phase == "exit"
    assert result.target == plan.exit
    assert not transit.completed


def test_wait_corrects_yaw_drift_before_releasing_a_stable_clear_exit():
    transit, plan, robot = waiting_transit()
    transit.step(robot, "upper", 3.0, CLEAR, 1.0)
    drifted = Pose2D(robot.x, robot.y, plan.exit.yaw + 0.4)

    correction = transit.step(drifted, "upper", 3.0, CLEAR, 2.1)

    assert correction.phase == "wait"
    assert not correction.hold and correction.target is None
    assert correction.yaw_error == pytest.approx(-0.4)
    assert transit.phase == "wait" and not transit.completed
    corrected = transit.step(robot, "upper", 3.0, CLEAR, 2.2)
    assert corrected.phase == "exit"
    assert corrected.target == plan.exit


def test_wait_yaw_correction_preserves_closed_depth_exit_guard():
    transit, plan, robot = waiting_transit()
    transit.step(robot, "upper", 3.0, CLEAR, 1.0)
    drifted = Pose2D(robot.x, robot.y, plan.exit.yaw - 0.4)

    correction = transit.step(drifted, "upper", 3.0, BLOCKED, 2.1)

    assert correction.phase == "wait" and not correction.hold
    assert correction.target is None and correction.yaw_error == pytest.approx(0.4)
    blocked = transit.step(robot, "upper", 3.0, BLOCKED, 2.2)
    assert blocked.phase == "wait" and blocked.hold and blocked.target is None
    assert transit.step(robot, "upper", 3.0, CLEAR, 2.3).phase == "exit"


def test_vertical_drift_restarts_stability_window():
    transit, _, robot = waiting_transit()
    assert transit.step(robot, "upper", 1.0, CLEAR, 1.0).phase == "wait"
    assert transit.step(robot, "upper", 1.2, CLEAR, 1.8).phase == "wait"
    assert transit.step(robot, "upper", 1.21, CLEAR, 2.1).phase == "wait"
    assert transit.step(robot, "upper", 1.20, CLEAR, 2.9).phase == "exit"


def test_vertical_displacement_is_measured_from_takeover_height():
    plan = make_plan()
    transit = ElevatorTransit(plan, 0.0)
    transit.step(START, "lower", 0.0, BLOCKED, 0.0)
    transit.step(plan.anchor, "lower", 2.0, BLOCKED, 5.0)
    transit.step(plan.boarding, "lower", 2.0, BLOCKED, 5.1)
    robot = Pose2D(plan.boarding.x, plan.boarding.y, plan.exit.yaw)
    transit.step(robot, "lower", 2.0, BLOCKED, 5.2)
    transit.step(robot, "upper", 2.2, CLEAR, 6.0)

    assert transit.step(robot, "upper", 2.2, CLEAR, 7.1).phase == "wait"


def test_resumed_vertical_motion_during_exit_reclaims_wait_ownership():
    transit, _, robot = waiting_transit()
    transit.step(robot, "upper", 3.0, CLEAR, 1.0)
    assert transit.step(robot, "upper", 3.0, CLEAR, 2.1).phase == "exit"

    resumed = transit.step(robot, "upper", 3.2, CLEAR, 2.2)

    assert resumed.phase == "wait" and resumed.hold and resumed.target is None
    assert transit.step(robot, "upper", 3.21, CLEAR, 2.5).phase == "wait"
    assert transit.step(robot, "upper", 3.21, CLEAR, 3.3).phase == "exit"


@pytest.mark.parametrize("depth", [np.zeros((80, 100)), np.full((80, 100), np.nan),
                                  np.empty((0, 0))])
def test_invalid_live_depth_never_releases(depth):
    transit, _, robot = waiting_transit()
    transit.step(robot, "upper", 3.0, CLEAR, 1.0)

    assert transit.step(robot, "upper", 3.0, depth, 3.0).phase == "wait"


def test_exit_holds_on_floor_loss_and_completes_only_at_exit_target():
    transit, plan, robot = waiting_transit()
    transit.step(robot, "upper", 3.0, CLEAR, 1.0)
    transit.step(robot, "upper", 3.0, CLEAR, 2.1)

    lost = transit.step(plan.exit, None, 3.0, CLEAR, 2.2)
    assert lost.hold and not lost.completed and lost.target is None
    assert not transit.completed
    done = transit.step(plan.exit, "upper", 3.0, CLEAR, 2.3)
    assert done.completed and transit.completed
    assert done.phase == "complete"
    assert transit.stats["steps"] == 7
    assert transit.stats["elapsed_s"] == pytest.approx(2.3)
    assert [event["phase"] for event in transit.stats["phase_events"]] == [
        "board", "turn", "wait", "exit", "complete"
    ]


def test_completion_remains_latched_for_later_invalid_observations():
    transit, plan, robot = waiting_transit()
    transit.step(robot, "upper", 3.0, CLEAR, 1.0)
    transit.step(robot, "upper", 3.0, CLEAR, 2.1)
    transit.step(plan.exit, "upper", 3.0, CLEAR, 2.2)

    result = transit.step(plan.exit, None, float("nan"), np.empty((0, 0)), 2.3)

    assert result.completed and transit.completed
    assert transit.stats["elapsed_s"] == pytest.approx(2.2)


def test_checkpoint_preserves_wait_window_and_has_no_plan_or_depth():
    transit, plan, robot = waiting_transit(start_z=5.0)
    transit.step(robot, "upper", 2.0, CLEAR, 1.0)
    state = json.loads(json.dumps(transit.state_dict(), allow_nan=False))
    restored = ElevatorTransit(plan, 5.0)
    restored.load_state_dict(state)

    result = restored.step(robot, "upper", 2.01, CLEAR, 2.1)

    assert result.phase == "exit"
    assert restored.stats == transit.stats or restored.stats["steps"] == transit.stats["steps"] + 1
    assert "plan" not in state and "depth" not in state


def test_checkpoint_cannot_restore_against_a_different_derived_plan():
    transit, plan, _ = waiting_transit()
    other_plan = replace(plan, anchor=Pose2D(plan.anchor.x + 1.0, plan.anchor.y, plan.anchor.yaw))
    restored = ElevatorTransit(other_plan, 0.0)

    with pytest.raises(ValueError, match="plan"):
        restored.load_state_dict(transit.state_dict())
