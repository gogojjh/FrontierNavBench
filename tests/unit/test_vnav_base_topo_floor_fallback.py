"""Scheme D advances close cross-floor targets without waiting for stall recovery."""

import math
from pathlib import Path
from typing import Optional, Tuple

import pytest
from vnav_t_fixtures import CAMERA_FORWARD_M, make_map

from frontier_nav_bench import mapview
from frontier_nav_bench.interfaces import Pose2D
from frontier_nav_bench.methods import base_topo as bt

START = Pose2D(0.0, 0.0, 0.0)
GOAL = Pose2D(9.0, 0.0, 0.0)
STEP_S = 0.2  # Selection must advance before the 10 s stall window expires.


def _handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, escape: bool = False
) -> mapview.SanitizedMap:
    """Build a two-floor route with close transition nodes and an optional detour."""
    poses = [(x, 0.0, 0.0) for x in (0.0, 3.0, 3.25, 6.0, 9.0)]
    poses.append((0.0, 4.0, 0.0))  # Longer escape route: 0 -> 5 -> 4.
    src = make_map(tmp_path / "src", poses=poses, trav_edges=[(0, 5)] if escape else [])
    mapview.build_sanitized_view(src, tmp_path / "view")
    handle = mapview.SanitizedMap(tmp_path / "view", "m", "dense", "trav", CAMERA_FORWARD_M)
    floors = {0: "f1", 1: "f2", 2: "f2", 3: "f2", 4: "f2", 5: "f1"}
    monkeypatch.setattr(handle, "keyframe_floor", floors.__getitem__)
    return handle


def test_multifloor_close_targets_advance_across_consecutive_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A robot still reporting f1 must move past nearby f2 targets in order."""
    core = bt.TopoNavCore(_handle(tmp_path, monkeypatch))
    core.start(START, GOAL, robot_floor="f1", goal_floor="f2")
    first = core.update(START, 0.0, robot_floor="f1")
    robot = core._graph.poses[1]
    second = core.update(robot, STEP_S, robot_floor="f1")
    third = core.update(robot, 2 * STEP_S, robot_floor="f1")
    fourth = core.update(robot, 3 * STEP_S, robot_floor="f1")

    assert [target.kf_id for target in (first, second, third, fourth)] == [1, 2, 3, 3]
    assert all(target.phase == bt.PHASE_SUBGOAL for target in (first, second, third, fourth))
    assert core.visited == (0, 1, 2)
    assert core.remaining == (3, 4)
    assert core.stats["subgoal_advances"] == 2
    assert core.stats["replans"] == 0


@pytest.mark.parametrize(
    "distance_m, advances",
    [(0.0, True), (bt.REACH_M - 0.001, True), (bt.REACH_M, False),
     (bt.REACH_M + 0.001, False)],
    ids=["at-target", "inside-reach", "exact-reach", "outside-reach"],
)
def test_empty_candidates_use_xy_distance_and_advance_only_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, distance_m: float, advances: bool
) -> None:
    """Ignore the floor for distance only; preserve the strict reach threshold."""
    core = bt.TopoNavCore(_handle(tmp_path, monkeypatch))
    core._remaining = [1, 2, 3, 4]
    core._visited = [0]
    pose = core._graph.poses[1]
    robot = Pose2D(pose.x - distance_m, pose.y, pose.yaw)
    assert math.isinf(core._dist_kf(robot, 1, floor="f1"))
    assert core._dist_kf(robot, 1, floor=None) == pytest.approx(distance_m)

    assert core._select(robot, "f1") == (2 if advances else 1)
    assert core.visited == ((0, 1) if advances else (0,))
    assert core.remaining == ((2, 3, 4) if advances else (1, 2, 3, 4))


def test_empty_candidates_can_exhaust_last_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = bt.TopoNavCore(_handle(tmp_path, monkeypatch))
    core._remaining = [4]

    assert core._select(core._graph.poses[4], "f1") is None
    assert core.remaining == ()
    assert core.visited == (4,)


@pytest.mark.parametrize(
    "remaining, robot_x, expected, visited",
    [((1, 2, 3), 0.0, 1, ()),
     ((1, 2, 3), 3.0, 2, (1,)),
     ((0, 1, 2, 3), 6.0, None, (0, 1, 2, 3)),
     ((2, 3, 4), 6.0, 4, (2, 3)),
     ((4,), 9.0, None, (4,)),
     ((), 0.0, None, ())],
    ids=["far", "near", "nearest-later", "suffix", "last", "empty"],
)
def test_unknown_floor_keeps_full_candidate_pool_and_legacy_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remaining: Tuple[int, ...],
    robot_x: float, expected: Optional[int], visited: Tuple[int, ...],
) -> None:
    """With floor=None, every remaining node is evaluated by the old nearest rule."""
    handle = _handle(tmp_path, monkeypatch)

    def forbidden_floor_lookup(kf: int) -> Optional[str]:
        raise AssertionError("floor=None must never evaluate the floor-break condition")

    monkeypatch.setattr(handle, "keyframe_floor", forbidden_floor_lookup)
    core = bt.TopoNavCore(handle)
    core._remaining = list(remaining)
    checked = []
    original_dist = core._dist_kf

    def record_distance(robot: Pose2D, kf: int, floor: Optional[str] = None) -> float:
        checked.append(kf)
        assert floor is None
        return original_dist(robot, kf, floor)

    monkeypatch.setattr(core, "_dist_kf", record_distance)

    assert core._select(Pose2D(robot_x, 0.0, 0.0), None) == expected
    assert checked == list(remaining)
    assert core.visited == visited
    assert core.remaining == remaining[len(visited):]


def test_r15_style_escape_advances_without_stall_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An available detour need not wait for the stall timer and edge blocking."""
    core = bt.TopoNavCore(_handle(tmp_path, monkeypatch, escape=True))
    route = core.start(START, GOAL, robot_floor="f1", goal_floor="f2")
    assert route.kf_path == (0, 1, 2, 3, 4)
    assert core.update(START, 0.0, robot_floor="f1").kf_id == 1
    robot = core._graph.poses[1]
    assert core._has_route_without(robot, (0, 1), "f1")

    assert core.update(robot, STEP_S, robot_floor="f1").kf_id == 2
    assert STEP_S < bt.STALL_WINDOW_S
    assert core.stats["subgoal_advances"] == 1
    assert core.stats["replans_on_switch"] == 1  # Normal target-switch planning is retained.
    for counter in ("replans", "blocked_edges", "stall_kept_no_alt", "stall_kept_after_cap",
                    "degraded", "degraded_no_route"):
        assert core.stats[counter] == 0
    assert core._planner.blocked == frozenset()
