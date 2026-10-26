"""T1/T4 base_topo tests: localization, navigation core and subgoal-image sequences."""

import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pytest
from vnav_t_fixtures import CAMERA_FORWARD_M, FakeWorld, TruthLocalizer, make_map, unicycle_to

from frontier_nav_bench import mapview as mv
from frontier_nav_bench.interfaces import EpisodeSpec, Pose2D
from frontier_nav_bench.methods import base_topo as bt

HOLD_S = 0.2
ALIGN_WZ = 0.6  # Fake-controller alignment yaw rate, at least 0.5 rad/s (plan section 6.6)


def _handle(tmp_path: Path, trav_edges: Sequence[Tuple[int, int]] = ()) -> mv.SanitizedMap:
    src = make_map(tmp_path / "src", trav_edges=trav_edges)
    mv.build_sanitized_view(src, tmp_path / "view")
    return mv.SanitizedMap(tmp_path / "view", "m", "dense", "trav", CAMERA_FORWARD_M)


def _spec() -> EpisodeSpec:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    return EpisodeSpec(
        "tok", 0, "env0", img, b"goal", np.zeros((48, 64), np.float32), (1.0,), 120.0
    )


def test_pose_algebra_roundtrip() -> None:
    a, b = Pose2D(1.0, -2.0, 0.7), Pose2D(0.3, 0.4, -2.0)
    ab = bt.compose(a, b)
    back = bt.compose(bt.inverse(a), ab)
    assert back == pytest.approx(b)
    assert bt.compose(a, bt.inverse(a)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert bt.wrap_angle(3 * math.pi) == pytest.approx(math.pi)


def test_contloc_tracker_maps_odom_to_map() -> None:
    world = FakeWorld((2.0, 1.0, math.pi / 2))
    tracker = bt.ContLocTracker(world.pose, world.observation().odom)
    world.step(0.5, 0.0, 0.3, 2.0)
    assert tracker.to_map(world.observation().odom) == pytest.approx(tuple(world.pose))
    odom_pose = tracker.map_to_odom(world.pose)
    assert odom_pose == pytest.approx(tuple(world.odom_pose()))


def test_localization_failure_raises() -> None:
    class Refuse:
        def localize(self, *args: object, purpose: str) -> bt.LocResult:
            return bt.LocResult(False, None, None, "quota")

    with pytest.raises(bt.LocalizationFailed, match="quota"):
        bt.InitLocProcedure(Refuse()).localize_start(FakeWorld((0, 0, 0)).observation())


def test_plan_uses_shortcut_edges(tmp_path: Path) -> None:
    chain = bt.TopoNavCore(_handle(tmp_path / "a"))
    route = chain.start(Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 4.2, math.pi / 2))
    assert route.kf_path == tuple(range(9)) and route.length_m == pytest.approx(8.0)
    short = bt.TopoNavCore(_handle(tmp_path / "b", trav_edges=[(0, 8)]))
    route = short.start(Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 4.2, math.pi / 2))
    assert route.kf_path == (0, 8) and route.length_m == pytest.approx(math.hypot(4, 4))


def test_stall_blocks_edge_and_replans(tmp_path: Path) -> None:
    core = bt.TopoNavCore(_handle(tmp_path, trav_edges=[(1, 8)]))
    robot = Pose2D(1.0, 0.0, 0.0)  # Place the body at keyframe 1
    route = core.start(robot, Pose2D(4.0, 4.0, math.pi / 2))
    assert route.kf_path == (1, 8)  # Initially take shortcut 1 -> 8
    target = core.update(robot, 0.0)
    assert target.kf_id == 8
    for step in range(1, 60):
        target = core.update(robot, step * HOLD_S)
        if core.stats["replans"]:
            break
    assert core.stats == {
        "replans": 1, "replans_on_switch": 0, "subgoal_advances": 0, "blocked_edges": 1,
        "degraded": 0, "degraded_no_route": 0, "stall_kept_no_alt": 0, "stall_kept_after_cap": 0,
    }
    assert core._planner.blocked == frozenset({(1, 8)})
    assert target.phase == bt.PHASE_SUBGOAL and target.kf_id == 2  # Switch to the chain route
    assert step * HOLD_S == pytest.approx(bt.STALL_WINDOW_S)


def test_stall_on_chain_keeps_subgoal(tmp_path: Path) -> None:
    """A stalled chain has no alternate route: retain the edge and continue pursuing the subgoal."""
    core = bt.TopoNavCore(_handle(tmp_path))
    robot = Pose2D(1.0, 0.0, 0.0)  # Place the body at keyframe 1 with subgoal 2
    core.start(robot, Pose2D(4.0, 4.0, math.pi / 2))
    target = core.update(robot, 0.0)
    steps = int(round((3 * bt.STALL_WINDOW_S + 1.0) / HOLD_S))  # Restart timing on the frame after
    # each detected stall
    for step in range(1, steps + 1):
        target = core.update(robot, step * HOLD_S)
    assert core.stats["stall_kept_no_alt"] == 3
    assert core.stats["degraded"] == 0 and core.stats["replans"] == 0
    assert core._planner.blocked == frozenset()
    assert target.phase == bt.PHASE_SUBGOAL and target.kf_id == 2


def test_stall_after_cap_keeps_subgoal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After exhausting blocked-edge replans, keep pursuing the subgoal without degrading."""
    monkeypatch.setattr(bt, "MAX_REPLANS", 1)
    core = bt.TopoNavCore(_handle(tmp_path, trav_edges=[(1, 8)]))
    robot = Pose2D(1.0, 0.0, 0.0)
    core.start(robot, Pose2D(4.0, 4.0, math.pi / 2))
    target = core.update(robot, 0.0)
    steps = int(round(2 * bt.STALL_WINDOW_S / HOLD_S))
    for step in range(1, steps + 1):
        target = core.update(robot, step * HOLD_S)
    assert core.stats["replans"] == 1 and core.stats["blocked_edges"] == 1  # The first replan still
    # blocks 1 -> 8
    assert core.stats["stall_kept_after_cap"] == 1 and core.stats["degraded"] == 0
    assert target.phase == bt.PHASE_SUBGOAL and target.kf_id == 2


def test_planner_failure_still_degrades(tmp_path: Path) -> None:
    """Count direct-goal fallback only when the planner cannot find a route during subgoal
    replanning.
    """
    core = bt.TopoNavCore(_handle(tmp_path))
    robot = Pose2D(0.0, 0.0, 0.0)
    core.start(robot, Pose2D(4.0, 4.0, math.pi / 2))
    core._planner.block_edge(1, 2)  # Deliberately disconnect the chain
    assert not core._replan(Pose2D(1.0, 0.0, 0.0), None)
    assert core.stats["degraded_no_route"] == 1 and core.stats["degraded"] == 1
    assert core.update(Pose2D(1.0, 0.0, 0.0), 0.0).phase == bt.PHASE_FINAL


def _drive(
    handle: mv.SanitizedMap, start: Tuple[float, float, float], goal: Tuple[float, float, float]
) -> Tuple[FakeWorld, TruthLocalizer, bt.TopoNavCore, bool]:
    """Run an episode with fake simulation/localization and a fake point controller, matching B0
    wiring.
    """
    world = FakeWorld(start)
    loc = TruthLocalizer(world, goal)
    obs = world.observation()
    start_map = bt.InitLocProcedure(loc).localize_start(obs)
    tracker = bt.ContLocTracker(start_map, obs.odom)
    goal_map = bt.localize_goal(loc, _spec())
    core = bt.TopoNavCore(handle)
    core.start(start_map, goal_map)
    while world.sim_time_s < world.budget_s:
        obs = world.observation()
        target = core.update(tracker.to_map(obs.odom), obs.sim_time_s)
        if target.phase == bt.PHASE_DONE:
            return world, loc, core, True
        if target.phase == bt.PHASE_ALIGN:
            cmd = (0.0, 0.0, math.copysign(ALIGN_WZ, target.yaw_err_rad))
        else:
            goal_odom = tracker.map_to_odom(target.pose_map)
            cmd = unicycle_to(tuple(obs.odom.pose), (goal_odom.x, goal_odom.y))
        world.step(*cmd, HOLD_S)
    return world, loc, core, False


@pytest.mark.parametrize("shortcut", [False, True])
def test_end_to_end_reaches_and_aligns(tmp_path: Path, shortcut: bool) -> None:
    handle = _handle(tmp_path, trav_edges=[(0, 8)] if shortcut else [])
    goal = (4.1, 4.3, math.pi / 2 + 0.3)
    world, loc, core, done = _drive(handle, (0.2, 0.1, math.pi), goal)
    assert done
    assert math.hypot(world.pose.x - goal[0], world.pose.y - goal[1]) <= bt.FINAL_POS_TOL_M
    assert abs(bt.wrap_angle(world.pose.yaw - goal[2])) <= bt.FINAL_YAW_TOL_RAD
    assert loc.calls == ["self", "goal"]  # Continuous localization propagates odometry without
    # further localizer calls
    if shortcut:
        assert world.path_m < 7.0 and core.stats["subgoal_advances"] == 0
    else:
        assert world.path_m > 7.0 and core.stats["subgoal_advances"] >= 6  # The 1 m progression
        # radius cuts corners


def test_route_length_from_includes_both_ends(tmp_path: Path) -> None:
    core = bt.TopoNavCore(_handle(tmp_path))
    robot, goal = Pose2D(0.0, -1.0, 0.0), Pose2D(5.0, 4.0, 0.0)
    route = core.start(robot, goal)
    total = bt.route_length_from(robot, goal, route, core._graph)
    assert total == pytest.approx(1.0 + 8.0 + 1.0)


def test_subgoal_feed_sequence(tmp_path: Path) -> None:
    handle = _handle(tmp_path)
    feed = bt.TopoSubgoalFeed.plan(handle, Pose2D(-0.3, 0.0, 0.0), Pose2D(4.0, 4.0, 1.57))
    assert feed.kf_ids == tuple(range(1, 9))  # Skip starting keyframe 0, which is 0.3 m from the
    # body
    assert feed.route.kf_path[0] == 0
    first = feed.current()
    assert first is not None and (first.index, first.kf_id, first.is_last) == (0, 1, False)
    assert first.rgb_jpeg == handle.keyframe_rgb_jpeg(1)
    assert feed.peek(2).kf_id == 3 and feed.index == 0
    state = feed.state_dict()
    for _ in range(7):
        feed.advance()
    last = feed.current()
    assert last is not None and last.kf_id == 8 and last.is_last
    assert feed.advance() is None and feed.finished and feed.advance() is None
    feed.load_state_dict(state)
    assert feed.current().kf_id == 1
    with pytest.raises(ValueError):
        feed.load_state_dict({"kf_ids": [9], "index": 0})


def test_subgoal_feed_single_keyframe(tmp_path: Path) -> None:
    handle = _handle(tmp_path)
    feed = bt.TopoSubgoalFeed.plan(handle, Pose2D(4.0, 4.0, 0.0), Pose2D(4.0, 4.1, 0.0))
    assert feed.kf_ids == (8,)
    assert feed.current().is_last


def _walk(core: bt.TopoNavCore, points, t0: float = 0.0):  # type: ignore[no-untyped-def]
    """Move the body through the poses and return each subgoal keyframe ID, or None at completion.
    """
    return [core.update(Pose2D(x, y, 0.0), t0 + i * 0.2).kf_id for i, (x, y) in enumerate(points)]


def test_rule_normal_walk_completes(tmp_path: Path) -> None:
    core = bt.TopoNavCore(_handle(tmp_path))
    core.start(Pose2D(-0.3, 0.0, 0.0), Pose2D(4.0, 4.3, 1.57))
    path = [(x * 0.25, 0.0) for x in range(0, 17)] + [(4.0, y * 0.25) for y in range(1, 17)]
    kfs = [k for k in _walk(core, path) if k is not None]
    assert kfs == sorted(kfs) and kfs[-1] == 8  # Subgoals advance only forward
    assert core.remaining == () or core.remaining == (8,)
    assert core.stats["replans"] == 0 and core.stats["replans_on_switch"] >= 6


def test_rule_corner_cut_jumps_ahead(tmp_path: Path) -> None:
    """Shortcut from (2, 0) to (4, 1.9): nearest pending keyframe 6 is within 1 m, so choose 7."""
    core = bt.TopoNavCore(_handle(tmp_path))
    core.start(Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 4.0, 1.57))
    assert core.update(Pose2D(2.0, 0.0, 0.0), 0.0).kf_id == 3
    assert core.update(Pose2D(4.0, 1.9, 0.0), 0.2).kf_id == 7
    assert set(core.visited) >= {0, 1, 2, 3, 4, 5, 6}
    assert core.stats["replans_on_switch"] == 1


def test_rule_off_route_then_back(tmp_path: Path) -> None:
    """At (1, -2.5), pending keyframe 3 remains nearest; resume progression after returning to the
    route.
    """
    core = bt.TopoNavCore(_handle(tmp_path))
    core.start(Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 4.0, 1.57))
    assert core.update(Pose2D(2.0, 0.0, 0.0), 0.0).kf_id == 3
    assert core.update(Pose2D(1.0, -2.5, 0.0), 0.2).kf_id == 3
    # Returning within 1 m of 3 selects 4; replanning drops 4 at 0.9 m and advances to 5.
    assert core.update(Pose2D(3.1, 0.0, 0.0), 0.4).kf_id == 5


def test_rule_never_turns_back(tmp_path: Path) -> None:
    """Remove visited points: returning near the start still targets pending frames, never 0/1."""
    core = bt.TopoNavCore(_handle(tmp_path))
    core.start(Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 4.0, 1.57))
    assert core.update(Pose2D(2.6, 0.0, 0.0), 0.0).kf_id == 4
    back = core.update(Pose2D(0.0, 0.0, 0.0), 0.2)
    assert back.kf_id == 4 and 0 not in core.remaining and 1 not in core.remaining


def test_pose_subgoal_feed_follows_pose(tmp_path: Path) -> None:
    handle = _handle(tmp_path)
    feed = bt.PoseSubgoalFeed(handle, Pose2D(-0.3, 0.0, 0.0), Pose2D(4.0, 4.3, 1.57))
    seen = []
    path = [(x * 0.5, 0.0) for x in range(0, 9)] + [(4.0, y * 0.5) for y in range(1, 9)]
    for t, (x, y) in enumerate(path):
        sg = feed.current_subgoal_image(Pose2D(x, y, 0.0), t * 0.2)
        if sg is not None:
            assert sg.rgb_jpeg == handle.keyframe_rgb_jpeg(sg.kf_id)
            if not seen or seen[-1] != sg.kf_id:
                seen.append(sg.kf_id)
    assert seen == list(range(seen[0], 9)) and seen[0] <= 2  # Advance forward through every frame
    # to keyframe 8
    assert feed.current_subgoal_image(Pose2D(4.0, 4.1, 0.0), 10.0) is None
    assert feed.phase == bt.PHASE_ALIGN  # Within 0.5 m of the goal, only the final goal image
    # remains
    assert feed.current_subgoal_image(Pose2D(4.0, 4.3, 1.57), 10.2) is None
    assert feed.phase == bt.PHASE_DONE
    last = bt.PoseSubgoalFeed(handle, Pose2D(3.6, 3.9, 0.0), Pose2D(4.0, 4.6, 1.57))
    sg = last.current_subgoal_image(Pose2D(3.6, 3.9, 0.0), 0.0)
    assert sg is None or sg.is_last


def test_pose_subgoal_feed_switch_bumps_version(tmp_path: Path) -> None:
    feed = bt.PoseSubgoalFeed(_handle(tmp_path), Pose2D(0.0, 0.0, 0.0), Pose2D(4.0, 4.0, 1.57))
    first = feed.current_subgoal_image(Pose2D(0.0, 0.0, 0.0), 0.0)
    assert first is not None and first.route_version == 0
    sg = feed.current_subgoal_image(Pose2D(4.0, 2.9, 0.0), 0.2)  # Take a shortcut near keyframe 7
    assert sg is not None and sg.kf_id == 8 and sg.is_last and sg.route_version == 1
    assert feed.stats["replans_on_switch"] == 1


def test_pose_goal_reached_shared_rule(tmp_path: Path) -> None:
    goal = Pose2D(4.0, 4.3, 1.57)
    assert bt.pose_goal_reached(Pose2D(4.2, 4.3, 1.57 + 0.5), goal)  # 0.2 m、0.5 rad < 30°
    assert bt.pose_goal_reached(Pose2D(4.45, 4.3, 1.57), goal)  # 0.45 m ≤ 0.5 m
    assert not bt.pose_goal_reached(Pose2D(4.55, 4.3, 1.57), goal)  # 0.55 m
    assert not bt.pose_goal_reached(Pose2D(4.0, 4.3, 1.57 + 0.6), goal)  # 0.6 rad > 30°
    feed = bt.PoseSubgoalFeed(_handle(tmp_path), Pose2D(0.0, 0.0, 0.0), goal)
    assert feed.pose_goal_reached(Pose2D(4.1, 4.2, 1.6))
    assert not feed.pose_goal_reached(Pose2D(0.0, 0.0, 0.0))
