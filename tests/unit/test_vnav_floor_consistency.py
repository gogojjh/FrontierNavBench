"""Floor-aware arrival checks use real scene definitions, without a simulator."""

from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from vnav_t_fixtures import CAMERA_FORWARD_M, FakeWorld, make_map

from frontier_nav_bench import mapview, metrics, oracle
from frontier_nav_bench.interfaces import Pose2D
from frontier_nav_bench.localizer import OracleLocalizer
from frontier_nav_bench.methods import base_topo as bt

GOAL = Pose2D(0.0, 0.0, 0.0)
LOW_Z = 0.4  # Body height on 1F.
HIGH_Z = 3.4  # Body height on 2F (3 m between floors).


def test_same_xy_on_wrong_floor_cannot_succeed() -> None:
    goal_floor = oracle.floor_of_z(oracle.SCENE_ENV3, HIGH_Z)
    trace = [metrics.TraceRow(0.0, 0.0, 0.0, 0.0, z=LOW_Z)]
    out = metrics.evaluate_run(
        trace, GOAL, 8.0, True, "done", 1,
        goal_scene=oracle.SCENE_ENV3, goal_floor=goal_floor,
    )
    assert not out.success and not out.success_loose and not out.ever_within_pose
    assert out.premature_done and out.spl == 0.0 and out.plr is None
    assert out.final_dist_m is None and out.final_yaw_err_deg is None
    trace.append(metrics.TraceRow(1.0, 0.0, 0.0, 0.0, z=HIGH_Z))
    reached = metrics.evaluate_run(
        trace, GOAL, 8.0, True, "done", 2,
        goal_scene=oracle.SCENE_ENV3, goal_floor=goal_floor,
    )
    assert reached.success and reached.success_loose and reached.ever_within_pose
    assert reached.final_dist_m == 0.0 and reached.final_yaw_err_deg == 0.0


@pytest.mark.parametrize("scene", [None, oracle.SCENE_ENV0, oracle.SCENE_ENV1])
def test_single_floor_outcome_is_exactly_unchanged(scene: Optional[str]) -> None:
    trace = [metrics.TraceRow(0.0, 1.0, 0.0, 0.0, z=LOW_Z),
             metrics.TraceRow(1.0, 0.0, 0.0, 0.0, z=HIGH_Z)]
    legacy = metrics.evaluate_run(trace, GOAL, 1.0, True, "done", 2)
    actual = metrics.evaluate_run(trace, GOAL, 1.0, True, "done", 2, goal_scene=scene)
    assert actual.to_dict() == legacy.to_dict()
    assert actual.success


def test_pose_arrival_requires_matching_known_floors() -> None:
    assert not bt.pose_goal_reached(GOAL, GOAL, robot_floor="1F", goal_floor="2F")
    assert bt.pose_goal_reached(GOAL, GOAL, robot_floor="2F", goal_floor="2F")
    assert bt.pose_goal_reached(GOAL, GOAL)


def test_multifloor_evaluation_requires_a_known_goal_floor() -> None:
    with pytest.raises(ValueError, match="goal_floor"):
        metrics.evaluate_run(
            [metrics.TraceRow(0, 0, 0, 0, z=LOW_Z)], GOAL, 1, True, "done", 1,
            goal_scene=oracle.SCENE_ENV3,
        )


def test_oracle_localizer_returns_current_and_goal_floor() -> None:
    height = [LOW_Z]
    loc = OracleLocalizer(
        lambda: GOAL, GOAL, np.eye(4).ravel(), scene=oracle.SCENE_ENV3,
        truth_z_now=lambda: height[0], goal_floor=oracle.floor_of_z(oracle.SCENE_ENV3, HIGH_Z),
    )
    low = loc.localize(b"", None, (), purpose="self")
    goal = loc.localize(b"", None, (), purpose="goal")
    assert low.floor != goal.floor
    height[0] = HIGH_Z
    assert loc.localize(b"", None, (), purpose="self").floor == goal.floor


def test_contloc_height_and_subgoal_checks(tmp_path: Path) -> None:
    src = make_map(tmp_path / "src", poses=[(0.0, 0.0, 0.0), (4.0, 0.0, 0.0),
                                                 (0.0, 0.0, 0.0)])
    # Lift the last camera pose to 2F before producing the sanitized map.
    lines = (src / "poses.txt").read_text().splitlines()
    parts = lines[-1].split()
    parts[6] = str(float(parts[6]) + 3.0)  # Camera y points down; inverse ty lifts map z.
    lines[-1] = " ".join(parts)
    (src / "poses.txt").write_text("\n".join(lines) + "\n")
    mapview.build_sanitized_view(src, tmp_path / "view")
    handle = mapview.SanitizedMap(
        tmp_path / "view", "stairs", "dense", "trav", CAMERA_FORWARD_M,
        scene=oracle.SCENE_ENV3, map_to_world=np.eye(4).ravel(),
    )
    low = oracle.floor_of_z(oracle.SCENE_ENV3, LOW_Z)
    high = oracle.floor_of_z(oracle.SCENE_ENV3, HIGH_Z)
    assert handle.keyframe_floor(0) == low and handle.keyframe_floor(2) == high
    obs = FakeWorld((0, 0, 0)).observation()
    tracker = bt.ContLocTracker(GOAL, obs.odom, map_handle=handle)
    assert tracker.floor(replace(obs.odom, position_xyz=(0, 0, LOW_Z))) == low
    assert tracker.floor(replace(obs.odom, position_xyz=(0, 0, HIGH_Z))) == high
    core = bt.TopoNavCore(handle)
    core.start(GOAL, GOAL, robot_floor=low, goal_floor=high)
    # The XY nearest keyframe on the other floor must not bypass the stairs.
    target = core.update(GOAL, 0.0, robot_floor=low)
    assert target.phase == bt.PHASE_SUBGOAL and target.kf_id == 1
    target = core.update(Pose2D(4, 0, 0), 1.0, robot_floor=low)
    assert target.phase == bt.PHASE_SUBGOAL and target.kf_id == 2
    assert core.update(GOAL, 2.0, robot_floor=low).phase != bt.PHASE_DONE
    assert core.update(GOAL, 3.0, robot_floor=high).phase == bt.PHASE_DONE


def test_cannot_skip_an_unvisited_floor_transition(tmp_path: Path) -> None:
    src = make_map(tmp_path / "src", poses=[(x, 0, 0) for x in (0, 3, 8, 12)])
    mapview.build_sanitized_view(src, tmp_path / "view")
    handle = mapview.SanitizedMap(tmp_path / "view", "m", "dense", "trav", CAMERA_FORWARD_M)
    floors = {0: "f1", 1: "f2", 2: "f1", 3: "f2"}
    handle.keyframe_floor = floors.__getitem__
    core = bt.TopoNavCore(handle)
    core.start(GOAL, Pose2D(12, 0, 0), robot_floor="f1", goal_floor="f2")
    assert core.update(GOAL, 0, robot_floor="f1").kf_id == 1
    assert core.update(Pose2D(8, 0, 0), 1, robot_floor="f1").kf_id == 1
    assert core.visited == (0,)


@pytest.mark.parametrize(
    "method_id", ["dense-contloc-imggoal-astra", "sparse-contloc-imggoal-astra"]
)
def test_imggoal_contloc_done_uses_localizer_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method_id: str
) -> None:
    from test_vnav_imggoal_vlm import GOAL as IMG_GOAL
    from test_vnav_imggoal_vlm import _Episode, _reply

    episode = _Episode(tmp_path, method_id, [_reply(done=True)] * 3)
    original = episode.inner_loc.localize
    current_floor = ["f1"]

    def localize(*args: object, purpose: str) -> bt.LocResult:
        result = original(*args, purpose=purpose)
        return replace(result, floor="f2" if purpose == "goal" else current_floor[0])

    monkeypatch.setattr(episode.inner_loc, "localize", localize)
    episode.method.reset(episode.spec, episode.world.observation(), episode.ctx)
    episode.world.pose = Pose2D(*IMG_GOAL)
    episode.loc.on_frame(1)
    assert not episode.method.act(episode.obs()).done
    current_floor[0] = "f2"
    episode.loc.on_frame(2)
    assert episode.method.act(episode.obs()).done
    episode.method.close()


@pytest.mark.parametrize("method_id", ["dense-contloc-ptgoal", "sparse-contloc-ptgoal"])
def test_ptgoal_contloc_done_uses_odometry_height(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method_id: str
) -> None:
    from test_vnav_imggoal_vlm import GOAL as IMG_GOAL
    from test_vnav_imggoal_vlm import _Episode
    from test_vnav_ptgoal_sru import UnicycleController

    from frontier_nav_bench.fairness import parse_method_id
    from frontier_nav_bench.methods.ptgoal_sru import PtGoalSruMethod

    episode = _Episode(tmp_path, "dense-contloc-imggoal-astra", [])
    original = episode.inner_loc.localize
    low = oracle.floor_of_z(oracle.SCENE_ENV3, LOW_Z)
    high = oracle.floor_of_z(oracle.SCENE_ENV3, HIGH_Z)

    def localize(*args: object, purpose: str) -> bt.LocResult:
        return replace(original(*args, purpose=purpose), floor=high if purpose == "goal" else low)

    monkeypatch.setattr(episode.inner_loc, "localize", localize)
    monkeypatch.setattr(episode.ctx.map, "floor_at_z",
                        lambda z: oracle.floor_of_z(oracle.SCENE_ENV3, z))
    ctx = replace(episode.ctx, profile=parse_method_id(method_id))
    method = PtGoalSruMethod(method_id, controller_factory=lambda _log: UnicycleController())
    obs = episode.world.observation()
    method.reset(episode.spec, obs, ctx)
    episode.world.pose = Pose2D(*IMG_GOAL)
    obs = episode.world.observation()
    assert not method.act(replace(obs, odom=replace(obs.odom, position_xyz=(0, 0, LOW_Z)))).done
    assert method.act(replace(obs, odom=replace(obs.odom, position_xyz=(0, 0, HIGH_Z)))).done
    method.close()


def test_runner_preserves_real_z_in_memory_and_csv(tmp_path: Path) -> None:
    from frontier_nav_bench.rundir import RunDir, read_trace
    from frontier_nav_bench.runner import _trace_rows
    from frontier_nav_bench.sim.ctrl_protocol import WirePose

    wire = WirePose((1.0, 2.0, HIGH_Z), (0.0, 0.0, 0.0, 1.0))
    rows = _trace_rows([(0.0, wire)])
    assert rows[0]["z"] == HIGH_Z
    rd = RunDir(tmp_path, "m", "e", 0)
    rd.begin_attempt()
    rd.append_trace(rows)
    rd.close_streams()
    assert read_trace(rd.path("trace.csv"))[0]["z"] == HIGH_Z
