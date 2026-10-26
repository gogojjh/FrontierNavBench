"""Test SimEnv truth isolation, odometry origin, budget truncation and localization quotas."""

import math

import numpy as np
import pytest

from frontier_nav_bench import fairness as fz
from frontier_nav_bench.interfaces import Action, Pose2D
from frontier_nav_bench.localizer import OracleLocalizer, map_from_world, transform_pose2d
from frontier_nav_bench.sim.backend import KinematicBackend
from frontier_nav_bench.sim.ctrl_client import InProcessClient
from frontier_nav_bench.sim.ctrl_server import CtrlServer
from frontier_nav_bench.sim_env import SimEnv, observation_field_names

IDENTITY16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _env(budget: float = 180.0) -> SimEnv:
    client = InProcessClient(CtrlServer(KinematicBackend(), "inproc://x"))
    client.hello()
    return SimEnv(client, budget)


def test_no_world_fields() -> None:
    names = observation_field_names()
    assert not [n for n in names if "world" in n or "truth" in n or "gt" in n.split("_")]


def test_odom_origin_is_start_pose() -> None:
    env = _env()
    obs0 = env.start(Pose2D(3.0, 4.0, 1.0))
    assert obs0.odom.pose == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert obs0.odom.quat_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)
    assert obs0.rgb.shape == (480, 640, 3) and obs0.rgb.dtype == np.uint8
    assert obs0.depth_m.dtype == np.float32 and float(np.median(obs0.depth_m)) == pytest.approx(3.0)
    obs, _ = env.step(Action(0.5, 0.0, 0.0, 1.0))
    assert obs.odom.pose.x == pytest.approx(0.5, abs=1e-3) and abs(obs.odom.pose.y) < 1e-3
    truth = env.truth_pose()
    assert truth.x == pytest.approx(3.0 + 0.5 * math.cos(1.0), abs=1e-3)
    assert truth.y == pytest.approx(4.0 + 0.5 * math.sin(1.0), abs=1e-3)
    assert len(obs.frames) == 10 and obs.frames[-1][0] == pytest.approx(1.0)
    assert obs.sim_time_s == pytest.approx(1.0) and obs.step_index == 1
    assert len(env.truth.samples) == 11  # Time zero plus ten 10 Hz samples.


def test_budget_clips_last_hold() -> None:
    env = _env(budget=1.25)
    env.start(Pose2D(0.0, 0.0, 0.0))
    env.step(Action(0.3, 0.0, 0.0, 1.0))
    obs, _ = env.step(Action(0.3, 0.0, 0.0, 1.0))  # Only 0.25 s remains, so truncate to 0.2 s.
    assert obs.sim_time_s == pytest.approx(1.2) and obs.budget_left_s == pytest.approx(0.05)
    assert env.out_of_budget
    with pytest.raises(RuntimeError):
        env.step(Action(0.3, 0.0, 0.0, 0.1))
    assert [h for _, _, h in env.applied] == [1.0, 0.2]


def test_transform_map_world() -> None:
    # Map frame: translate (1,2), rotate 90 deg; world (1,3), yaw +y becomes map (1,0), yaw zero.
    c, s = 0.0, 1.0
    m2w = [c, -s, 0, 1, s, c, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1]
    p = transform_pose2d(map_from_world(m2w), Pose2D(1.0, 3.0, math.pi / 2))
    assert p == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_oracle_localizer_goal_and_self() -> None:
    env = _env()
    env.start(Pose2D(2.0, 1.0, 0.0))
    goal = Pose2D(5.0, 6.0, 0.3)
    loc = OracleLocalizer(env.truth_pose, goal, IDENTITY16)
    g = loc.localize(b"", None, (), purpose="goal")
    assert g.ok and g.pose_map == pytest.approx(tuple(goal))
    env.step(Action(0.4, 0.0, 0.0, 0.5))
    me = loc.localize(b"", None, (), purpose="self")
    assert me.pose_map.x == pytest.approx(2.2, abs=1e-3)


def test_quota_contloc_initloc_noloc() -> None:
    env = _env()
    obs = env.start(Pose2D(0.0, 0.0, 0.0))
    inner = OracleLocalizer(env.truth_pose, Pose2D(1, 1, 0), IDENTITY16)
    cont = fz.QuotaLocalizer(inner, fz.parse_method_id("dense-contloc-ptgoal"))
    init = fz.QuotaLocalizer(inner, fz.parse_method_id("dense-initloc-imggoal-astra"))
    for k in range(3):
        cont.on_frame(obs.step_index)
        init.on_frame(obs.step_index)
        assert cont.localize(b"", None, (), purpose="self").ok
        assert init.localize(b"", None, (), purpose="self").ok == (k == 0)
        obs, _ = env.step(Action(0.2, 0.0, 0.0, 0.2))
    assert fz.parse_method_id("nomap-noloc-imggoal-qwen").self_loc_quota == 0
