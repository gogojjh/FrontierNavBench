"""Facility control pauses ordinary advancement and resumes afterward, preserving controller
lifecycle.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from frontier_nav_bench.interfaces import OdomState, Pose2D
from frontier_nav_bench.methods.base_topo import REACH_M, TopoNavCore
from frontier_nav_bench.methods.nav_template import Decision, NavMethod
from frontier_nav_bench.methods.sru.client import PointGoalCommand
from frontier_nav_bench.topo import TopoGraph


class StubMethod(NavMethod):
    def setup(self, spec, obs, ctx):
        self.normal = 0

    def subgoal(self, obs):
        self.normal += 1
        return SimpleNamespace(system_done=False)

    def decide(self, obs, sub):
        return Decision(.6, 0, 0, .2)


def test_shared_shell_pauses_normal_progress():
    method = StubMethod()
    method.reset(None, None, None)
    method._transit_driver = SimpleNamespace(act=lambda obs: Decision(0, 0, 0, .2))
    assert method.act(None).vx == 0
    assert method.normal == 0
    method._transit_driver = SimpleNamespace(act=lambda obs: None)
    assert method.act(None).vx == .6
    assert method.normal == 1


def test_reset_releases_previous_owned_facility_driver():
    method = StubMethod()
    method.reset(None, None, None)
    closed = []
    method._transit_driver = SimpleNamespace(close=lambda: closed.append(True))
    method.reset(None, None, None)
    assert closed == [True]
    assert method._transit_driver is None


def test_core_consumes_only_completed_prefix_and_keeps_one_meter():
    graph = TopoGraph({i: Pose2D(i * 2, 0, 0) for i in range(4)}, {
        0: ((1, 2),), 1: ((0, 2), (2, 2)), 2: ((1, 2), (3, 2)), 3: ((2, 2),),
    })
    handle = SimpleNamespace(topo_graph=lambda: graph, keyframe_floor=lambda kf: None)
    core = TopoNavCore(handle)
    core.start(Pose2D(0, 0, 0), Pose2D(6, 0, 0))
    core.consume_prefix((0, 1))
    assert core.remaining == (2, 3) or core.remaining == [2, 3]
    assert list(core.visited) == [0, 1]
    assert REACH_M == 1.0 and core._reach_m == 1.0
    core.consume_prefix((0, 1))
    assert list(core.visited) == [0, 1]


def observation(t=0., x=0.):
    odom = OdomState(Pose2D(x, 0, 0), (x, 0, .4), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0))
    return SimpleNamespace(odom=odom, depth_m=np.ones((10, 10)), sim_time_s=t, step_index=0)


class Controller:
    def __init__(self):
        self.reset_count = 0
        self.closed = False

    def reset(self):
        self.reset_count += 1

    def step(self, depth, odom, target):
        return PointGoalCommand(.2, 0, 0, False, .001)

    def close(self):
        self.closed = True


@pytest.mark.parametrize('owned', [False, True])
def test_driver_holds_and_resumes_without_restarting_existing_controller(owned):
    from frontier_nav_bench.methods.base_topo import ContLocTracker
    from frontier_nav_bench.methods.transit_driver import TransitDriver

    controller = Controller()
    finished = []
    steps = iter([
        SimpleNamespace(phase='board', target=Pose2D(2, 0, 0), yaw_error=0,
                        hold=False, completed=False),
        SimpleNamespace(phase='wait', target=None, yaw_error=0, hold=True, completed=False),
        SimpleNamespace(phase='complete', target=None, yaw_error=0, hold=True, completed=True),
    ])
    transit = SimpleNamespace(
        step=lambda *args: next(steps), phase='board', stats={},
        plan=SimpleNamespace(consumed_kfs=(0, 1)),
    )
    obs = observation()
    driver = TransitDriver(
        transit, ContLocTracker(Pose2D(0, 0, 0), obs.odom),
        lambda ids: finished.append(ids), None if owned else controller,
        lambda: controller,
    )
    assert driver.act(obs).vx == .2
    hold = driver.act(observation(.2))
    assert (hold.vx, hold.vy, hold.wz, hold.done) == (0, 0, 0, False)
    assert driver.act(observation(.4)) is None
    assert finished == [(0, 1)]
    driver.close()
    assert controller.closed == owned
    assert driver.act(observation(.6)) is None


def test_wait_heading_correction_rotates_without_translation_or_starting_sru():
    from frontier_nav_bench.methods.base_topo import ContLocTracker
    from frontier_nav_bench.methods.transit_driver import TransitDriver

    def forbidden():
        raise AssertionError('等待转向不需要启动SRU')

    obs = observation()
    transit = SimpleNamespace(step=lambda *args: SimpleNamespace(
        phase='wait', target=None, yaw_error=.4, hold=False, completed=False,
    ))
    driver = TransitDriver(transit, ContLocTracker(Pose2D(0, 0, 0), obs.odom),
                           lambda ids: None, None, forbidden)
    dec = driver.act(obs)
    assert (dec.vx, dec.vy, dec.done) == (0., 0., False)
    assert dec.wz > 0
    assert dec.debug['phase'] == 'elevator_wait'
    driver.close()


@pytest.mark.parametrize('method_id', [
    f'{variant}-{loc}-imggoal-{vlm}'
    for variant in ('dense', 'sparse')
    for loc in ('contloc', 'initloc')
    for vlm in ('astra', 'qwen')
])
def test_mapped_image_methods_share_takeover_without_model_calls(tmp_path, monkeypatch, method_id):
    from test_vnav_imggoal_vlm import _Episode

    configured = []

    class Driver:
        def act(self, obs):
            return Decision(0, 0, 0, .2)

        def close(self):
            pass

        def summary(self):
            return {}

    def configure(method, ctx, route, start, obs, controller=None):
        configured.append(tuple(route))
        assert ctx.map is not None
        method._transit_driver = Driver()

    monkeypatch.setattr(NavMethod, 'configure_transit', configure)
    ep = _Episode(tmp_path, method_id, [])
    ep.method.reset(ep.spec, ep.world.observation(), ep.ctx)
    assert configured and ep.fake.requests == []
    assert ep.method.act(ep.obs()).vx == 0
    assert ep.fake.requests == []
    if ep.method._pose_feed is not None:
        route = ep.method._pose_feed.route.kf_path
        ep.method.transit_completed(route[:2])
        assert not set(route[:2]) & set(ep.method._pose_feed.core.remaining)
    else:
        ids = ep.method._feed.kf_ids[:2]
        ep.method.transit_completed(ids)
        current = ep.method._feed.current()
        assert current is None or current.kf_id not in ids
    ep.method.close()


def test_nomap_never_configures_facility(tmp_path, monkeypatch):
    from test_vnav_imggoal_vlm import _Episode

    def forbidden(*args, **kwargs):
        raise AssertionError('无图方法不能开启地图设施执行器')

    monkeypatch.setattr(NavMethod, 'configure_transit', forbidden)
    ep = _Episode(tmp_path, 'nomap-noloc-imggoal-astra', [])
    ep.method.reset(ep.spec, ep.world.observation(), ep.ctx)
    assert ep.method._transit_driver is None
    ep.method.close()


@pytest.mark.parametrize('loc_mode', ['contloc', 'initloc'])
def test_completed_transfer_checkpoint_keeps_route_progress_and_feedback(tmp_path, loc_mode):
    from test_vnav_imggoal_vlm import _Episode

    from frontier_nav_bench.methods.base_topo import ContLocTracker
    from frontier_nav_bench.methods.transit_driver import TransitDriver

    def attach(ep, consumed):
        transit = SimpleNamespace(
            plan=SimpleNamespace(consumed_kfs=consumed),
            state_dict=lambda: {}, load_state_dict=lambda state: None,
        )
        driver = TransitDriver(
            transit, ContLocTracker(Pose2D(0, 0, 0), ep.world.observation().odom),
            ep.method.transit_completed, None, lambda: Controller(),
        )
        ep.method._transit_driver = driver
        return driver

    method_id = f'dense-{loc_mode}-imggoal-astra'
    original = _Episode(tmp_path / 'original', method_id, [])
    original.method.reset(original.spec, original.world.observation(), original.ctx)
    feed = original.method._pose_feed or original.method._feed
    consumed = feed.route.kf_path[:3]
    attach(original, consumed)._resumed = True
    original.method.transit_completed(consumed)
    state = original.method.state_dict()

    restored = _Episode(tmp_path / 'restored', method_id, [])
    restored.method.reset(restored.spec, restored.world.observation(), restored.ctx)
    attach(restored, consumed)
    restored.method.load_state_dict(state)
    if loc_mode == 'contloc':
        assert not set(consumed) & set(restored.method._pose_feed.core.remaining)
    else:
        assert restored.method._feed.state_dict() == original.method._feed.state_dict()
    assert restored.method._feedback == original.method._feedback
    original.method.close_backend()
    restored.method.close_backend()
