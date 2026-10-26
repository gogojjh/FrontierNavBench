"""Evaluation-metric boundary cases."""

import math
from typing import List, Optional

import numpy as np
import pytest

from vnav_benchmark import metrics as mx
from vnav_benchmark import oracle
from vnav_benchmark.interfaces import Pose2D

GOAL = Pose2D(5.0, 5.0, 0.0)


def _rows(points: List[tuple], dt: float = 0.1, up_z: float = 1.0) -> List[mx.TraceRow]:
    return [mx.TraceRow(round(k * dt, 6), x, y, yaw, up_z) for k, (x, y, yaw) in enumerate(points)]


def _straight(n: int = 51, end=(5.0, 5.0), yaw: float = 0.0) -> List[mx.TraceRow]:
    xs = np.linspace(2.0, end[0], n)
    ys = np.linspace(1.0, end[1], n)
    return _rows([(x, y, yaw) for x, y in zip(xs, ys)])


def test_pose_error_wraps() -> None:
    assert mx.pose_error(Pose2D(0, 0, math.radians(179)), Pose2D(0, 0, math.radians(-179)))[
        1
    ] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "dx,yaw_deg,ok",
    [
        (0.5, 0.0, True),
        (0.0, 45.0, True),
        (0.5001, 0.0, False),
        (0.0, 45.01, False),
        (0.3, -44.9, True),
    ],
)
def test_within_pose_boundaries(dx: float, yaw_deg: float, ok: bool) -> None:
    pose = Pose2D(GOAL.x + dx, GOAL.y, math.radians(yaw_deg))
    assert mx.within_pose(pose, GOAL) is ok


def test_success_needs_declared_done() -> None:
    trace = _straight()
    timeout = mx.evaluate_run(
        trace, GOAL, 5.0, declared_done=False, termination="timeout", decisions=9
    )
    assert (
        not timeout.success
        and timeout.ever_within_pose
        and timeout.spl == 0.0
        and timeout.plr is None
    )
    done = mx.evaluate_run(trace, GOAL, 5.0, declared_done=True, termination="done", decisions=9)
    assert done.success and done.success_loose and not done.premature_done
    assert done.path_length_m == pytest.approx(5.0, abs=1e-6)
    assert done.spl == pytest.approx(1.0) and done.plr == pytest.approx(1.0)


@pytest.mark.parametrize("scene", [None, oracle.SCENE_ENV0, oracle.SCENE_ENV1, oracle.SCENE_ENV3])
def test_only_multifloor_path_length_includes_height(scene: Optional[str]) -> None:
    trace = _straight()
    for i, row in enumerate(trace):
        row.z = 3.0 * i / (len(trace) - 1)
    xy_length = mx.path_length_m([(r.x, r.y) for r in trace])

    out = mx.evaluate_run(
        trace, GOAL, 5.0, True, "done", 9, goal_scene=scene,
        goal_floor="f2" if scene == oracle.SCENE_ENV3 else None,
    )

    if scene == oracle.SCENE_ENV3:
        assert out.path_length_m == pytest.approx(math.sqrt(5.0**2 + 3.0**2))
        assert out.path_length_m > xy_length
    else:
        # Reuse the existing 3-4-5 trace, including the original floating-point accumulation.
        assert out.path_length_m.hex() == xy_length.hex()
        assert out.path_length_m == pytest.approx(5.0, abs=1e-6)


def test_loose_criterion_2m_90deg() -> None:
    """Relaxed success is 2 m / 90 degrees; both criteria require declaring done."""
    out = mx.evaluate_run(
        _straight(end=(6.5, 5.0), yaw=math.radians(80)), GOAL, 5.0, True, "done", 3
    )
    assert not out.success and out.success_loose and out.premature_done
    assert out.final_dist_m == pytest.approx(1.5) and out.final_yaw_err_deg == pytest.approx(80.0)
    for end, yaw_deg, ok in (((7.0, 5.0), 0.0, True), ((7.01, 5.0), 0.0, False),
                             ((5.0, 5.0), 90.0, True), ((5.0, 5.0), 90.5, False)):  # fmt: skip
        pose = Pose2D(end[0], end[1], math.radians(yaw_deg))
        assert mx.within_loose(pose, GOAL) is ok


def test_method_error_never_success() -> None:
    out = mx.evaluate_run(
        _straight(), GOAL, 5.0, declared_done=True, termination="method_error", decisions=1
    )
    assert not out.success and not out.success_loose and not out.premature_done


def test_fall_detection_and_truncation() -> None:
    pts = [(2.0 + 0.1 * k, 1.0, 0.0) for k in range(40)]
    rows = _rows(pts)
    for r in rows[10:19]:  # Nine inverted samples span only 0.8 s, so they do not trigger a fall
        r.up_z = 0.3
    assert mx.find_fall(rows) is None
    for r in rows[20:]:  # Continuous inversion from t=2.0 triggers a fall at t=3.0
        r.up_z = 0.2
    out = mx.evaluate_run(rows, GOAL, 5.0, declared_done=True, termination="done", decisions=4)
    assert out.termination == "fallen" and out.falls == 1 and out.fallen_at_s == pytest.approx(3.0)
    assert out.path_length_m == pytest.approx(3.0, abs=1e-6) and not out.success


def test_path_hysteresis() -> None:
    rng = np.random.default_rng(1)
    jitter = [(1.0 + rng.uniform(-0.02, 0.02), 1.0 + rng.uniform(-0.02, 0.02)) for _ in range(300)]
    assert mx.path_length_m(jitter) < 0.06
    line = [(0.01 * k, 0.0) for k in range(301)]
    assert mx.path_length_m(line) == pytest.approx(3.0, abs=1e-6)
    assert mx.path_length_m([(0.0, 0.0)]) == 0.0


def test_spl_plr_edges() -> None:
    assert mx.spl(True, 5.0, 4.0) == 1.0  # Clamp to one if quantization makes the measured path
    # shorter than the shortest path
    assert mx.spl(True, 5.0, 10.0) == pytest.approx(0.5)
    assert mx.plr(True, 5.0, 10.0) == pytest.approx(2.0)
    assert mx.spl(False, 5.0, 5.0) == 0.0 and mx.plr(False, 5.0, 5.0) is None
    assert mx.spl(True, 0.0, 1.0) == 0.0


def test_stuck_monitor() -> None:
    still = mx.StuckMonitor()
    for k in range(301):  # Standing still for 30 s counts stuck events at 15 s and 30 s
        still.update(k * 0.1, 1.0, 1.0)
    assert len(still.events) == 2
    moving = mx.StuckMonitor()
    for k in range(301):
        moving.update(k * 0.1, 0.05 * k * 0.1, 0.0)  # At 0.05 m/s, travel 0.75 m in 15 s
    assert moving.events == []


def test_proximity() -> None:
    depth = np.full((480, 640), 3.0, np.float32)
    assert not mx.is_proximity(depth)
    depth[160:320, 213:426] = 0.2
    assert mx.is_proximity(depth)
    assert not mx.is_proximity(np.zeros((480, 640), np.float32))


def test_latency_and_realtime_factor() -> None:
    out = mx.evaluate_run(
        _straight(), GOAL, 5.0, True, "done", decisions=5, act_wall_s=[1.0, 2.0, 3.0, 4.0, 10.0]
    )
    assert out.act_wall_s["median"] == pytest.approx(3.0) and out.act_wall_s["max"] == 10.0
    assert out.compute_per_sim_s == pytest.approx(20.0 / 5.0)
    assert mx.latency_stats([])["p95"] is None


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        mx.evaluate_run(_straight(), GOAL, 5.0, True, "exploded", decisions=1)
    with pytest.raises(ValueError):
        mx.evaluate_run([], GOAL, 5.0, True, "done", decisions=1)


def test_bootstrap_ci() -> None:
    assert mx.bootstrap_ci([1.0] * 8) == (1.0, 1.0)
    lo, hi = mx.bootstrap_ci([0, 1, 0, 1, 1, 0, 1, 0])
    assert 0.0 <= lo < 0.5 < hi <= 1.0
