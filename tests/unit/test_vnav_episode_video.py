"""Episode-data and video-rendering tests use synthetic episode directories without
simulation/network.

Pixel checks inspect coarse color statistics in important regions rather than exact golden
images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest
from vnav_episode_fixtures import fake_grid, frame_color_bgr, make_episode_dir

from frontier_nav_bench.report import episode_video as ev
from frontier_nav_bench.report.episode_data import load_episode
from frontier_nav_bench.report.results_io import REPORT_METHOD_IDS

# Reuse ev.video so tests exercise the same shared video backend as the episode renderer.
# This also keeps imports aligned with the renderer's vendored visualization dependencies.
video = ev.video


# --- 1. load_episode ---
def test_load_episode_counts_and_fields(tmp_path: Path) -> None:
    run_dir = make_episode_dir(tmp_path, "dense-initloc-imggoal-astra", n_decisions=12)
    ep = load_episode(run_dir, grid_fn=lambda _scene: fake_grid())

    assert ep.method_id == "dense-initloc-imggoal-astra"
    # results_io.METHOD_CODES maps initloc + imggoal to M4 among seven methods / 12 configurations.
    assert ep.code == "M4"
    assert ep.planner == "imggoal" and ep.vlm == "astra"
    assert len(ep.decisions) == 12
    # A 10 Hz trace covers 12 s duration, 1 s hold and 1 s buffer, including both endpoints.
    assert ep.trace.shape[0] == int(round((12 * 1.0 + 1.0 + 1.0) * 10)) + 1
    assert len(ep.ego_frames) == ep.trace.shape[0]
    assert len(ep.depth_frames) == 12  # One depth frame per decision.
    assert ep.goal_jpeg
    assert ep.start == (12.5, 2.6, 0.0)
    assert ep.goal == (15.5, 2.6, 0.0)
    assert ep.waypoints == [(12.5, 2.6), (15.5, 2.6)]


def test_decision_at_frame_at_boundaries(tmp_path: Path) -> None:
    run_dir = make_episode_dir(tmp_path, "sparse-contloc-ptgoal", n_decisions=10, success=True)
    ep = load_episode(run_dir, grid_fn=lambda _scene: fake_grid())

    assert ep.decision_at(-1.0) is None  # Before the first decision.
    assert ep.decision_at(0.0).i == 0
    assert ep.decision_at(0.05).i == 0  # Between decisions with 0.2 s hold time.
    assert ep.decision_at(100.0).i == 9  # Beyond the end, retain the last decision.

    assert ep.frame_at(-1.0) is None  # Before the first frame.
    first_path = ep.frame_at(0.0)
    assert first_path is not None and first_path.name == "00000000.jpg"
    # Between 10 Hz frames, choose the latest timestamp <=0.15 s, namely 0.1 s.
    mid_path = ep.frame_at(0.15)
    assert mid_path is not None and mid_path.name == "00000100.jpg"
    last_path = ep.frame_at(1e6)
    assert last_path == ep.ego_frames[-1][1]

    assert ep.depth_at(-1.0) is None
    assert ep.depth_at(1e6) == ep.depth_frames[-1][1]


def test_pose_at_interpolation(tmp_path: Path) -> None:
    run_dir = make_episode_dir(tmp_path, "dense-contloc-ptgoal", n_decisions=10, success=True)
    ep = load_episode(run_dir, grid_fn=lambda _scene: fake_grid())

    x0, y0, yaw0 = ep.pose_at(0.0)
    assert (x0, y0) == pytest.approx((12.5, 2.6), abs=1e-6)
    assert yaw0 == pytest.approx(0.0, abs=1e-6)

    duration_s = 10 * 0.2  # Point-goal hold_s is 0.2 s; see the fixture.
    x_end, y_end, _ = ep.pose_at(duration_s)
    assert (x_end, y_end) == pytest.approx((15.5, 2.6), abs=1e-6)

    x_half, _y_half, _ = ep.pose_at(duration_s / 2.0)
    assert x_half == pytest.approx((12.5 + 15.5) / 2.0, abs=1e-2)

    # Clamp to the endpoint beyond the trace; do not extrapolate.
    x_over, y_over, _ = ep.pose_at(duration_s + 100.0)
    assert (x_over, y_over) == pytest.approx((x_end, y_end), abs=1e-6)


def test_history_jpeg_counts_and_subgoal(tmp_path: Path) -> None:
    run_dir = make_episode_dir(tmp_path, "dense-initloc-imggoal-astra", n_decisions=12)
    ep = load_episode(run_dir, grid_fn=lambda _scene: fake_grid())
    counts = [len(d.history_jpegs) for d in ep.decisions[:5]]
    assert counts == [0, 1, 2, 2, 3]  # VlmMemory follows HISTORY_OFFSETS = (8, 4, 2, 1).
    # M4 starts in NAVIGATE with subgoal images (dense map, initloc image-goal control).
    assert ep.decisions[0].subgoal_jpeg is not None

    m6_dir = make_episode_dir(tmp_path, "nomap-noloc-imggoal-qwen", n_decisions=12)
    m6 = load_episode(m6_dir, grid_fn=lambda _scene: fake_grid())
    # M6 has no map and never receives subgoal images.
    assert all(d.subgoal_jpeg is None for d in m6.decisions)
    assert m6.decisions[0].vlm is not None  # VLM logs still exist even without a subgoal.


def test_contloc_imggoal_m2_m3_also_have_subgoal_images(tmp_path: Path) -> None:
    """M2/M3 image-goal methods also have VLM logs and subgoal photographs.

    They render like M4/M5/M6, while arrival is judged by pose.
    """
    for method_id in ("dense-contloc-imggoal-astra", "sparse-contloc-imggoal-qwen"):
        run_dir = make_episode_dir(tmp_path / method_id, method_id, n_decisions=12)
        ep = load_episode(run_dir, grid_fn=lambda _scene: fake_grid())
        assert ep.vlm is not None
        assert ep.decisions[0].vlm is not None
        assert ep.decisions[0].subgoal_jpeg is not None


def test_subgoal_chip_label_follows_progress_judge_not_planner() -> None:
    """Subgoal chips follow fairness.parse_method_id(...).progress_judge.

    Continuous-localization image-goal methods and point-goal methods are judged by pose;
    initloc/noloc image-goal methods M4/M5/M6 are judged by the VLM.
    """
    assert ev._subgoal_chip_text("dense-contloc-ptgoal") == "judged by pose"
    assert ev._subgoal_chip_text("dense-contloc-imggoal-astra") == "judged by pose"
    assert ev._subgoal_chip_text("sparse-contloc-imggoal-qwen") == "judged by pose"
    assert ev._subgoal_chip_text("dense-initloc-imggoal-astra") == "judged by VLM"
    assert ev._subgoal_chip_text("sparse-initloc-imggoal-qwen") == "judged by VLM"
    assert ev._subgoal_chip_text("nomap-noloc-imggoal-astra") == "judged by VLM"


def test_ptgoal_has_no_vlm_fields(tmp_path: Path) -> None:
    run_dir = make_episode_dir(tmp_path, "sparse-contloc-ptgoal", n_decisions=8, success=True)
    ep = load_episode(run_dir, grid_fn=lambda _scene: fake_grid())
    assert ep.vlm is None
    for d in ep.decisions:
        assert d.vlm is None
        assert d.history_jpegs == ()
        assert d.subgoal_jpeg is None
        assert d.usage_usd == 0.0 and d.tokens_in == 0 and d.tokens_out == 0


# Section 2: prepare jobs for seven methods / 12 configurations using REPORT_METHOD_IDS.
# Use results_io method codes and colors independently of fairness.V1_METHOD_IDS.
def test_report_method_ids_cover_7_methods_12_configs() -> None:
    assert len(REPORT_METHOD_IDS) == 12


@pytest.mark.parametrize("method_id", REPORT_METHOD_IDS)
def test_prepare_builds_job_for_all_report_methods(tmp_path: Path, method_id: str) -> None:
    run_dir = make_episode_dir(tmp_path / method_id, method_id, n_decisions=6, success=True)
    job = ev.prepare(run_dir, tmp_path / "frames", lambda _msg: None)

    assert job.size == (ev.W, ev.H)
    for k in (0, job.timeline.n_run // 2, job.timeline.n_total - 1):
        frame = job.frame(k)
        assert frame.shape == (ev.H, ev.W, 3)
        assert frame.dtype == np.uint8


# Section 3: pixel checks.
def _prepared(tmp_path: Path, method_id: str, n_decisions: int = 12, success: bool = True):
    run_dir = make_episode_dir(tmp_path, method_id, n_decisions=n_decisions, success=success)
    return ev.prepare(run_dir, tmp_path / "frames", lambda _msg: None)


def _mean_bgr(frame: np.ndarray, rect) -> Tuple[float, float, float]:
    patch = frame[rect.y : rect.y1, rect.x : rect.x1]
    b, g, r = (float(patch[:, :, i].mean()) for i in range(3))
    return b, g, r


def test_title_bar_has_dark_text(tmp_path: Path) -> None:
    job = _prepared(tmp_path, "dense-initloc-imggoal-astra")
    frame = job.frame(0)
    title_strip = frame[10:40, 16:600]
    assert title_strip.min() < 80  # Dark text must appear over the white background.


def test_ego_panel_matches_fixture_frame_color(tmp_path: Path) -> None:
    run_dir = make_episode_dir(tmp_path, "dense-initloc-imggoal-astra", n_decisions=12)
    job = ev.prepare(run_dir, tmp_path / "frames", lambda _msg: None)
    t, _final = job.timeline.time_at(0)
    expected = frame_color_bgr(0)  # k=0 corresponds to t=0 and the t_ms=0 frame.
    b, g, r = _mean_bgr(job.frame(0), ev.EGO)
    assert abs(b - expected[0]) < 20 and abs(g - expected[1]) < 20 and abs(r - expected[2]) < 20


def test_topdown_has_method_color_trajectory(tmp_path: Path) -> None:
    job = _prepared(tmp_path, "sparse-initloc-imggoal-qwen")  # M5
    k = job.timeline.n_run // 2  # At the midpoint, some trajectory should already be visible.
    frame = job.frame(k)
    code_color = ev._code_color_bgr("M5")
    top_region = frame[ev.TOP.y : ev.TOP.y1, ev.TOP.x : ev.TOP.x1]
    diff = np.abs(top_region.astype(np.int32) - np.array(code_color, dtype=np.int32)).sum(axis=2)
    # Some pixels should match the method color in the trace, goal star or current-pose triangle.
    assert bool((diff < 10).any())


KF_COLORS = {0: (40, 40, 200), 1: (40, 200, 40), 2: (200, 40, 40), 3: (40, 200, 200),
             4: (200, 40, 200), 5: (200, 200, 40)}  # fmt: skip


def _kf_jpeg(kf: int) -> bytes:
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = KF_COLORS[kf]
    return cv2.imencode(".jpg", img)[1].tobytes()


def _with_route(tmp_path: Path, method_id: str, cur_kf: int = 1):  # type: ignore[no-untyped-def]
    """Create a six-keyframe synthetic route with solid-color images and cur_kf as every decision's
    subgoal.
    """
    import dataclasses

    from frontier_nav_bench.report.episode_data import RouteKeyframes

    run_dir = make_episode_dir(tmp_path, method_id)
    route = RouteKeyframes(tuple(range(6)), {k: (1.0 + k, 1.0, 0.0) for k in range(6)}, _kf_jpeg)
    ep = load_episode(run_dir, grid_fn=lambda _s: fake_grid(), route_fn=lambda _d, _r: route)
    ep.decisions = [dataclasses.replace(d, subgoal_kf=cur_kf) for d in ep.decisions]
    return ev.job_for(ep)


def _close(bgr: Tuple[float, float, float], want: Tuple[int, int, int], tol: float = 25) -> bool:
    return all(abs(a - b) < tol for a, b in zip(bgr, want))


def test_ego_is_60_percent_wide_4_by_3_and_has_no_action_arrow(tmp_path: Path) -> None:
    assert 0.58 <= ev.EGO.w / ev.W <= 0.61 and abs(ev.EGO.w * 3 - ev.EGO.h * 4) <= 4
    frame = _prepared(tmp_path, "dense-contloc-ptgoal").frame(5)
    ego = frame[ev.EGO.y : ev.EGO.y1, ev.EGO.x : ev.EGO.x1].astype(int)
    # Fluorescent lime color formerly used by the action arrow.
    lime = np.abs(ego - np.array((50, 255, 120))).sum(axis=2) < 40
    assert lime.sum() < 20


def test_panels_tile_canvas_without_overlap(tmp_path: Path) -> None:
    rects = [ev.TITLE, ev.EGO, ev.GOAL, ev.SUBGOAL, ev.TOP, ev.STATUS, ev.TEXT]
    rects += [ev._thumb_rect(k) for k in range(5)]
    cover = np.zeros((ev.H, ev.W), np.int32)
    for r in rects:
        assert 0 <= r.x and r.x1 <= ev.W and 0 <= r.y and r.y1 <= ev.H
        cover[r.y : r.y1, r.x : r.x1] += 1
    assert cover.max() == 1  # Panels must not overlap.
    assert (cover > 0).mean() > 0.95  # Panels fill the canvas, leaving only narrow gaps.


def test_ptgoal_strip_shows_route_ahead_and_subgoal_photo(tmp_path: Path) -> None:
    frame = _with_route(tmp_path, "dense-contloc-ptgoal", cur_kf=1).frame(3)
    # The subgoal cell displays the current keyframe photograph.
    assert _close(_mean_bgr(frame, ev.SUBGOAL), KF_COLORS[1])
    # Four lower thumbnails show upcoming keyframes 2-5 rather than empty VLM cells.
    for k in range(4):
        r = ev._thumb_rect(k + 1)
        inner = ev.draw.Rect(r.x, r.y + 40, r.w, r.h - 40)  # Avoid the corner label chip.
        assert _close(_mean_bgr(frame, inner), KF_COLORS[2 + k])


def test_topdown_marks_current_subgoal_and_route_keyframes(tmp_path: Path) -> None:
    frame = _with_route(tmp_path, "sparse-contloc-ptgoal", cur_kf=2).frame(3)
    top = frame[ev.TOP.y : ev.TOP.y1, ev.TOP.x : ev.TOP.x1].astype(int)
    orange = np.abs(top - np.array(ev.SUBGOAL_MARK_BGR)).sum(axis=2) < 40
    assert orange.sum() > 60  # Orange diamond for the current subgoal.
    grey = np.abs(top - np.array(ev.KF_PASSED_BGR)).sum(axis=2) < 10
    assert grey.sum() > 20  # Gray dots for traversed keyframes 0 and 1.


def test_m6_has_no_subgoal_image_and_vlm_history_strip(tmp_path: Path) -> None:
    # M6 has no map and never receives a subgoal image.
    job = _prepared(tmp_path, "nomap-noloc-imggoal-astra")
    frame = job.frame(job.timeline.n_run - 1)
    b, g, r = _mean_bgr(frame, ev.SUBGOAL)
    # Gray placeholder background.
    assert abs(b - 225) < 6 and abs(g - 225) < 6 and abs(r - 225) < 6
    hist = ev._thumb_rect(4)  # The d-1 image contains content rather than placeholder gray.
    assert np.std(frame[hist.y + 40 : hist.y1, hist.x : hist.x1]) > 1 or not _close(
        _mean_bgr(frame, hist), (225, 225, 225), 3
    )


def test_final_frame_banner_success_green_failure_red(tmp_path: Path) -> None:
    job_ok = _prepared(tmp_path / "ok", "dense-contloc-ptgoal", n_decisions=8, success=True)
    last_ok = job_ok.frame(job_ok.timeline.n_total - 1)
    cx, cy = ev.W // 2, ev.H // 2
    b, g, r = last_ok[cy, cx].tolist()
    assert g > r and g > b  # Green dominates the banner.

    job_bad = _prepared(tmp_path / "bad", "sparse-contloc-ptgoal", n_decisions=8, success=False)
    last_bad = job_bad.frame(job_bad.timeline.n_total - 1)
    b2, g2, r2 = last_bad[cy, cx].tolist()
    assert r2 > g2 and r2 > b2  # Red dominates the banner.


# --- 4. --stills ---
def test_stills_produces_seven_full_size_jpegs(tmp_path: Path) -> None:
    run_dir = make_episode_dir(tmp_path / "src", "dense-initloc-imggoal-astra", n_decisions=8)
    out_dir = tmp_path / "out"
    result = video.make_video(
        run_dir, ev.prepare, out=out_dir / "demo", stills=True, log=lambda _msg: None
    )
    assert len(result.stills) == 7
    for path in result.stills:
        assert path.is_file()
        img = cv2.imread(str(path))
        assert img is not None
        assert img.shape == (ev.H, ev.W, 3)
