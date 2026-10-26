"""Comparison-video data and panel rendering tests, without video encoding.

Check grid layout, crop bounds and panel pixel properties.
"""

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from vnav_benchmark.oracle import TruthGrid
from vnav_benchmark.report import compare_video as cv_
from vnav_benchmark.report.episode_data import load_episode

TOKEN = "ep_test000001"
ASTRA_METHODS = tuple(mid for mid in cv_.REPORT_METHOD_IDS if not mid.endswith("-qwen"))


def _grid() -> TruthGrid:
    free = np.ones((100, 100), bool)
    free[:, :3] = False
    return TruthGrid("synthetic", 0.0, 0.0, 0.1, free, ~free, (), ())


def _run(root: Path, method_id: str, success: bool, termination: str) -> Path:
    run = root / "runs" / method_id / TOKEN / "r0"
    (run / "goal").mkdir(parents=True)
    ok, jpg = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))
    (run / "goal" / "goal.jpg").write_bytes(jpg.tobytes())
    rows = ["t_sim,x,y,yaw,up_z,vx_body,wz_body"]
    rows += [f"{0.1 * k:.6f},{1 + 0.05 * k:.6f},2.000000,0,1,0.5,0" for k in range(60)]
    (run / "trace.csv").write_text("\n".join(rows) + "\n")
    (run / "steps.jsonl").write_text("")
    doc = {
        "method_id": method_id,
        "status": "scored",
        "episode": {"scene": "synthetic", "start": {"x": 1, "y": 2, "yaw": 0},
                    "goal": {"x": 4, "y": 2, "yaw": 0}, "waypoints": [[1, 2], [4, 2]],
                    "tier": "near", "shortest_m": 3.0},
        "outcome": {"success": success, "termination": termination, "sim_time_s": 5.9},
    }  # fmt: skip
    (run / "run.json").write_text(json.dumps(doc))
    return run


def test_find_and_order(tmp_path: Path) -> None:
    _run(tmp_path, "dense-contloc-ptgoal", True, "done")
    _run(tmp_path, "nomap-noloc-imggoal-qwen", False, "timeout")
    runs = cv_.find_runs(tmp_path, TOKEN, 0)
    assert sorted(runs) == ["dense-contloc-ptgoal", "nomap-noloc-imggoal-qwen"]
    order = cv_.cell_order(list(runs))
    assert (
        len(order) == 12
        and order[0] == "dense-contloc-ptgoal"
        and order[-1] == "nomap-noloc-imggoal-qwen"
    )
    assert order[1:-1] == [None] * 10
    selected = ("nomap-noloc-imggoal-qwen", "sparse-contloc-ptgoal", "dense-contloc-ptgoal")
    assert cv_.cell_order(list(runs), selected, cols=4, rows=2) == [selected[0], None, selected[2]]


def test_cell_pixels(tmp_path: Path) -> None:
    run = _run(tmp_path, "sparse-contloc-imggoal-astra", False, "done")
    ep = load_episode(run, lambda scene: _grid())
    box = cv_.crop_box([ep])
    assert box[2] - box[0] == box[3] - box[1]  # Square bounds.
    view = cv_.MapView(ep.grid, box, 300)
    color = cv_.code_color_bgr(ep.code)
    mid = cv_.render_cell(view, ep, ep.method_id, 2.0)
    end = cv_.render_cell(view, ep, ep.method_id, 99.0)
    trail = lambda img: int((np.abs(img.astype(int) - np.array(color)).sum(axis=2) < 10).sum())  # noqa: E731
    assert 0 < trail(mid) < trail(end)  # The displayed trajectory grows with time.
    assert cv_.outcome_chip(ep)[0].startswith("WRONG STOP")
    chip = (slice(cv_.CELL_TITLE_H + 8, cv_.CELL_TITLE_H + 22), slice(60, 90))
    grey = np.array(cv_.RUNNING_BGR)
    near = lambda img, c: (np.abs(img[chip].astype(int) - np.array(c)).sum(axis=2) < 40).mean()  # noqa: E731
    # RUNNING is gray until the final outcome label replaces it.
    assert near(mid, grey) > near(end, grey)
    assert near(end, cv_.FAIL_BGR) > near(mid, cv_.FAIL_BGR)
    empty = cv_.render_cell(view, None, "nomap-noloc-imggoal-qwen", 2.0)
    region = empty[cv_.CELL_TITLE_H + 20 : cv_.CELL_TITLE_H + 100, 60:300]
    # Unrun configurations have empty panels.
    assert (region == np.array(cv_.EMPTY_BGR)).all(axis=2).mean() > 0.9


def test_side_panel_shape() -> None:
    side = cv_.render_side(b"", {"token": TOKEN, "pair": "near", "repeat": "0"}, 12.0)
    assert side.shape == (cv_.SIDE.h, cv_.SIDE.w, 3)


def test_default_grid_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the default 12-cell 1080p layout and sidebar; missing runs do not create goal
    cells.
    """
    _run(tmp_path, ASTRA_METHODS[0], True, "done")
    monkeypatch.setattr("vnav_benchmark.oracle.build_truth_grid", lambda scene: _grid())
    job = cv_.prepare_for(TOKEN, 0)(tmp_path, tmp_path, lambda msg: None)
    with patch.object(cv_, "render_cell", wraps=cv_.render_cell) as cells, \
            patch.object(cv_, "render_side", wraps=cv_.render_side) as side, \
            patch.object(cv_, "render_goal_cell") as goal:
        frame = job.frame(0)
    assert job.size == (1920, 1080)
    assert frame.shape == (1080, 1920, 3)
    assert cells.call_count == 12
    assert all(call.args[4:] == (380, 340, False) for call in cells.call_args_list)
    assert sum(call.args[1] is None for call in cells.call_args_list) == 11
    side.assert_called_once()
    goal.assert_not_called()
    expected = cv_.render_side(side.call_args.args[0], side.call_args.args[1], 0.0)
    np.testing.assert_array_equal(frame[cv_.SIDE.y:cv_.SIDE.y1, cv_.SIDE.x:cv_.SIDE.x1], expected)


@pytest.mark.parametrize("method_count", [6, 7])
def test_compact_grid_goal_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method_count: int
) -> None:
    methods = ASTRA_METHODS[:method_count]
    for mid in methods:
        _run(tmp_path, mid, mid != methods[-1], "timeout")
    # Unselected Qwen runs must affect neither loading, cropping nor the timeline.
    _run(tmp_path, "nomap-noloc-imggoal-qwen", False, "timeout")
    monkeypatch.setattr("vnav_benchmark.oracle.build_truth_grid", lambda scene: _grid())
    with patch.object(cv_, "load_episode", wraps=cv_.load_episode) as load:
        job = cv_.prepare_for(TOKEN, 0, methods, cols=4, rows=2, w=1280, h=720,
                              compact_labels=True)(tmp_path, tmp_path, lambda msg: None)
    assert load.call_count == method_count
    assert job.info["methods"] == sorted(methods)
    with patch.object(cv_, "render_cell", wraps=cv_.render_cell) as cells, \
            patch.object(cv_, "render_goal_cell", wraps=cv_.render_goal_cell) as goals, \
            patch.object(cv_, "render_side") as side, \
            patch.object(cv_.draw, "blit_text", wraps=cv_.draw.blit_text) as text:
        frame = job.frame(10000)
    assert job.size == (1280, 720)
    assert frame.shape == (720, 1280, 3)
    assert cells.call_count == method_count
    assert goals.call_count == 8 - method_count
    assert all(call.args[4:] == (320, 360, True) for call in cells.call_args_list)
    side.assert_not_called()
    assert "not run" not in [call.args[3] for call in text.call_args_list]
    # The last cell's upper half must display the gray-128 goal.jpg, not a placeholder or text.
    np.testing.assert_array_equal(frame[420:530, 960:1280], 128)


def test_compact_labels_omit_subtitle(tmp_path: Path) -> None:
    ep = load_episode(_run(tmp_path, ASTRA_METHODS[2], False, "timeout"), lambda scene: _grid())
    view = cv_.MapView(ep.grid, cv_.crop_box([ep]), 300)
    for compact in (False, True):
        with patch.object(cv_.draw, "blit_text", wraps=cv_.draw.blit_text) as text, \
                patch.object(cv_.draw, "label_chip", wraps=cv_.draw.label_chip) as chip:
            cv_.render_cell(view, ep, ep.method_id, 99.0, compact_labels=compact)
        labels = [call.args[3] for call in text.call_args_list]
        assert cv_.method_label(ep.method_id) in labels
        assert (ep.method_id in labels) is (not compact)
        assert "TIMEOUT 6 s" in chip.call_args.args[0]
        assert chip.call_args.args[1] == (cv_.COMPACT_CHIP_PX if compact else 15)


def test_goal_cell_route_room_and_missing_photo() -> None:
    info = {"token": TOKEN, "goal_room": "open_office"}
    with patch.object(cv_.draw, "blit_text", wraps=cv_.draw.blit_text) as text:
        cell = cv_.render_goal_cell(b"", info, 90.0, 320, 360)
    labels = [call.args[3] for call in text.call_args_list]
    assert TOKEN in labels and "Goal: open_office" in labels
    assert "Goal + 0.5 m success radius" in labels
    np.testing.assert_array_equal(cell[54:180], np.broadcast_to(cv_.EMPTY_BGR, (126, 320, 3)))
    with patch.object(cv_.draw, "blit_text", wraps=cv_.draw.blit_text) as text:
        cv_.render_goal_cell(b"", {"token": TOKEN}, 0.0, 320, 360)
    assert "Goal image" in [call.args[3] for call in text.call_args_list]


def test_grid_fills_indivisible_canvas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(tmp_path, ASTRA_METHODS[0], True, "done")
    monkeypatch.setattr("vnav_benchmark.oracle.build_truth_grid", lambda scene: _grid())
    job = cv_.prepare_for(TOKEN, 0, ASTRA_METHODS, cols=4, rows=2, w=1281, h=721,
                          compact_labels=True)(tmp_path, tmp_path, lambda msg: None)
    def cell(view, ep, mid, t, w, h, compact):
        return np.full((h, w, 3), 100, np.uint8)

    def goal(jpg, info, t, w, h):
        return np.full((h, w, 3), 100, np.uint8)

    monkeypatch.setattr(cv_, "render_cell", cell)
    monkeypatch.setattr(cv_, "render_goal_cell", goal)
    assert (job.frame(0) == 100).all()


def test_cli_defaults_and_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    args = cv_.build_parser().parse_args(["/tmp/campaign", "--token", TOKEN])
    assert args.methods is None and not args.compact_labels
    assert (args.cols, args.rows, args.canvas_w, args.canvas_h) == (4, 3, 1920, 1080)

    def run_cli(parser, prepare_for, argv):
        prepare_for(parser.parse_args(argv))
        return 0

    monkeypatch.setattr(cv_.video, "run_cli", run_cli)
    with patch.object(cv_, "prepare_for") as prepare:
        assert cv_.main(["/tmp/campaign", "--token", TOKEN, "--methods", ",".join(ASTRA_METHODS),
                         "--cols", "4", "--rows", "2", "--canvas-w", "1280", "--canvas-h", "720",
                         "--compact-labels", "--stills"]) == 0
    prepare.assert_called_once_with(TOKEN, 0, grid_method_ids=list(ASTRA_METHODS), cols=4, rows=2,
                                    w=1280, h=720, compact_labels=True)


@pytest.mark.parametrize("kwargs", [
    {"cols": 0}, {"h": -1}, {"grid_method_ids": []},
    {"grid_method_ids": [ASTRA_METHODS[0]] * 2}, {"grid_method_ids": ["typo"]},
    {"cols": 4, "rows": 2},
])
def test_invalid_layout(kwargs: dict) -> None:
    with pytest.raises(cv_.video.VideoError):
        cv_.prepare_for(TOKEN, 0, **kwargs)
