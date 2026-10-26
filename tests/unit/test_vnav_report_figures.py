"""Render SR/SPL, cost/SR, difficulty-tier SR and trajectory figures from synthetic data."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from vnav_report_fixtures import synthetic

from vnav_benchmark.report import aggregate as agg
from vnav_benchmark.report import figures as fg
from vnav_benchmark.report.colors import code_color_hex
from vnav_benchmark.report.results_io import row_from_run


def _rows():  # type: ignore[no-untyped-def]
    return agg.scored([row_from_run(r, s) for r, s in synthetic()])


def _fake_grid() -> SimpleNamespace:
    free = np.ones((120, 100), bool)
    blocked = np.zeros_like(free)
    blocked[[0, -1], :] = True
    blocked[:, [0, -1]] = True
    blocked[60, 20:80] = True
    return SimpleNamespace(free=free & ~blocked, raw_blocked=blocked, x0=-1.0, y0=-2.0, cell_m=0.1)


def _tracks(rows):  # type: ignore[no-untyped-def]
    out = []
    for r in rows:
        k = int(r.episode_id[2:])
        xy = np.linspace([0.0, 0.0], [8.0, 1.0 + k], 30)
        out.append(
            fg.Track(
                r.method_id, r.code, r.vlm, r.episode_id, xy, r.success, (0.0, 0.0), (8.0, 1.0 + k)
            )
        )
    return out


def _png_ok(path: Path) -> np.ndarray:
    assert path.exists() and path.with_suffix(".pdf").exists()
    img = cv2.imread(str(path))
    assert img is not None and img.shape[0] > 200 and img.shape[1] > 200
    return img


def test_all_figures_render(tmp_path: Path) -> None:
    rows = _rows()
    out = fg.build_all_figures(rows, tmp_path, _tracks(rows), _fake_grid())
    assert set(out) == {"sr_spl", "sr_by_distance", "cost_vs_sr", "trajectories"}
    for path in out.values():
        _png_ok(path)


def test_bars_use_method_colors(tmp_path: Path) -> None:
    fg._style()  # Set typography.
    img = _png_ok(fg.sr_spl_bars(_rows(), tmp_path / "bars"))
    for code in ("B0", "M4"):
        rgb = np.array([int(code_color_hex(code)[i : i + 2], 16) for i in (1, 3, 5)])
        bgr = rgb[::-1]
        assert (np.abs(img.astype(int) - bgr).sum(axis=2) < 30).sum() > 50, code


def test_ptgoal_only_campaign_skips_cost_plot(tmp_path: Path) -> None:
    rows = [r for r in _rows() if r.vlm is None]
    out = fg.build_all_figures(rows, tmp_path)
    assert set(out) == {"sr_spl", "sr_by_distance"}
