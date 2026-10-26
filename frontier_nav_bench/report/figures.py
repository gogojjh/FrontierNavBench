"""Paper figures: SR/SPL bars, cost versus SR, difficulty-tier SR and trajectory facets.

Use tools/viz/style.py for typography and report/colors.py for method colors, shared with
videos. Hatching distinguishes models (diagonal for Qwen). Labels are English; matplotlib 3.1 is
supported without bar_label or set_box_aspect. Each figure writes PNG and PDF.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from frontier_nav_bench.metrics import bootstrap_ci  # noqa: E402
from frontier_nav_bench.report import aggregate as agg  # noqa: E402
from frontier_nav_bench.report.colors import code_color_hex  # noqa: E402
from frontier_nav_bench.report.results_io import RunRow  # noqa: E402
from frontier_nav_bench.report.tables import loose_label  # noqa: E402

MODEL_HATCH = {None: "", "astra": "", "qwen": "///"}
DPI = 200
FONT_SIZE = 11
DIST_JITTER_M = 0.12  # Horizontal method offsets in the difficulty-tier SR figure, meters.
SCALE_BAR_M = 5.0  # Trajectory-panel scale-bar length, meters.
TRAJ_COLS = 4  # Trajectory panels per row.
TRAJ_PANEL_IN = 3.2  # Panel side length, inches.


def _style() -> Any:
    from frontier_nav_bench.report import style

    style.setting_font(FONT_SIZE, FONT_SIZE, FONT_SIZE - 1)
    # Labels contain percent and degree symbols; disable LaTeX text rendering.
    matplotlib.rcParams["text.usetex"] = False
    return style


def _label(r: RunRow) -> str:
    return r.code if r.vlm is None else f"{r.code}-{r.vlm}"


def _color(style: Any, code: str) -> str:
    return code_color_hex(code)  # Method colors are defined only in report/colors.py.


def _save(fig: Any, out_stem: Path) -> Path:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_stem.with_suffix(".png")), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_stem.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    return out_stem.with_suffix(".png")


def _per_method(rows: Sequence[RunRow]) -> List[Tuple[str, List[RunRow]]]:
    groups = agg.group(rows, lambda r: r.method_id)
    return [(m, groups[m]) for m in agg.ordered_methods(rows)]


def sr_spl_bars(rows: Sequence[RunRow], out_stem: Path) -> Path:
    """Draw primary SR with endpoint-bootstrap 95% intervals, loose SR and SPL panels."""
    style = _style()
    methods = _per_method(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.6))
    x = np.arange(len(methods))
    panels = ((axes[0], "sr", "SR@0.5m/45° (%)"), (axes[1], "loose", f"{loose_label()} (%)"),
              (axes[2], "spl", "SPL"))  # fmt: skip
    for ax, metric, ylabel in panels:
        for i, (_m, rs) in enumerate(methods):
            vals = [
                float(r.success) if metric == "sr"
                else float(r.success_loose) if metric == "loose" else r.spl
                for r in rs
            ]  # fmt: skip
            scale = 1.0 if metric == "spl" else 100.0
            mean = float(np.mean(vals)) * scale
            lo, hi = (v * scale for v in bootstrap_ci(vals))
            ax.bar(
                x[i],
                mean,
                0.7,
                color=_color(style, rs[0].code),
                hatch=MODEL_HATCH[rs[0].vlm],
                edgecolor="black",
                linewidth=0.6,
            )
            ax.errorbar(
                x[i], mean, yerr=[[mean - lo], [hi - mean]], color="black", capsize=3, linewidth=0.8
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_label(rs[0]) for _m, rs in methods], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05 if metric == "spl" else 105)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_title("Success rate (95% CI over episode pairs)")
    axes[1].set_title("Looser success criterion")
    axes[2].set_title("Success weighted by path length")
    fig.tight_layout()
    return _save(fig, out_stem)


def cost_vs_sr(rows: Sequence[RunRow], out_stem: Path) -> Path:
    """Plot one point per VLM method: list-price dollars per episode versus success rate."""
    style = _style()
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for _m, rs in _per_method([r for r in rows if r.vlm is not None]):
        cost = float(np.mean([r.list_usd for r in rs]))
        sr = 100.0 * float(np.mean([r.success for r in rs]))
        marker = "o" if rs[0].vlm == "astra" else "s"
        ax.scatter(
            [cost],
            [sr],
            s=70,
            color=_color(style, rs[0].code),
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        # Stagger provider labels vertically so overlapping points remain readable.
        dy = 4 if rs[0].vlm == "astra" else -12
        ax.annotate(_label(rs[0]), (cost, sr), textcoords="offset points", xytext=(6, dy))
    ax.set_xlabel("List price per episode (USD)")
    ax.set_ylabel("SR@0.5m/45° (%)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.set_title("Cost vs. success (o: astra, s: qwen)")
    fig.tight_layout()
    return _save(fig, out_stem)


def sr_by_distance(rows: Sequence[RunRow], out_stem: Path) -> Path:
    """Plot SR by difficulty tier, ordered by mean shortest-path length."""
    style = _style()
    bands = agg.ordered_bands(rows)
    centers = [float(np.mean([r.shortest_m for r in rows if r.band == b])) for b in bands]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    methods = _per_method(rows)
    for k, (_m, rs) in enumerate(methods):
        # Offset methods horizontally to distinguish overlapping lines.
        dx = DIST_JITTER_M * (k - (len(methods) - 1) / 2)
        srs = []
        for b in bands:
            sub = [r.success for r in rs if r.band == b]
            srs.append(100.0 * float(np.mean(sub)) if sub else float("nan"))
        ax.plot(
            [c + dx for c in centers],
            srs,
            marker="o" if rs[0].vlm != "qwen" else "s",
            linestyle="-" if rs[0].vlm != "qwen" else "--",
            color=_color(style, rs[0].code),
            label=_label(rs[0]),
        )
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{b}\n({c:.1f} m)" for b, c in zip(bands, centers)])
    ax.set_xlabel("Difficulty tier (mean shortest path)")
    ax.set_ylabel("SR@0.5m/45° (%)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=FONT_SIZE - 3, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return _save(fig, out_stem)


@dataclass(frozen=True)
class Track:
    method_id: str
    code: str
    vlm: Optional[str]
    episode_id: str
    xy: np.ndarray  # Shape (N, 2), ground-truth trajectory in world coordinates.
    success: bool
    start: Tuple[float, float]
    goal: Tuple[float, float]


def load_tracks(campaign_dir: Path, rows: Sequence[RunRow]) -> List[Track]:
    """Read trace.csv and run.json endpoints for main-table episodes; these are evaluator-only
    data.
    """
    out: List[Track] = []
    for r in rows:
        d = campaign_dir / "runs" / r.method_id / r.episode_id / f"r{r.repeat}"
        trace = d / "trace.csv"
        if not trace.exists():
            continue
        with open(trace, newline="") as fh:
            xy = np.array([[float(row["x"]), float(row["y"])] for row in csv.DictReader(fh)])
        ep = json.loads((d / "run.json").read_text())["episode"]
        out.append(
            Track(
                r.method_id,
                r.code,
                r.vlm,
                r.episode_id,
                xy,
                r.success,
                (ep["start"]["x"], ep["start"]["y"]),
                (ep["goal"]["x"], ep["goal"]["y"]),
            )
        )
    return out


def trajectory_facets(tracks: Sequence[Track], grid: Any, out_stem: Path) -> Path:
    """Draw each method's trajectories over a truth grid, with solid successes and dashed failures.
    """
    style = _style()
    by = agg.group(tracks, lambda t: t.method_id)  # type: ignore[arg-type,return-value]
    methods = agg.ordered_methods(tracks)  # type: ignore[arg-type]  # Only .method_id is used.
    n_cols = min(TRAJ_COLS, max(1, len(methods)))
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(TRAJ_PANEL_IN * n_cols, TRAJ_PANEL_IN * n_rows), squeeze=False
    )
    nx, ny = grid.free.shape
    extent = (grid.x0, grid.x0 + nx * grid.cell_m, grid.y0, grid.y0 + ny * grid.cell_m)
    base = np.where(grid.raw_blocked.T, 0.35, np.where(grid.free.T, 1.0, 0.85))
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= len(methods):
            continue
        ts = by[methods[k]]
        ax.imshow(base, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1)
        color = _color(style, ts[0].code)
        for t in ts:
            ax.plot(
                t.xy[:, 0],
                t.xy[:, 1],
                color=color,
                linewidth=1.0,
                linestyle="-" if t.success else "--",
                alpha=0.9,
            )
            ax.plot(*t.start, marker="o", color="#2ca02c", markersize=4)
            ax.plot(*t.goal, marker="*", color="#d62728", markersize=7)
        n_ok = sum(t.success for t in ts)
        ax.set_title(f"{_track_label(ts[0])}  ({n_ok}/{len(ts)} success)", fontsize=FONT_SIZE - 1)
        ax.set_aspect("equal")
        if k == 0:  # Draw the scale bar only in the first panel's lower-left corner.
            x0 = extent[0] + 0.06 * (extent[1] - extent[0])
            y0 = extent[2] + 0.04 * (extent[3] - extent[2])
            ax.plot([x0, x0 + SCALE_BAR_M], [y0, y0], color="black", linewidth=2.5)
            ax.text(
                x0 + SCALE_BAR_M / 2,
                y0 + 0.4,
                f"{SCALE_BAR_M:.0f} m",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE - 2,
            )
    handles = [
        Line2D([], [], color="#2ca02c", marker="o", linestyle="", markersize=6, label="start"),
        Line2D([], [], color="#d62728", marker="*", linestyle="", markersize=9, label="goal"),
        Line2D([], [], color="black", linestyle="-", label="success"),
        Line2D([], [], color="black", linestyle="--", label="failure"),
    ]
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0)
    )
    return _save(fig, out_stem)


def _track_label(t: Track) -> str:
    return t.code if t.vlm is None else f"{t.code}-{t.vlm}"


def build_all_figures(
    rows: Sequence[RunRow],
    out_dir: Path,
    tracks: Optional[Sequence[Track]] = None,
    grid: Optional[Any] = None,
) -> Dict[str, Path]:
    out = {
        "sr_spl": sr_spl_bars(rows, out_dir / "sr_spl"),
        "sr_by_distance": sr_by_distance(rows, out_dir / "sr_by_distance"),
    }
    if any(r.vlm is not None for r in rows):
        out["cost_vs_sr"] = cost_vs_sr(rows, out_dir / "cost_vs_sr")
    if tracks and grid is not None:
        out["trajectories"] = trajectory_facets(tracks, grid, out_dir / "trajectories")
    return out
