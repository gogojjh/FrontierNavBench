"""Review a completed campaign with result tables and method-by-route truth trajectories.

Read campaign/runs/<method>/<episode>/r<repeat>/run.json, trace.csv and method/summary.json.
Resolve route labels through run.json episode.route and maps.yaml routes. This evaluator tool
reads ground truth and is not method code.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vnav_benchmark import mapview
from vnav_benchmark.oracle import TruthGrid
from vnav_benchmark.rundir import read_trace

STRICT = "0.5 m / 45°"  # Primary success criterion.
LOOSE = "2 m / 90°"  # Loose success criterion.
FRAME_PAD_M = 1.5  # Extra top-down margin around trajectories and endpoints, meters.


def collect(root: Path, repeat: Optional[int] = None) -> List[Dict[str, Any]]:
    config = mapview.load_maps_config()
    rows = []
    for run_json in sorted(Path(root).glob("runs/*/*/r*/run.json")):
        doc = json.loads(run_json.read_text())
        if repeat is not None and doc["repeat_index"] != repeat:
            continue
        ep, out, met = doc["episode"], doc.get("outcome") or {}, doc.get("metrics") or {}
        try:
            label = config.route_info(ep["route"]).label
        except mapview.MapViewError:
            label = ep["route"]
        summary_path = run_json.parent / "method" / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        rows.append({
            "dir": str(run_json.parent), "method": doc["method_id"], "route": label,
            "repeat": doc["repeat_index"], "status": doc["status"],
            "termination": out.get("termination"), "success": met.get("success"),
            "success_loose": met.get("success_loose"), "final_dist_m": out.get("final_dist_m"),
            "yaw_err_deg": out.get("final_yaw_err_deg"), "path_m": met.get("path_length_m"),
            "shortest_m": ep.get("shortest_m"), "plr": met.get("plr"), "spl": met.get("spl"),
            "sim_s": out.get("sim_time_s"), "falls": out.get("falls"),
            "stuck": out.get("stuck_events"), "replans": summary.get("replans"),
            "replans_on_switch": summary.get("replans_on_switch"),
            "degraded": summary.get("degraded"),
            "kept": (summary["stall_kept_no_alt"] + summary["stall_kept_after_cap"]
                     # Available only after stuck rule C was introduced on 2026-10-21.
                     if "stall_kept_no_alt" in summary else None),
            "sru_speed": summary.get("sru_speed"),
            "start": ep["start"], "goal": ep["goal"], "waypoints": ep.get("waypoints") or [],
        })  # fmt: skip
    return sorted(rows, key=lambda r: (r["method"], r["route"], r["repeat"]))


def _f(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        f"| 方法 | 路线 | 遍 | 结束 | 成功 {STRICT} | 成功 {LOOSE} | 终点误差 m / ° | 走了 m "
        "| 最短路 m | PLR | SPL | 仿真 s | 翻倒 | 卡住 | 卡住重规划 / 换子目标重规划 "
        "| 退化 / 卡住续追 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['route']} | {r['repeat']} | {_f(r['termination'])} | "
            f"{_f(r['success'])} | {_f(r['success_loose'])} | {_f(r['final_dist_m'])} / "
            f"{_f(r['yaw_err_deg'], 0)} | {_f(r['path_m'], 1)} | {_f(r['shortest_m'], 1)} | "
            f"{_f(r['plr'])} | {_f(r['spl'])} | {_f(r['sim_s'], 1)} | {_f(r['falls'])} | "
            f"{_f(r['stuck'])} | "
            f"{_f(r['replans'])} / {_f(r['replans_on_switch'])} | "
            f"{_f(r['degraded'])} / {_f(r['kept'])} |"
        )
    lines += ["", f"| 方法 | 回合 | 成功率 {STRICT} | 成功率 {LOOSE} | 成功回合 PLR 中位 "
              "| SPL 平均 | 用时中位 s |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for method in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == method and r["status"] != "unscored_infra"]
        plrs = [r["plr"] for r in rs if r["success"] and r["plr"] is not None]
        strict = sum(bool(r["success"]) for r in rs)
        loose = sum(bool(r["success_loose"]) for r in rs)
        med = f"{float(np.median(plrs)):.2f}" if plrs else "—"
        spls = [r["spl"] for r in rs if r["spl"] is not None]
        spl = f"{float(np.mean(spls)):.2f}" if spls else "—"
        secs = [r["sim_s"] for r in rs if r["sim_s"] is not None]
        sec = f"{float(np.median(secs)):.1f}" if secs else "—"
        lines.append(
            f"| {method} | {len(rs)} | {strict}/{len(rs)} | {loose}/{len(rs)} | {med} | {spl} "
            f"| {sec} |"
        )
    return "\n".join(lines) + "\n"


def sru_speed_markdown(rows: List[Dict[str, Any]]) -> str:
    """Summarize raw, unclipped SRU velocities from debug-truthpath-sru; return empty text if
    absent.
    """
    rs = [r for r in rows if r.get("sru_speed")]
    if not rs:
        return ""
    lines = [
        "| 方法 | 路线 | 遍 | 步数 | vx 5/50/95% | vy 5/50/95% | wz 5/50/95% | 倒车占比 "
        "| vx≈0 占比 | 转向贴上限占比 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rs:
        s = r["sru_speed"]
        lines.append(
            f"| {r['method']} | {r['route']} | {r['repeat']} | {s['n']} | {s['vx']} | {s['vy']} | "
            f"{s['wz']} | {s['vx_negative_frac']} | {s['vx_near_zero_frac']} | "
            f"{s['wz_saturated_frac']} |"
        )
    return "\n".join(lines) + "\n"


def frame_sheet(rows: List[Dict[str, Any]], out_jpg: Path, per_route: int = 6) -> bool:
    """Draw one row per episode with per_route evenly spaced method/frames/*.jpg images."""
    from vnav_benchmark.viz_topdown import contact_sheet

    images, labels = [], []
    for r in rows:
        frames = sorted((Path(r["dir"]) / "method" / "frames").glob("*.jpg"))
        if not frames:
            continue
        idx = np.linspace(0, len(frames) - 1, min(per_route, len(frames))).round().astype(int)
        for i in idx:
            images.append(frames[i].read_bytes())
            labels.append(f"{r['route']} t={frames[i].stem.lstrip('0') or '0'}s")
        # Repeat the last available frame to keep contact-sheet rows aligned.
        images += [images[-1]] * (per_route - len(idx))
        labels += [labels[-1]] * (per_route - len(idx))
    if not images:
        return False
    contact_sheet(images, labels, out_jpg, cols=per_route, width=320)
    return True


def draw(rows: List[Dict[str, Any]], grid: TruthGrid, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vnav_benchmark.report.style import (
        acquire_color_palette,
        setting_font,
    )

    setting_font(fontsize=8, titlesize=9, legend_fontsize=7)
    palette = acquire_color_palette()
    methods = sorted({r["method"] for r in rows})
    routes = sorted({r["route"] for r in rows})
    size = (3.4 * len(routes), 3.6 * len(methods))
    fig, axes = plt.subplots(len(methods), len(routes), figsize=size, squeeze=False)
    nx, ny = grid.free.shape
    extent = (grid.x0, grid.x0 + nx * grid.cell_m, grid.y0, grid.y0 + ny * grid.cell_m)
    traces = {}
    for r in rows:
        trace_path = Path(r["dir"]) / "trace.csv"  # Infrastructure failures have no trajectory.
        traces[r["dir"]] = read_trace(trace_path) if trace_path.is_file() else []
    # Share route bounds across a column, including all methods, endpoints and waypoints.
    bounds = {}
    for route in routes:
        pts = [(t["x"], t["y"]) for r in rows if r["route"] == route for t in traces[r["dir"]]]
        for r in (r for r in rows if r["route"] == route):
            pts += [(r["start"]["x"], r["start"]["y"]), (r["goal"]["x"], r["goal"]["y"])]
            pts += [tuple(w) for w in r["waypoints"]]
        xs, ys = zip(*pts) if pts else ((0.0, 20.0), (0.0, 20.0))
        bounds[route] = (min(xs) - FRAME_PAD_M, max(xs) + FRAME_PAD_M,
                         min(ys) - FRAME_PAD_M, max(ys) + FRAME_PAD_M)  # fmt: skip
    for i, method in enumerate(methods):
        for j, route in enumerate(routes):
            ax = axes[i][j]
            ax.imshow(np.where(grid.raw_blocked.T, 0.3, np.where(grid.free.T, 1.0, 0.8)),
                      origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")  # fmt: skip
            cell = [r for r in rows if r["method"] == method and r["route"] == route
                    # Unscored runs contribute neither drawings nor repeat counts.
                    and r["status"] != "unscored_infra"]
            for k, r in enumerate(cell):
                if r["waypoints"]:
                    wx, wy = zip(*r["waypoints"])
                    ax.plot(wx, wy, "--", color="0.45", lw=1.0)
                trace = traces[r["dir"]]
                color = palette[0] if r["success"] else palette[1]
                ax.plot([t["x"] for t in trace], [t["y"] for t in trace], "-", color=color,
                        lw=1.4, alpha=1.0 if k == 0 else 0.5)  # fmt: skip
                g = r["goal"]
                ax.plot(g["x"], g["y"], "*", color=palette[2], ms=11)
                tip = (g["x"] + 0.8 * math.cos(g["yaw"]), g["y"] + 0.8 * math.sin(g["yaw"]))
                ax.annotate("", xy=tip, xytext=(g["x"], g["y"]),
                            arrowprops={"arrowstyle": "->", "color": palette[2]})
                ax.plot(r["start"]["x"], r["start"]["y"], "o", color=palette[3], ms=6)
            if len(cell) == 1:
                r = cell[0]
                ax.set_title(f"{method}\n{route}: {r['termination']}, "
                             f"err {_f(r['final_dist_m'])} m, PLR {_f(r['plr'])}")  # fmt: skip
            # Multiple repeats: successful count and median PLR among successful episodes.
            elif cell:
                plrs = [r["plr"] for r in cell if r["success"] and r["plr"] is not None]
                med = f"{float(np.median(plrs)):.2f}" if plrs else "—"
                wins = sum(bool(r["success"]) for r in cell)
                ax.set_title(f"{method}\n{route}: {wins}/{len(cell)} success, PLR med {med}")
            ax.set_xticks([])
            ax.set_yticks([])
            x0, x1, y0, y1 = bounds[route]
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
