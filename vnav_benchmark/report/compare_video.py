"""Comparison-video recipe: synchronize ground-truth panels for one endpoint pair.

The 1920 by 1080 canvas has a 4 by 3 method grid for 12 configurations, ordered B0, M1, M2-Astra
through M6-Qwen, plus a right column for the goal photo, legend and clock. Panels share a square
crop around all trajectories, endpoints and shortest paths, with a 1.5 m margin. They show
walls, free space, dashed shortest paths, colored trajectories, heading triangles, starts, goal
stars and 0.5 m goal circles. Finished episodes freeze and display their outcome and duration.
tools/viz/video.py handles encoding; see tools/viz/README.md section 7.

Use --methods, --cols, --rows, --canvas-w and --canvas-h to customize the grid. Spare cells show
the goal and legend. --compact-labels omits full method names.

Generate seven stills with:
python -m vnav_benchmark.report.compare_video <campaign> --token ep_xxxxxxxxxx --stills
Render full videos on an idle desktop with --repeat 0 --out <output-stem>.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vnav_benchmark.fairness import method_label

# Reuse tools/viz imports and the shared method-color rules.
from vnav_benchmark.report import episode_video as ev
from vnav_benchmark.report.colors import code_color_bgr
from vnav_benchmark.report.episode_data import EpisodeData, load_episode
from vnav_benchmark.report.results_io import REPORT_METHOD_IDS

draw, style, video = ev.draw, ev.style, ev.video

# Section 1: playback speed, canvas and layout.
SPEED = 4.0  # 4x playback, matching the episode-video convention.
FPS = 20
HOLD_S = 3.0  # Final freeze duration.
W, H = 1920, 1080
TITLE_H = 60
COLS, ROWS = 4, 3  # Twelve configurations.
CELL_W, CELL_H = 380, 340
GRID_X0, GRID_Y0 = 0, TITLE_H
SIDE = draw.Rect(COLS * CELL_W + 8, TITLE_H + 8, W - COLS * CELL_W - 16, H - TITLE_H - 16)
CELL_TITLE_H = 26
CELL_PAD = 6
CROP_MARGIN_M = 1.5  # Extra margin around trajectories and endpoints.
SUCCESS_RADIUS_M = 0.5

BG_BGR = (255, 255, 255)
FREE_BGR = (255, 255, 255)
BUFFER_BGR = (215, 215, 215)
WALL_BGR = (95, 95, 95)
DASH_BGR = (140, 140, 140)
TEXT_BGR = (10, 10, 10)
SUB_TEXT_BGR = (80, 80, 80)
EMPTY_BGR = (235, 235, 235)
OK_BGR = (40, 150, 40)
FAIL_BGR = (40, 40, 190)
RUNNING_BGR = (120, 120, 120)
# Outcome corner labels; done without success means arrival was declared at the wrong place.
FAIL_LABELS = {
    "done": "WRONG STOP",
    "timeout": "TIMEOUT",
    "fallen": "FELL",
    "method_error": "ERROR",
    "act_timeout": "SLOW DECISION",
    "cost_guard": "COST CAP",
}
TRAIL_PX = 3
POSE_TRI_PX = 10
START_PX = 6
GOAL_STAR_PX = 9
COMPACT_TITLE_H = 32  # Compact 720p titles leave enough space for 20 px text.
COMPACT_LABEL_PX = 20
COMPACT_CHIP_PX = 19


# Section 2: data.
def find_runs(campaign: Path, token: str, repeat: int) -> Dict[str, Path]:
    """Index completed campaign/runs/<method>/<token>/r<k> episodes with run.json by method ID."""
    out: Dict[str, Path] = {}
    for run_json in sorted((Path(campaign) / "runs").glob(f"*/{token}/r{repeat}/run.json")):
        out[run_json.parent.parent.parent.name] = run_json.parent
    return out


def cell_order(
    method_ids: Sequence[str],
    grid_method_ids: Sequence[str] = REPORT_METHOD_IDS,
    cols: int = COLS,
    rows: int = ROWS,
) -> List[Optional[str]]:
    """Arrange requested methods; None marks missing runs and spare cells hold the goal image."""
    present = set(method_ids)
    return [m if m in present else None for m in grid_method_ids][: cols * rows]


def crop_box(episodes: Sequence[EpisodeData]) -> Tuple[float, float, float, float]:
    """Return the square bounds of all traces, endpoints and shortest paths plus CROP_MARGIN_M."""
    xs: List[float] = []
    ys: List[float] = []
    for ep in episodes:
        xs += [ep.start[0], ep.goal[0]] + [p[0] for p in ep.waypoints]
        ys += [ep.start[1], ep.goal[1]] + [p[1] for p in ep.waypoints]
        if ep.trace.shape[0]:
            xs += ep.trace[:, 1].tolist()
            ys += ep.trace[:, 2].tolist()
    x0, x1 = min(xs) - CROP_MARGIN_M, max(xs) + CROP_MARGIN_M
    y0, y1 = min(ys) - CROP_MARGIN_M, max(ys) + CROP_MARGIN_M
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


class MapView:
    """Convert world meters to panel pixels with upward y; cache the cropped occupancy background.
    """

    def __init__(self, grid: object, box: Tuple[float, float, float, float], size: int) -> None:
        self.box = box
        self.size = size
        self.scale = size / (box[2] - box[0])
        img = np.empty((size, size, 3), np.uint8)
        img[:] = WALL_BGR
        if grid is not None:
            us, vs = np.meshgrid(np.arange(size), np.arange(size))
            wx = box[0] + (us + 0.5) / self.scale
            wy = box[3] - (vs + 0.5) / self.scale
            i = np.floor((wx - grid.x0) / grid.cell_m).astype(int)  # type: ignore[attr-defined]
            j = np.floor((wy - grid.y0) / grid.cell_m).astype(int)  # type: ignore[attr-defined]
            nx, ny = grid.free.shape  # type: ignore[attr-defined]
            inside = (i >= 0) & (j >= 0) & (i < nx) & (j < ny)
            ic, jc = np.clip(i, 0, nx - 1), np.clip(j, 0, ny - 1)
            free = inside & grid.free[ic, jc]  # type: ignore[attr-defined]
            raw = ~inside | grid.raw_blocked[ic, jc]  # type: ignore[attr-defined]
            img[free] = FREE_BGR
            img[~free & ~raw] = BUFFER_BGR
        self.base = img

    def px(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(round((x - self.box[0]) * self.scale)),
            int(round((self.box[3] - y) * self.scale)),
        )


# Section 3: one method panel.
def _dashed(img: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> None:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(length // 8))
    for k in range(0, n, 2):
        p = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
        q = (int(a[0] + (b[0] - a[0]) * (k + 1) / n), int(a[1] + (b[1] - a[1]) * (k + 1) / n))
        cv2.line(img, p, q, DASH_BGR, 1, cv2.LINE_AA)


def end_time(ep: EpisodeData) -> float:
    return float(ep.trace[-1, 0]) if ep.trace.shape[0] else 0.0


def outcome_chip(ep: EpisodeData) -> Tuple[str, Tuple[int, int, int]]:
    out = ep.outcome or {}
    t = float(out.get("sim_time_s", end_time(ep)))
    if out.get("success"):
        return f"SUCCESS {t:.0f} s", OK_BGR
    term = str(out.get("termination", "unknown"))
    return f"{FAIL_LABELS.get(term, term.upper())} {t:.0f} s", FAIL_BGR


def render_cell(
    view: MapView,
    ep: Optional[EpisodeData],
    method_id: Optional[str],
    t: float,
    cell_w: int = CELL_W,
    cell_h: int = CELL_H,
    compact_labels: bool = False,
) -> np.ndarray:
    """Draw a title above a square map; default dimensions and typography preserve the prior
    layout.
    """
    cell = np.full((cell_h, cell_w, 3), BG_BGR, np.uint8)
    scale = min(cell_w / CELL_W, cell_h / CELL_H)
    pad = max(1, round(CELL_PAD * scale))
    title_h = COMPACT_TITLE_H if compact_labels else round(CELL_TITLE_H * scale)
    label = method_id or ""
    code = ""
    if method_id is not None:
        code = method_label(method_id)
    label_px = COMPACT_LABEL_PX if compact_labels else max(1, round(18 * scale))
    draw.blit_text(cell, pad, round(3 * scale), code, label_px, True, TEXT_BGR)
    if ep is not None and not compact_labels:
        draw.blit_text(cell, pad + round(112 * scale), round(6 * scale), label,
                       max(1, round(13 * scale)), False, SUB_TEXT_BGR)  # fmt: skip
    size = view.size
    x0 = (cell_w - size) // 2
    y0 = title_h
    if ep is None:
        cell[y0 : y0 + size, x0 : x0 + size] = EMPTY_BGR
        draw.blit_text(cell, cell_w // 2, y0 + size // 2 - 10, "not run", label_px, False,
                       SUB_TEXT_BGR, align="center")  # fmt: skip
        return cell
    img = view.base.copy()
    color = code_color_bgr(ep.code)
    if len(ep.waypoints) >= 2:
        pts = [view.px(x, y) for x, y in ep.waypoints]
        for a, b in zip(pts, pts[1:]):
            _dashed(img, a, b)
    t_here = min(t, end_time(ep))
    if ep.trace.shape[0]:
        mask = ep.trace[:, 0] <= t_here + 1e-9
        if mask.sum() >= 2:
            pts = np.array([view.px(x, y) for x, y in ep.trace[mask, 1:3].tolist()], np.int32)
            cv2.polylines(img, [pts], False, color, TRAIL_PX, cv2.LINE_AA)
    sx, sy, _ = ep.start
    cv2.circle(img, view.px(sx, sy), START_PX, style.color_bgr255(0), -1, cv2.LINE_AA)
    cv2.circle(img, view.px(sx, sy), START_PX, (0, 0, 0), 1, cv2.LINE_AA)
    gx, gy, _ = ep.goal
    star = draw.star_points(np.array(view.px(gx, gy), np.float64), GOAL_STAR_PX).astype(np.int32)
    cv2.fillPoly(img, [star], color, cv2.LINE_AA)
    cv2.polylines(img, [star], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(img, view.px(gx, gy), max(2, int(SUCCESS_RADIUS_M * view.scale)), color, 1,
               cv2.LINE_AA)  # fmt: skip
    x, y, yaw = ep.pose_at(t_here)
    tri = ev._pose_triangle(view.px(x, y), yaw, POSE_TRI_PX)
    cv2.fillPoly(img, [tri], color, cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
    cell[y0 : y0 + size, x0 : x0 + size] = img
    finished = t >= end_time(ep) - 1e-9
    if finished:
        text, chip_bgr = outcome_chip(ep)
        if compact_labels:
            # Reserve space for drawn success/failure marks; P052 lacks these glyphs.
            text = "   " + text
    else:
        # A gray label indicates an episode that is still running.
        text, chip_bgr = "RUNNING", RUNNING_BGR
    chip_px = COMPACT_CHIP_PX if compact_labels else max(1, round(15 * scale))
    color_f, alpha = draw.label_chip(text, chip_px, True, (255, 255, 255), chip_bgr[::-1])
    if compact_labels and finished:
        # label_chip caches shared sprites, so do not modify them in place.
        color_f = color_f.copy()
        cx, cy = 16, color_f.shape[0] // 2
        if ep.outcome.get("success"):
            mark = np.array([(cx - 5, cy), (cx - 1, cy + 4), (cx + 7, cy - 5)], np.int32)
            cv2.polylines(color_f, [mark], False, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.line(color_f, (cx - 5, cy - 5), (cx + 5, cy + 5), (255, 255, 255), 2)
            cv2.line(color_f, (cx - 5, cy + 5), (cx + 5, cy - 5), (255, 255, 255), 2)
    draw.blend(cell, x0 + pad, y0 + pad, color_f, alpha)
    return cell


# Section 4: goal photograph, legend and clock in the right column.
def render_side(goal_jpeg: bytes, info: Dict[str, str], t: float) -> np.ndarray:
    side = np.full((SIDE.h, SIDE.w, 3), BG_BGR, np.uint8)
    y = 0
    draw.blit_text(side, 0, y, "Goal image (all methods)", 20, True, TEXT_BGR)
    y += 30
    goal = cv2.imdecode(np.frombuffer(goal_jpeg, np.uint8), cv2.IMREAD_COLOR) if goal_jpeg else None
    gh = int(SIDE.w * 3 / 4)
    if goal is not None:
        side[y : y + gh, :] = cv2.resize(goal, (SIDE.w, gh), interpolation=cv2.INTER_AREA)
    else:
        side[y : y + gh, :] = EMPTY_BGR
    y += gh + 20
    draw.blit_text(side, 0, y, f"Sim time  {draw.format_mmss(t)}", 26, True, TEXT_BGR)
    y += 44
    for line in (
        f"Episode {info.get('token', '')}",
        info.get("pair", ""),
        f"Repeat {info.get('repeat', '0')}",
    ):
        draw.blit_text(side, 0, y, line, 17, False, SUB_TEXT_BGR)
        y += 26
    y += 14
    draw.blit_text(side, 0, y, "Legend", 20, True, TEXT_BGR)
    y += 34
    cv2.circle(side, (10, y + 9), START_PX, style.color_bgr255(0), -1, cv2.LINE_AA)
    draw.blit_text(side, 28, y, "start", 16, False, TEXT_BGR)
    y += 28
    star = draw.star_points(np.array((10.0, y + 9.0)), GOAL_STAR_PX).astype(np.int32)
    cv2.fillPoly(side, [star], (120, 120, 120), cv2.LINE_AA)
    draw.blit_text(side, 28, y, "goal + 0.5 m success circle", 16, False, TEXT_BGR)
    y += 28
    _dashed(side, (2, y + 9), (22, y + 9))
    draw.blit_text(side, 28, y, "shortest path (ground truth)", 16, False, TEXT_BGR)
    y += 28
    cv2.line(side, (2, y + 9), (22, y + 9), (60, 60, 60), TRAIL_PX, cv2.LINE_AA)
    draw.blit_text(side, 28, y, "driven path (method color)", 16, False, TEXT_BGR)
    y += 28
    side[y : y + 16, 2:22] = BUFFER_BGR
    draw.blit_text(side, 28, y, "too close to obstacles for the robot", 16, False, TEXT_BGR)
    y += 34
    for k, code in enumerate(("B0", "M1", "M2", "M3", "M4", "M5", "M6")):
        cx = 10 + (k % 4) * 100
        cy = y + (k // 4) * 28
        cv2.rectangle(side, (cx - 8, cy + 2), (cx + 8, cy + 16), code_color_bgr(code), -1)
        draw.blit_text(side, cx + 14, cy, code, 16, True, TEXT_BGR)
    return side


def render_goal_cell(
    goal_jpeg: bytes, info: Dict[str, str], t: float, cell_w: int, cell_h: int
) -> np.ndarray:
    """Fill spare cells with the route token, known goal room, photograph and compact legend."""
    # Scale content from the blog's 320 by 360 reference panel to other grid sizes.
    base_w, base_h = 320, 360
    cell = np.full((base_h, base_w, 3), BG_BGR, np.uint8)
    draw.blit_text(cell, 6, 4, info.get("token", ""), 18, True, TEXT_BGR)
    room = info.get("goal_room", "")
    draw.blit_text(cell, 6, 29, f"Goal: {room}" if room else "Goal image", 17, True, TEXT_BGR)
    photo_y, photo_h = 54, 126  # Fill the upper half below the title with the photograph.
    goal = cv2.imdecode(np.frombuffer(goal_jpeg, np.uint8), cv2.IMREAD_COLOR) if goal_jpeg else None
    if goal is None:
        cell[photo_y : photo_y + photo_h] = EMPTY_BGR
    else:
        # Preserve aspect ratio and center-crop to fill the panel.
        gh, gw = goal.shape[:2]
        scale = max(base_w / gw, photo_h / gh)
        goal = cv2.resize(goal, (math.ceil(gw * scale), math.ceil(gh * scale)),
                          interpolation=cv2.INTER_AREA)
        x, y = (goal.shape[1] - base_w) // 2, (goal.shape[0] - photo_h) // 2
        cell[photo_y : photo_y + photo_h] = goal[y : y + photo_h, x : x + base_w]
    legend_y, line_h = 194, 28  # Four legend rows and the clock occupy the lower half.
    for i, text in enumerate(("Start", "Goal + 0.5 m success radius",
                              "Shortest path (ground truth)", "Trajectory (method color)")):
        y = legend_y + i * line_h
        cy = y + 9
        if i == 0:
            cv2.circle(cell, (16, cy), START_PX, style.color_bgr255(0), -1, cv2.LINE_AA)
        elif i == 1:
            star = draw.star_points(np.array((16.0, float(cy))), GOAL_STAR_PX).astype(np.int32)
            cv2.fillPoly(cell, [star], (120, 120, 120), cv2.LINE_AA)
            cv2.circle(cell, (16, cy), 12, (120, 120, 120), 1, cv2.LINE_AA)
        elif i == 2:
            _dashed(cell, (4, cy), (28, cy))
        else:
            cv2.line(cell, (4, cy), (28, cy), (60, 60, 60), TRAIL_PX, cv2.LINE_AA)
        draw.blit_text(cell, 36, y, text, 16, False, TEXT_BGR)
    draw.blit_text(cell, 6, 318, f"Sim time  {draw.format_mmss(t)}", 20, True, TEXT_BGR)
    return cv2.resize(cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)


# === §5 prepare() / frame(k) / main() ===
def prepare_for(
    token: str,
    repeat: int,
    grid_method_ids: Sequence[str] = REPORT_METHOD_IDS,
    cols: int = COLS,
    rows: int = ROWS,
    w: int = W,
    h: int = H,
    compact_labels: bool = False,
) -> Callable[[Path, Path, Callable[[str], None]], object]:
    grid_method_ids = tuple(grid_method_ids)
    if min(cols, rows, w, h) <= 0:
        raise video.VideoError("网格行列和画布尺寸必须是正整数")
    if not grid_method_ids or len(grid_method_ids) > cols * rows:
        raise video.VideoError("方法列表不能为空,且方法数不能超过格子数")
    if len(set(grid_method_ids)) != len(grid_method_ids):
        raise video.VideoError("方法列表不能有重复项")
    unknown = set(grid_method_ids) - set(REPORT_METHOD_IDS)
    if unknown:
        raise video.VideoError(f"未知正式配置: {sorted(unknown)}")

    has_goal_cells = len(grid_method_ids) < cols * rows
    title_h = 0 if has_goal_cells else round(TITLE_H * h / H)
    grid_w = w if has_goal_cells else round(COLS * CELL_W * w / W)
    rects = []
    for idx in range(cols * rows):
        r, c = divmod(idx, cols)
        x0, x1 = c * grid_w // cols, (c + 1) * grid_w // cols
        y0, y1 = title_h + r * (h - title_h) // rows, title_h + (r + 1) * (h - title_h) // rows
        rects.append(draw.Rect(x0, y0, x1 - x0, y1 - y0))
    cell_w, cell_h = min(rect.w for rect in rects), min(rect.h for rect in rects)
    scale = min(cell_w / CELL_W, cell_h / CELL_H)
    cell_title_h = COMPACT_TITLE_H if compact_labels else round(CELL_TITLE_H * scale)
    map_size = min(cell_w, cell_h - cell_title_h) - 2 * max(1, round(CELL_PAD * scale))
    if map_size <= 0:
        raise video.VideoError("画布太小,无法放下地图和标题")
    side = draw.Rect(round(SIDE.x * w / W), round(SIDE.y * h / H),
                     round(SIDE.w * w / W), round(SIDE.h * h / H))

    def prepare(source: Path, frames_dir: Path, log: Callable[[str], None]) -> video.VideoJob:
        del frames_dir
        runs = find_runs(Path(source), token, repeat)
        if not runs:
            raise video.VideoError(f"{source} 里没有 {token} r{repeat} 的回合")
        grid_cache: Dict[str, object] = {}

        def grid_fn(scene: str) -> object:
            if scene not in grid_cache:
                from vnav_benchmark.oracle import build_truth_grid

                grid_cache[scene] = build_truth_grid(scene)
            return grid_cache[scene]

        episodes: Dict[str, EpisodeData] = {}
        for mid, run_dir in runs.items():
            if mid in grid_method_ids:
                episodes[mid] = load_episode(run_dir, grid_fn)
            else:
                log(f"跳过未选配置 {mid}")
        if not episodes:
            raise video.VideoError("没有可画的正式配置回合")
        log(f"{token} r{repeat}: {len(episodes)} 个配置 {sorted(episodes)}")
        first = next(iter(episodes.values()))
        order = cell_order(list(episodes), grid_method_ids, cols, rows)
        box = crop_box(list(episodes.values()))
        view = MapView(first.grid, box, map_size)
        t_end = max(end_time(ep) for ep in episodes.values())
        tl = video.Timeline(0.0, t_end, speed=SPEED, fps=FPS, hold_s=HOLD_S)
        ep_info = first.run.get("episode", {})
        info = {
            "token": token,
            "repeat": str(repeat),
            "goal_room": str(ep_info.get("goal_room") or ""),
            "pair": "{}, shortest path {:.1f} m".format(
                ep_info.get("tier", ""), float(ep_info.get("shortest_m", 0.0))
            ),
        }

        def frame(k: int) -> np.ndarray:
            t, _final = tl.time_at(k)
            canvas = np.full((h, w, 3), BG_BGR, np.uint8)
            if not has_goal_cells:
                title = f"Same start and goal, all methods (ground-truth top view, {SPEED:.0f}x)"
                draw.blit_text(canvas, round(16 * w / W), round(14 * h / H), title,
                               max(1, round(26 * min(w / W, h / H))), True, TEXT_BGR)
            for idx, rect in enumerate(rects):
                if idx >= len(order):
                    cell = render_goal_cell(first.goal_jpeg, info, t, rect.w, rect.h)
                else:
                    mid = order[idx]
                    cell = render_cell(view, episodes.get(mid) if mid else None, mid, t,
                                       rect.w, rect.h, compact_labels)
                canvas[rect.y : rect.y1, rect.x : rect.x1] = cell
            if not has_goal_cells:
                panel = render_side(first.goal_jpeg, info, t)
                if panel.shape[:2] != (side.h, side.w):
                    panel = cv2.resize(panel, (side.w, side.h), interpolation=cv2.INTER_AREA)
                canvas[side.y : side.y1, side.x : side.x1] = panel
            return canvas

        tr = first.trace
        odom = ev.PoseTrack.from_samples(tr[:, 0].tolist(), tr[:, 1:3].tolist(), tr[:, 3].tolist())
        sync = video.SyncInputs(turn_t=None, t0=0.0, odom=odom, ego=None, third=None)
        meta = {"token": token, "repeat": repeat, "methods": sorted(episodes)}
        return video.VideoJob(timeline=tl, frame=frame, size=(w, h), info=meta, sync_inputs=sync)

    return prepare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="同一对起终点、所有方法的俯视对比视频")
    parser.add_argument("source", type=Path, help="跑批目录(里面有 runs/)")
    parser.add_argument("--token", required=True, help="起终点 token,如 ep_xxxxxxxxxx")
    parser.add_argument("--repeat", type=int, default=0, help="第几遍")
    parser.add_argument("--stills", action="store_true", help="只出 7 张静帧,不编码")
    parser.add_argument("--out", type=Path, default=None, help="产物路径(不含后缀)")
    parser.add_argument("--methods", default=None, help="逗号分隔的方法列表,默认全部正式配置")
    parser.add_argument("--cols", type=int, default=COLS, help="网格列数")
    parser.add_argument("--rows", type=int, default=ROWS, help="网格行数")
    parser.add_argument("--canvas-w", type=int, default=W, help="画布宽度(px)")
    parser.add_argument("--canvas-h", type=int, default=H, help="画布高度(px)")
    parser.add_argument("--compact-labels", action="store_true", help="只显示方法短名和结局")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return video.run_cli(
        build_parser(),
        lambda a: prepare_for(
            a.token, a.repeat,
            grid_method_ids=(REPORT_METHOD_IDS if a.methods is None
                             else [m.strip() for m in a.methods.split(",")]),
            cols=a.cols, rows=a.rows, w=a.canvas_w, h=a.canvas_h,
            compact_labels=a.compact_labels,
        ),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
