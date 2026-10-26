"""Episode-video recipe with a 1920 by 1080 canvas and 8 px panel gaps.

The left side contains a 1136 by 852 first-person view and five thumbnails: depth plus either
the VLM history frames at d-8, d-4, d-2 and d-1 or upcoming point-goal route keyframes. The
upper right shows goal and current subgoal images. The middle right shows ground-truth
trajectories, route keyframes, subgoal and endpoint markers beside decision, command, phase,
usage and legend information. The lower right shows VLM observation/memory or point-goal planner
text and an evaluator distance-to-goal curve. A final outcome banner overlays the frozen ending.
Velocity appears in the status panel; there is no third-person view or action arrow.

report/video.py handles bounded rendering and encoding. Example:
python -m vnav_benchmark.report.episode_video <episode-directory> --stills
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vnav_benchmark.fairness import parse_method_id
from vnav_benchmark.report import drawing as draw
from vnav_benchmark.report import style, video
from vnav_benchmark.report.colors import code_color_bgr
from vnav_benchmark.report.episode_data import DecisionInfo, EpisodeData, load_episode
from vnav_benchmark.report.tracks import PoseTrack

# Section 1: playback speed, canvas and layout.
SPEED = 2.5  # 2.5x playback, selected on 2026-10-22.
# At 25 fps and 2.5x playback, each frame advances one 0.1 s camera period without repeats.
FPS = 25
HOLD_S = 3.0  # Final freeze duration, seconds.

W, H = 1920, 1080
GAP = 8  # Gap between panels.
TITLE = draw.Rect(0, 0, W, 40)
# 4:3 first-person view occupies approximately 60% of the canvas width.
EGO = draw.Rect(GAP, 44, 1136, 852)
STRIP_Y = EGO.y1 + GAP  # Thumbnail row below the first-person view.
RIGHT_X = EGO.x1 + GAP + 8  # Right-column start; the lower-left strip extends 16 px beyond EGO.
THUMB = (222, 166)  # Five approximately 4:3 thumbnails plus gaps fill the left strip.
RIGHT_W = W - GAP - RIGHT_X
GOAL = draw.Rect(RIGHT_X, 44, (RIGHT_W - GAP) // 2, 279)
SUBGOAL = draw.Rect(GOAL.x1 + GAP, 44, RIGHT_W - GOAL.w - GAP, 279)
TOP = draw.Rect(RIGHT_X, GOAL.y1 + GAP, 468, 468)  # Ground-truth top-down panel; env0 is square.
STATUS = draw.Rect(TOP.x1 + GAP, TOP.y, RIGHT_X + RIGHT_W - TOP.x1 - GAP, TOP.h)
TEXT = draw.Rect(RIGHT_X, TOP.y1 + GAP, RIGHT_W, H - GAP - TOP.y1 - GAP)
CURVE_MIN_H = 60  # Minimum distance-curve height; text consumes the remaining panel space.

BG_BGR = (232, 232, 232)  # Canvas color visible through gaps between panels.
PANEL_BGR = (255, 255, 255)
TITLE_TEXT_BGR = (10, 10, 10)
SUB_TEXT_BGR = (60, 60, 60)
EVAL_TEXT_BGR = (130, 130, 130)  # Gray marks evaluator-only information unavailable to methods.
PLACEHOLDER_BG = (225, 225, 225)
PLACEHOLDER_FG = (90, 90, 90)
BODY_TEXT_SIZE = 18
LINE_HEIGHT = 25
SMALL_TEXT_SIZE = 15
SMALL_LINE_HEIGHT = 21
CHIP_SIZE = 15

DEPTH_MAX_M = 8.0
DEPTH_COLORMAP = cv2.COLORMAP_TURBO  # Pseudocolor palette for the depth thumbnail.

# Dark gray: walls, tall furniture and exterior cells (raw_blocked).
TOPDOWN_WALL_BGR = (90, 90, 90)
TOPDOWN_FREE_BGR = (255, 255, 255)  # Free space after inflation by oracle.ROBOT_RADIUS_M.
TOPDOWN_BUFFER_BGR = (205, 205, 205)  # Narrow blocked band introduced only by obstacle inflation.
WAYPOINT_DASH_BGR = (150, 150, 150)  # Dashed ground-truth shortest path.
WAYPOINT_DASH_LEN_PX = 8
WAYPOINT_GAP_PX = 6
START_RADIUS_PX = 7
GOAL_STAR_RADIUS_PX = 11
SUCCESS_RADIUS_M = 0.5  # Success-radius circle shown in the top-down panel.
POSE_TRI_SIZE_PX = 13
KF_RADIUS_PX = 5  # Route keyframe points.
KF_PASSED_BGR = (150, 150, 150)  # Traversed keyframes are filled gray.
# Orange BGR diamond highlights the current subgoal independently of method colors.
SUBGOAL_MARK_BGR = (0, 140, 255)
SUBGOAL_MARK_PX = 11

BANNER_W, BANNER_H = 760, 170
BANNER_ALPHA = 0.97  # Opaque enough to hide text beneath the banner.
BANNER_SUCCESS_BGR = (40, 150, 40)
BANNER_FAILURE_BGR = (40, 40, 190)
BANNER_TEXT_BGR = (255, 255, 255)


# Section 2: small helpers.
def _id_label(ep: EpisodeData) -> str:
    """Return the short title ID, such as M2-astra or B0."""
    return ep.code if ep.vlm is None else f"{ep.code}-{ep.vlm}"


def _code_color_bgr(code: str) -> Tuple[int, int, int]:
    return code_color_bgr(code)  # Method colors are defined only in report/colors.py.


def _wrap_pi(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _wrap(text: str, width_chars: int, max_lines: int) -> List[str]:
    lines = textwrap.wrap(text.replace("\n", " "), width_chars) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _truncate(lines[-1] + " …", width_chars)
    return lines


def _decode(jpeg: Optional[bytes]) -> Optional[np.ndarray]:
    if not jpeg:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def _put(canvas: np.ndarray, rect: draw.Rect, img: Optional[np.ndarray], empty: str) -> None:
    """Resize an image to fill rect, or draw a gray placeholder when unavailable."""
    if img is None:
        draw.placeholder(canvas, rect, empty, 16, PLACEHOLDER_BG, PLACEHOLDER_FG)
        return
    canvas[rect.y : rect.y1, rect.x : rect.x1] = cv2.resize(
        img, (rect.w, rect.h), interpolation=cv2.INTER_AREA
    )


def _chip(canvas: np.ndarray, rect: draw.Rect, text: str, right: bool = False) -> None:
    """Draw a translucent corner chip, preserving panel space otherwise used by a title row."""
    color, alpha = draw.label_chip(text, CHIP_SIZE)
    ch, cw = alpha.shape
    x = rect.x1 - cw - 6 if right else rect.x + 6
    draw.blend(canvas, x, rect.y + 6, color, alpha)
    del ch


def _dashed_line(
    canvas: np.ndarray, a: Tuple[int, int], b: Tuple[int, int], color_bgr: Tuple[int, int, int]
) -> None:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length < 1.0:
        return
    step = WAYPOINT_DASH_LEN_PX + WAYPOINT_GAP_PX
    for k in range(int(length / step) + 1):
        t0 = k * step / length
        if t0 >= 1.0:
            break
        t1 = min(1.0, t0 + WAYPOINT_DASH_LEN_PX / length)
        p0 = (int(round(a[0] + (b[0] - a[0]) * t0)), int(round(a[1] + (b[1] - a[1]) * t0)))
        p1 = (int(round(a[0] + (b[0] - a[0]) * t1)), int(round(a[1] + (b[1] - a[1]) * t1)))
        cv2.line(canvas, p0, p1, color_bgr, 2, cv2.LINE_AA)


def _pose_triangle(center_px: Tuple[int, int], yaw: float, size_px: float) -> np.ndarray:
    """Return heading-triangle vertices, converting world yaw to downward-y screen coordinates."""

    def _pt(ang: float, r: float) -> Tuple[float, float]:
        return center_px[0] + r * math.cos(ang), center_px[1] - r * math.sin(ang)

    pts = [_pt(yaw, size_px), _pt(yaw + 2.5, size_px * 0.55), _pt(yaw - 2.5, size_px * 0.55)]
    return np.array(pts, dtype=np.int32)


def _diamond(center: Tuple[int, int], r: int) -> np.ndarray:
    cx, cy = center
    return np.array([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], dtype=np.int32)


def _lines(
    canvas: np.ndarray,
    x: int,
    y: int,
    lines: Sequence[Tuple[str, Tuple[int, int, int], bool]],
    size: int = BODY_TEXT_SIZE,
    step: int = LINE_HEIGHT,
) -> int:
    """Draw (text, color, bold) rows and return the next row's y coordinate."""
    for text, color, bold in lines:
        draw.blit_text(canvas, x, y, text, size, bold, color)
        y += step
    return y


# Section 3: static objects built once in prepare() and read by frame(k).
class TopDown:
    """Ground-truth background and world-to-pixel transform, aspect-fitted within TOP.

    Pad unused panel space with the wall color.
    """

    def __init__(self, grid: Any, rect: draw.Rect) -> None:
        self.grid = grid
        self.rect = rect
        self.img = np.full((rect.h, rect.w, 3), TOPDOWN_WALL_BGR, np.uint8)
        if grid is None:
            self.inner = rect
            return
        nx, ny = grid.free.shape
        scale = min(rect.w / nx, rect.h / ny)
        w, h = int(round(nx * scale)), int(round(ny * scale))
        self.inner = draw.Rect(rect.x + (rect.w - w) // 2, rect.y + (rect.h - h) // 2, w, h)
        free = np.asarray(grid.free, dtype=bool).T
        blocked = np.asarray(grid.raw_blocked, dtype=bool).T
        base = np.empty((ny, nx, 3), dtype=np.uint8)
        base[:] = TOPDOWN_BUFFER_BGR
        base[free] = TOPDOWN_FREE_BGR
        base[blocked] = TOPDOWN_WALL_BGR
        base = cv2.resize(base[::-1], (w, h), interpolation=cv2.INTER_NEAREST)
        ox, oy = self.inner.x - rect.x, self.inner.y - rect.y
        self.img[oy : oy + h, ox : ox + w] = base
        self.px_per_m = scale / float(grid.cell_m)

    def to_px(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to panel pixels relative to TOP; drawing clips at panel
        boundaries.
        """
        g, r = self.grid, self.inner
        nx, ny = g.free.shape
        u = (x - g.x0) / (nx * g.cell_m)
        v = (y - g.y0) / (ny * g.cell_m)
        ox, oy = r.x - self.rect.x, r.y - self.rect.y
        return int(round(ox + u * r.w)), int(round(oy + (1.0 - v) * r.h))


def _distance_series(ep: EpisodeData) -> Tuple[np.ndarray, np.ndarray]:
    """Return evaluator distance to the goal over time from the 10 Hz trace."""
    if ep.trace.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    gx, gy, _ = ep.goal
    d = np.hypot(ep.trace[:, 1] - gx, ep.trace[:, 2] - gy)
    return ep.trace[:, 0], d


# Section 4: panel drawing.
def _draw_title_bar(canvas: np.ndarray, ep: EpisodeData, t: float) -> None:
    canvas[TITLE.y : TITLE.y1, :] = PANEL_BGR
    left = f"Image-Goal Navigation Benchmark  |  {_id_label(ep)}  {ep.method_id}"
    draw.blit_text(canvas, 16, 9, left, 22, True, TITLE_TEXT_BGR)
    right = f"{SPEED:g}x  |  sim {draw.format_mmss(t)}"
    draw.blit_text(canvas, W - 16, 9, right, 20, True, SUB_TEXT_BGR, align="right")


def _draw_ego(canvas: np.ndarray, ep: EpisodeData, t: float) -> None:
    path = ep.frame_at(t)
    img = cv2.imread(str(path)) if path is not None else None
    _put(canvas, EGO, img, "No ego frame")
    _chip(canvas, EGO, "First-person RGB")


def _thumb_rect(k: int) -> draw.Rect:
    return draw.Rect(GAP + k * (THUMB[0] + GAP), STRIP_Y, THUMB[0], THUMB[1])


def _draw_depth(canvas: np.ndarray, ep: EpisodeData, t: float) -> None:
    rect = _thumb_rect(0)
    path = ep.depth_at(t)
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path is not None else None
    if depth_mm is None:
        _put(canvas, rect, None, "No depth frame")
    else:
        depth_mm = cv2.resize(depth_mm, (rect.w, rect.h), interpolation=cv2.INTER_NEAREST)
        norm = np.clip(depth_mm.astype(np.float32) / 1000.0 / DEPTH_MAX_M, 0.0, 1.0)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), DEPTH_COLORMAP)
        colored[depth_mm == 0] = (0, 0, 0)
        canvas[rect.y : rect.y1, rect.x : rect.x1] = colored
    _chip(canvas, rect, "Depth (SRU input)" if ep.planner == "ptgoal" else "Depth (not to VLM)")


def _route_ahead(ep: EpisodeData, decision: Optional[DecisionInfo]) -> List[int]:
    """Return route keyframes after the current subgoal for point-goal thumbnails."""
    if ep.route is None or decision is None or decision.subgoal_kf not in ep.route.kf_ids:
        return []
    idx = ep.route.kf_ids.index(decision.subgoal_kf)
    return list(ep.route.kf_ids[idx + 1 : idx + 5])


def _draw_strip(canvas: np.ndarray, ep: EpisodeData, decision: Optional[DecisionInfo]) -> None:
    """Select four VLM history frames or upcoming point-goal keyframes beside the depth thumbnail.
    """
    if ep.vlm is not None:
        jpegs = list(decision.history_jpegs) if decision is not None else []
        labels = ["d-8", "d-4", "d-2", "d-1"][4 - len(jpegs) :] if jpegs else []
        for k in range(4):
            rect = _thumb_rect(k + 1)
            j = k - (4 - len(jpegs))
            if j < 0:
                _put(canvas, rect, None, "no history yet")
                continue
            _put(canvas, rect, _decode(jpegs[j]), "")
            _chip(canvas, rect, f"VLM history {labels[j]}")
        return
    ahead = _route_ahead(ep, decision)
    shown = [(k, f"Route ahead +{i + 1} (kf {k})") for i, k in enumerate(ahead)]
    # Near the goal, fill any missing future thumbnails with recent route keyframes.
    if ep.route is not None and len(shown) < 4:
        cur = decision.subgoal_kf if decision is not None else None
        idx = ep.route.kf_ids.index(cur) if cur in ep.route.kf_ids else len(ep.route.kf_ids)
        # Keyframes preceding the current subgoal have been traversed.
        tail = list(ep.route.kf_ids[:idx])[-(4 - len(shown)) :]
        shown += [(k, f"Route passed (kf {k})") for k in tail]
    for k in range(4):
        rect = _thumb_rect(k + 1)
        if k >= len(shown) or ep.route is None:
            _put(canvas, rect, None, "no route keyframe")
            continue
        _put(canvas, rect, _decode(ep.route.jpeg(shown[k][0])), "")
        _chip(canvas, rect, shown[k][1])


def _passed_kfs(ep: EpisodeData, t: float, decision: Optional[DecisionInfo]) -> List[int]:
    """Return keyframes replaced as subgoals or preceding the current subgoal; include all at the
    end.
    """
    if ep.route is None:
        return []
    kfs = list(ep.route.kf_ids)
    visited = set(ep.visited_kfs(t))
    cur = decision.subgoal_kf if decision is not None else None
    cur_idx = kfs.index(cur) if cur in kfs else (len(kfs) if visited else 0)
    return [k for i, k in enumerate(kfs) if k in visited or i < cur_idx]


def _subgoal_chip_text(method_id: str) -> str:
    """Continuous-localization methods judge arrival by pose; initloc/noloc VLM methods use the
    model.
    """
    judge = parse_method_id(method_id).progress_judge
    return "judged by pose" if judge == "pose" else "judged by VLM"


def _draw_goal_and_subgoal(
    canvas: np.ndarray, ep: EpisodeData, decision: Optional[DecisionInfo]
) -> None:
    _put(canvas, GOAL, _decode(ep.goal_jpeg), "No goal image")
    _chip(canvas, GOAL, "Goal image")
    img: Optional[np.ndarray] = None
    kf = decision.subgoal_kf if decision is not None else None
    if ep.vlm is not None and decision is not None:
        # The image actually shown to the model for this decision.
        img = _decode(decision.subgoal_jpeg)
    elif kf is not None and ep.route is not None:
        # Point-goal methods display the current subgoal keyframe's photograph.
        img = _decode(ep.route.jpeg(kf))
    empty = "No subgoal image (goal only)" if ep.route is None else "Final approach (goal only)"
    _put(canvas, SUBGOAL, img, empty)
    _chip(canvas, SUBGOAL, "Current subgoal" + ("" if kf is None else f" (kf {kf})"))
    _chip(canvas, SUBGOAL, _subgoal_chip_text(ep.method_id), right=True)


def _draw_topdown(
    canvas: np.ndarray,
    ep: EpisodeData,
    td: TopDown,
    code_color: Tuple[int, int, int],
    t: float,
    decision: Optional[DecisionInfo],
) -> None:
    r = TOP
    full = canvas
    canvas = full[r.y : r.y1, r.x : r.x1]  # Panel slices write directly into the shared canvas.
    canvas[:] = td.img
    if td.grid is None:
        _chip(full, TOP, "Top-down (ground truth, evaluator only)")
        return
    if len(ep.waypoints) >= 2:
        pts = [td.to_px(x, y) for x, y in ep.waypoints]
        for a, b in zip(pts, pts[1:]):
            _dashed_line(canvas, a, b, WAYPOINT_DASH_BGR)

    cur_kf = decision.subgoal_kf if decision is not None else None
    if ep.route is not None:
        kfs = list(ep.route.kf_ids)
        pts = [td.to_px(*ep.route.poses_world[k][:2]) for k in kfs]
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, style.lighten(code_color), 2,
                      cv2.LINE_AA)  # fmt: skip
        passed = set(_passed_kfs(ep, t, decision))
        for k, p in zip(kfs, pts):
            if k in passed:
                cv2.circle(canvas, p, KF_RADIUS_PX, KF_PASSED_BGR, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, p, KF_RADIUS_PX, PANEL_BGR, -1, cv2.LINE_AA)
                cv2.circle(canvas, p, KF_RADIUS_PX, code_color, 2, cv2.LINE_AA)

    if ep.trace.shape[0] > 0:
        mask = ep.trace[:, 0] <= t
        xy = ep.trace[mask, 1:3]
        if xy.shape[0] >= 2:
            pts = np.array([td.to_px(x, y) for x, y in xy.tolist()], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, code_color, 3, cv2.LINE_AA)

    sx, sy, _ = ep.start
    cv2.circle(canvas, td.to_px(sx, sy), START_RADIUS_PX, style.color_bgr255(0), -1, cv2.LINE_AA)
    cv2.circle(canvas, td.to_px(sx, sy), START_RADIUS_PX, (0, 0, 0), 1, cv2.LINE_AA)
    gx, gy, _ = ep.goal
    star = draw.star_points(np.array(td.to_px(gx, gy), dtype=np.float64), GOAL_STAR_RADIUS_PX)
    cv2.fillPoly(canvas, [star.astype(np.int32)], code_color, cv2.LINE_AA)
    cv2.polylines(canvas, [star.astype(np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(canvas, td.to_px(gx, gy), int(round(SUCCESS_RADIUS_M * td.px_per_m)), code_color,
               1, cv2.LINE_AA)  # fmt: skip

    x, y, yaw = ep.pose_at(t)
    robot = td.to_px(x, y)
    if cur_kf is not None and ep.route is not None and cur_kf in ep.route.poses_world:
        sub = td.to_px(*ep.route.poses_world[cur_kf][:2])
        cv2.line(canvas, robot, sub, SUBGOAL_MARK_BGR, 1, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_diamond(sub, SUBGOAL_MARK_PX)], SUBGOAL_MARK_BGR, cv2.LINE_AA)
        cv2.polylines(canvas, [_diamond(sub, SUBGOAL_MARK_PX)], True, (0, 0, 0), 2, cv2.LINE_AA)
    tri = _pose_triangle(robot, yaw, POSE_TRI_SIZE_PX)
    cv2.fillPoly(canvas, [tri], code_color, cv2.LINE_AA)
    cv2.polylines(canvas, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
    _chip(full, TOP, "Top-down (ground truth, evaluator only)")


def _draw_status(
    canvas: np.ndarray,
    ep: EpisodeData,
    decision: Optional[DecisionInfo],
    t: float,
    code_color: Tuple[int, int, int],
) -> None:
    r = STATUS
    canvas[r.y : r.y1, r.x : r.x1] = PANEL_BGR
    budget = float((ep.run.get("settings") or {}).get("budget_s") or 0.0)
    rows: List[Tuple[str, Tuple[int, int, int], bool]] = [("Status", TITLE_TEXT_BGR, True)]
    if decision is None:
        rows.append(("Decision -", SUB_TEXT_BGR, False))
    else:
        vx, vy, wz = decision.applied
        rows += [
            (f"Decision {decision.i}", SUB_TEXT_BGR, False),
            (f"Sim {t:.1f} s" + (f" / {budget:g} s" if budget else ""), SUB_TEXT_BGR, False),
            (f"vx {vx:+.2f} m/s", SUB_TEXT_BGR, False),
            (f"vy {vy:+.2f} m/s", SUB_TEXT_BGR, False),
            (f"wz {wz:+.2f} rad/s", SUB_TEXT_BGR, False),
            (f"hold {decision.hold_s:.1f} s", SUB_TEXT_BGR, False),
        ]
        if ep.route is not None and decision.subgoal_kf in ep.route.kf_ids:
            n = len(ep.route.kf_ids)
            rows.append((f"Subgoal kf {decision.subgoal_kf} ({n} on route)", SUB_TEXT_BGR, False))
        if decision.vlm is not None:
            lat = decision.vlm.get("latency") or {}
            spent = sum(d.usage_usd for d in ep.decisions if d.t_sim <= t)
            rows += [
                (f"Phase {decision.vlm.get('phase', '')}", SUB_TEXT_BGR, False),
                (f"API {float(lat.get('vlm_api_s', 0.0)):.1f} s", SUB_TEXT_BGR, False),
                (f"tok {decision.tokens_in}/{decision.tokens_out}", SUB_TEXT_BGR, False),
                (f"$ {spent:.3f} so far", SUB_TEXT_BGR, False),
            ]
        else:
            lat = (decision.debug or {}).get("latency") or {}
            ctrl = lat.get("controller_s")
            rows.append((f"Phase {(decision.debug or {}).get('phase', '')}", SUB_TEXT_BGR, False))
            if ctrl is not None:
                rows.append((f"SRU {1000 * float(ctrl):.0f} ms/step", SUB_TEXT_BGR, False))
    y = _lines(canvas, r.x + 12, r.y + 10, rows, SMALL_TEXT_SIZE + 1, SMALL_LINE_HEIGHT + 1)
    _draw_legend(canvas, r.x + 12, max(y + 8, r.y1 - 7 * SMALL_LINE_HEIGHT - 6), code_color)


def _draw_legend(canvas: np.ndarray, x: int, y: int, code_color: Tuple[int, int, int]) -> None:
    items = ["robot", "start", "goal", "subgoal", "kf ahead", "kf passed", "shortest"]
    for k, name in enumerate(items):
        cy = y + k * SMALL_LINE_HEIGHT + 9
        c = (x + 9, cy)
        if name == "robot":
            tri = _pose_triangle(c, 0.0, 9)
            cv2.fillPoly(canvas, [tri], code_color, cv2.LINE_AA)
        elif name == "start":
            cv2.circle(canvas, c, 6, style.color_bgr255(0), -1, cv2.LINE_AA)
        elif name == "goal":
            star = draw.star_points(np.array(c, dtype=np.float64), 8)
            cv2.fillPoly(canvas, [star.astype(np.int32)], code_color, cv2.LINE_AA)
        elif name == "subgoal":
            cv2.fillPoly(canvas, [_diamond(c, 7)], SUBGOAL_MARK_BGR, cv2.LINE_AA)
        elif name == "kf ahead":
            cv2.circle(canvas, c, KF_RADIUS_PX, code_color, 2, cv2.LINE_AA)
        elif name == "kf passed":
            cv2.circle(canvas, c, KF_RADIUS_PX, KF_PASSED_BGR, -1, cv2.LINE_AA)
        else:
            _dashed_line(canvas, (x, cy), (x + 18, cy), WAYPOINT_DASH_BGR)
        draw.blit_text(canvas, x + 26, cy - 9, name, SMALL_TEXT_SIZE, False, SUB_TEXT_BGR)


def _text_rows(
    ep: EpisodeData, decision: Optional[DecisionInfo], t: float
) -> List[Tuple[str, Tuple[int, int, int], bool]]:
    width = 78  # Approximate characters per line at 18 px text in a 744 px column.
    rows: List[Tuple[str, Tuple[int, int, int], bool]] = []
    if decision is not None and decision.vlm is not None:
        reply = decision.vlm.get("reply") or {}
        obs = _wrap(f'VLM observation: "{reply.get("observation", "")}"', width, 2)
        mem = _wrap(f'VLM memory: "{reply.get("memory", "")}"', width, 3)
        rows += [(s, TITLE_TEXT_BGR, False) for s in obs]
        rows += [(s, SUB_TEXT_BGR, False) for s in mem]
        flags = [k for k in ("target_visible", "subgoal_reached", "done") if reply.get(k)]
        rows.append(("VLM says: " + (", ".join(flags) if flags else "-"), SUB_TEXT_BGR, False))
    else:
        rows.append(("Point-goal planner SRU (depth + pose); no VLM in the loop", SUB_TEXT_BGR,
                     True))  # fmt: skip
        if ep.route is not None:
            passed = len(_passed_kfs(ep, t, decision))
            rows.append((f"Route: {len(ep.route.kf_ids)} keyframes, {passed} passed so far",
                         SUB_TEXT_BGR, False))  # fmt: skip
        if decision is not None and decision.note:
            rows.append((f"Planner phase: {decision.note}", SUB_TEXT_BGR, False))
    x, y, yaw = ep.pose_at(t)
    gx, gy, gyaw = ep.goal
    rows.append(
        (
            f"Evaluator (not visible to the method): distance {math.hypot(gx - x, gy - y):.2f} m"
            f"  yaw {math.degrees(_wrap_pi(gyaw - yaw)):.0f} deg",
            EVAL_TEXT_BGR,
            False,
        )
    )
    return rows


def _draw_text(
    canvas: np.ndarray,
    ep: EpisodeData,
    decision: Optional[DecisionInfo],
    t: float,
    series: Tuple[np.ndarray, np.ndarray],
    code_color: Tuple[int, int, int],
) -> None:
    r = TEXT
    canvas[r.y : r.y1, r.x : r.x1] = PANEL_BGR
    y = _lines(canvas, r.x + 12, r.y + 10, _text_rows(ep, decision, t))
    # Remaining space holds evaluator-only distance to the goal over time.
    ts, ds = series
    top = min(y + 8, r.y1 - CURVE_MIN_H)
    draw.blit_text(canvas, r.x + 12, top, "Distance to goal (evaluator)", SMALL_TEXT_SIZE, False,
                   EVAL_TEXT_BGR)  # fmt: skip
    plot = draw.Rect(
        r.x + 52, top + SMALL_LINE_HEIGHT, r.w - 64, r.y1 - top - SMALL_LINE_HEIGHT - 22
    )
    cv2.rectangle(canvas, (plot.x, plot.y), (plot.x1, plot.y1), (200, 200, 200), 1)
    if ts.size < 2:
        return
    t_max = max(float(ts[-1]), 1e-6)
    d_max = max(float(ds.max()), 1.0)
    draw.blit_text(canvas, plot.x - 6, plot.y, f"{d_max:.0f} m", SMALL_TEXT_SIZE - 2, False,
                   EVAL_TEXT_BGR, align="right")  # fmt: skip
    draw.blit_text(canvas, plot.x - 6, plot.y1 - 14, "0", SMALL_TEXT_SIZE - 2, False,
                   EVAL_TEXT_BGR, align="right")  # fmt: skip
    draw.blit_text(canvas, plot.x1, plot.y1 + 3, f"{t_max:.0f} s", SMALL_TEXT_SIZE - 2, False,
                   EVAL_TEXT_BGR, align="right")  # fmt: skip
    keep = ts <= t
    px = plot.x + (ts[keep] / t_max * plot.w)
    py = plot.y1 - (ds[keep] / d_max * plot.h)
    if px.size >= 2:
        pts = np.stack([px, py], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, code_color, 2, cv2.LINE_AA)
    ok_y = int(plot.y1 - SUCCESS_RADIUS_M / d_max * plot.h)
    cv2.line(canvas, (plot.x, ok_y), (plot.x1, ok_y), (120, 200, 120), 1, cv2.LINE_AA)


def _draw_banner(canvas: np.ndarray, ep: EpisodeData) -> None:
    success = bool(ep.outcome.get("success", False))
    termination = str(ep.outcome.get("termination", "unknown"))
    sim_time_s = float(ep.outcome.get("sim_time_s", 0.0))
    dist = ep.outcome.get("final_dist_m", 0.0)
    yaw_err = ep.outcome.get("final_yaw_err_deg", 0.0)
    spl = float(ep.outcome.get("spl", 0.0))
    x0, y0 = (W - BANNER_W) // 2, (H - BANNER_H) // 2
    color = np.empty((BANNER_H, BANNER_W, 3), dtype=np.float32)
    color[:] = BANNER_SUCCESS_BGR if success else BANNER_FAILURE_BGR
    draw.blend(canvas, x0, y0, color, np.full((BANNER_H, BANNER_W), BANNER_ALPHA, np.float32))
    headline = f"SUCCESS — reached in {sim_time_s:.0f} s" if success else f"FAILURE — {termination}"
    draw.blit_text(canvas, W // 2, y0 + 46, headline, 34, True, BANNER_TEXT_BGR, align="center")
    detail = (
        f"distance {dist:.2f} m, yaw {yaw_err:.0f} deg, SPL {spl:.2f}"
        if dist is not None and yaw_err is not None
        else f"different floor, distance / yaw N/A, SPL {spl:.2f}"
    )
    draw.blit_text(canvas, W // 2, y0 + 100, detail, 22, False, BANNER_TEXT_BGR, align="center")


# === §5 prepare() / frame(k) / main() ===
def prepare(source: Path, frames_dir: Path, log: Callable[[str], None]) -> video.VideoJob:
    # Read original episode images directly; this recipe writes no intermediate frames.
    del frames_dir
    ep = load_episode(Path(source))
    log(f"loaded episode {ep.method_id} / {ep.run_dir.name}: {len(ep.decisions)} decisions")
    return job_for(ep)


def job_for(ep: EpisodeData) -> video.VideoJob:
    """Build a rendering job from an EpisodeData object; tests may inject synthetic episodes."""
    t0 = 0.0
    t_end = ep.decisions[-1].t_sim + ep.decisions[-1].hold_s if ep.decisions else t0
    if ep.trace.shape[0] > 0:
        t_end = min(t_end if t_end > t0 else float(ep.trace[-1, 0]), float(ep.trace[-1, 0]))
    tl = video.Timeline(t0, t_end, speed=SPEED, fps=FPS, hold_s=HOLD_S)
    pose = (
        PoseTrack.from_samples(
            ep.trace[:, 0].tolist(), ep.trace[:, 1:3].tolist(), ep.trace[:, 3].tolist()
        )
        if ep.trace.shape[0] > 0
        else PoseTrack.from_samples([], [], [])
    )
    td = TopDown(ep.grid, TOP)
    code_color = _code_color_bgr(ep.code)
    series = _distance_series(ep)

    def frame(k: int) -> np.ndarray:
        t, final = tl.time_at(k)
        canvas = np.full((H, W, 3), BG_BGR, np.uint8)
        decision = ep.decision_at(t)
        _draw_title_bar(canvas, ep, t)
        _draw_ego(canvas, ep, t)
        _draw_depth(canvas, ep, t)
        _draw_strip(canvas, ep, decision)
        _draw_goal_and_subgoal(canvas, ep, decision)
        _draw_topdown(canvas, ep, td, code_color, t, decision)
        _draw_status(canvas, ep, decision, t, code_color)
        _draw_text(canvas, ep, decision, t, series, code_color)
        if final:
            _draw_banner(canvas, ep)
        return canvas

    info = {
        "method_id": ep.method_id,
        "code": ep.code,
        "episode_token": ep.run_dir.parent.name,
        "outcome": {k: ep.outcome.get(k) for k in ("termination", "success", "spl", "sim_time_s")},
    }
    sync = video.SyncInputs(turn_t=None, t0=t0, odom=pose, ego=None, third=None)
    return video.VideoJob(timeline=tl, frame=frame, size=(W, H), info=info, sync_inputs=sync)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单回合视频:第一人称 / 深度 + 历史帧或前方关键帧 / 目标与子目标 / 真值俯视"
    )
    parser.add_argument(
        "source", type=Path, help="回合目录(<campaign>/runs/<method_id>/<episode_token>/r<k>/)"
    )
    parser.add_argument("--stills", action="store_true", help="只出 7 张静帧,不编码")
    parser.add_argument("--out", type=Path, default=None, help="产物路径(不含后缀)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return video.run_cli(build_parser(), lambda _args: prepare, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
