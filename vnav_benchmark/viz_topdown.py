"""Ground-truth top-down plots and goal contact sheets using shared tools/viz styling.

All figure labels are English.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from vnav_benchmark.oracle import TruthGrid

TIER_ORDER = ("near", "mid", "far", "other")
# Single-floor figure width; height follows the scene bounding-box aspect ratio.
FIGURE_WIDTH_IN = 9.0


def _style():  # type: ignore[no-untyped-def]
    from vnav_benchmark.report import style

    return style


def plot_episodes(
    grid: TruthGrid, records: Sequence, out_png: Path, title: str, label_rooms: bool = True
) -> None:
    """Plot the truth grid, walls, circular starts, oriented triangular goals and shortened paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = _style()
    style.setting_font(11, 12, 9)
    width, height = (n * grid.cell_m for n in grid.shape)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * height / width))
    extent = (
        grid.x0,
        grid.x0 + grid.shape[0] * grid.cell_m,
        grid.y0,
        grid.y0 + grid.shape[1] * grid.cell_m,
    )
    img = np.where(grid.raw_blocked.T, 0.35, np.where(grid.free.T, 1.0, 0.82))
    ax.imshow(
        img, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
    )
    if label_rooms:
        for room in grid.rooms:
            cx, cy = sum(room.x_range) / 2, sum(room.y_range) / 2
            ax.text(cx, cy, room.id, ha="center", va="center", fontsize=8, color="#666666")
    for k, rec in enumerate(records):
        color = style.color_hex(k)
        wp = np.asarray(rec.waypoints)
        ax.plot(wp[:, 0], wp[:, 1], "-", color=color, lw=1.8, alpha=0.9)
        ax.plot(rec.start.x, rec.start.y, "o", color=color, ms=8, mec="black", mew=0.6)
        g = rec.goal
        ax.plot(g.x, g.y, "^", color=color, ms=10, mec="black", mew=0.6)
        ax.arrow(g.x, g.y, 0.7 * math.cos(g.yaw), 0.7 * math.sin(g.yaw), width=0.05, color=color)
        # Label goals because shared starts would overlap across the four routes.
        ax.text(g.x + 0.25, g.y + 0.25, _name(rec, k), fontsize=10, weight="bold", color=color)
    handles = [
        plt.Line2D([], [], color=style.color_hex(k), lw=2,
                   label=f"{_name(r, k)}: {r.tier}, l={r.shortest_m:.1f} m, "
                         f"detour {r.detour_ratio:.2f}")
        for k, r in enumerate(records)
    ]  # fmt: skip
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _name(rec: object, k: int) -> str:
    return str(getattr(rec, "label", "") or k + 1)


def contact_sheet(
    images: Sequence[bytes], labels: Sequence[str], out_jpg: Path, cols: int = 4, width: int = 480
) -> None:
    """Draw one goal image per contact-sheet cell with an English label below."""
    style = _style()
    from PIL import Image, ImageDraw

    tiles: List[np.ndarray] = []
    font = style.pil_font(20)
    for data, label in zip(images, labels):
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        h = int(bgr.shape[0] * width / bgr.shape[1])
        tile = cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA)
        canvas = Image.new("RGB", (width, h + 34), "white")
        canvas.paste(Image.fromarray(tile[:, :, ::-1]), (0, 0))
        ImageDraw.Draw(canvas).text((8, h + 6), label, fill="black", font=font)
        tiles.append(np.asarray(canvas))
    if not tiles:
        raise ValueError("没有图")
    blank = np.full_like(tiles[0], 255)
    rows: List[np.ndarray] = []
    for r in range(0, len(tiles), cols):
        row = tiles[r : r + cols] + [blank] * (cols - len(tiles[r : r + cols]))
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), sheet[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def tier_sort_key(record: object, tiers: Optional[Sequence[str]] = None) -> tuple:
    order = list(tiers or TIER_ORDER)
    return (order.index(getattr(record, "tier")), getattr(record, "shortest_m"))
