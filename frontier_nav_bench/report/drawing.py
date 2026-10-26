# Upstream: tools/viz/draw.py
# Original SHA256: 1837fe129380f0d690e28205c4799b84b68a02a5f9473f343e404e4a4973073b.
"""Pixel-space panels, cached text sprites and clipped alpha composition."""

import functools
import math
from typing import NamedTuple, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import style


class Rect(NamedTuple):
    """Rect."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h


def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Rgb to bgr."""
    r, g, b = rgb
    return int(b), int(g), int(r)


TEXT_PAD_PX = 3


@functools.lru_cache(maxsize=2048)
def text_sprite(text: str, size: int, bold: bool) -> np.ndarray:
    """Text sprite."""
    font = style.pil_font(size, bold)
    w, h = font.getsize(text)
    pad = TEXT_PAD_PX
    img = Image.new("L", (max(w, 1) + pad * 2, max(h, 1) + pad * 2), 0)
    ctx = ImageDraw.Draw(img)
    ctx.text((pad, pad), text, fill=255, font=font)
    return np.asarray(img, dtype=np.float32) / 255.0


def text_size(text: str, size: int, bold: bool) -> Tuple[int, int]:
    """Text size."""
    sprite = text_sprite(text, size, bold)
    h, w = sprite.shape
    return w - 2 * TEXT_PAD_PX, h - 2 * TEXT_PAD_PX


def blend(img: np.ndarray, x: int, y: int, color: np.ndarray, alpha: np.ndarray) -> None:
    """Blend."""
    h, w = alpha.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0 - y : y1 - y, x0 - x : x1 - x, None]
    patch = img[y0:y1, x0:x1].astype(np.float32)
    blended = patch * (1.0 - a) + color[y0 - y : y1 - y, x0 - x : x1 - x] * a
    img[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


def blit_text(
    img_bgr: np.ndarray,
    x: float,
    y: float,
    text: str,
    size: int,
    bold: bool,
    color_bgr: Tuple[int, int, int],
    align: str = "left",
) -> int:
    """Blit text."""
    alpha = text_sprite(text, size, bold)
    pad = TEXT_PAD_PX
    sh, sw = alpha.shape
    w = sw - 2 * pad
    if align == "center":
        left = x - w / 2.0
    elif align == "right":
        left = x - w
    else:
        left = x
    x0 = int(round(left)) - pad
    y0 = int(round(y)) - pad
    color_img = np.broadcast_to(np.asarray(color_bgr, dtype=np.float32), (sh, sw, 3))
    blend(img_bgr, x0, y0, color_img, alpha)
    return w


def rounded_rect(img: np.ndarray, rect: Rect, color_bgr: Tuple[int, int, int], radius: int) -> None:
    """Rounded rect."""
    x, y, w, h = rect
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color_bgr, -1)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color_bgr, -1)
    for cx, cy in (
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius),
    ):
        cv2.circle(img, (cx, cy), radius, color_bgr, -1, cv2.LINE_AA)


PANEL_CHIP_SIZE = 17

_CHIP_PAD_X, _CHIP_PAD_Y, _CHIP_RADIUS = 9, 4, 6
_CHIP_ALPHA = 0.92


@functools.lru_cache(maxsize=64)
def label_chip(
    text: str,
    size: int = PANEL_CHIP_SIZE,
    bold: bool = True,
    text_rgb: Tuple[int, int, int] = (0x10, 0x16, 0x20),
    bg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, np.ndarray]:
    """Label chip."""
    glyph = text_sprite(text, size, bold)
    gh, gw = glyph.shape
    h, w = gh + 2 * _CHIP_PAD_Y, gw + 2 * _CHIP_PAD_X
    mask = np.zeros((h, w), dtype=np.uint8)
    rounded_rect(mask, Rect(0, 0, w - 1, h - 1), 255, _CHIP_RADIUS)
    alpha = mask.astype(np.float32) / 255.0 * _CHIP_ALPHA
    color = np.empty((h, w, 3), dtype=np.float32)
    color[:] = np.array(rgb_to_bgr(bg_rgb), dtype=np.float32)
    text_bgr = np.array(rgb_to_bgr(text_rgb), dtype=np.float32)
    region = color[_CHIP_PAD_Y : _CHIP_PAD_Y + gh, _CHIP_PAD_X : _CHIP_PAD_X + gw]
    region[:] = region * (1.0 - glyph[..., None]) + text_bgr * glyph[..., None]
    alpha[_CHIP_PAD_Y : _CHIP_PAD_Y + gh, _CHIP_PAD_X : _CHIP_PAD_X + gw] = np.maximum(
        alpha[_CHIP_PAD_Y : _CHIP_PAD_Y + gh, _CHIP_PAD_X : _CHIP_PAD_X + gw], glyph
    )
    return color, alpha


def paste(img: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Paste."""
    ph, pw = patch.shape[:2]
    ih, iw = img.shape[:2]
    sy0, sy1 = max(0, -y), min(ph, ih - y)
    sx0, sx1 = max(0, -x), min(pw, iw - x)
    if sy1 <= sy0 or sx1 <= sx0:
        return
    img[y + sy0 : y + sy1, x + sx0 : x + sx1] = patch[sy0:sy1, sx0:sx1]


def placeholder(
    img: np.ndarray,
    rect: Rect,
    text: str,
    size: int,
    bg_bgr: Tuple[int, int, int],
    text_bgr: Tuple[int, int, int],
) -> None:
    """Placeholder."""
    img[rect.y : rect.y1, rect.x : rect.x1] = bg_bgr
    cx = rect.x + rect.w // 2
    _text_w, text_h = text_size(text, size, False)
    cy_top = rect.y + rect.h // 2 - text_h // 2
    blit_text(img, cx, cy_top, text, size, False, text_bgr, align="center")


def star_points(center: np.ndarray, radius: float, rotation: float = -math.pi / 2.0) -> np.ndarray:
    """Star points."""
    pts = []
    for i in range(10):
        r = radius if i % 2 == 0 else radius * 0.45
        ang = rotation + i * math.pi / 5.0
        pts.append((center[0] + r * math.cos(ang), center[1] + r * math.sin(ang)))
    return np.asarray(pts, dtype=np.float64)


def square(
    img: np.ndarray,
    center: Tuple[int, int],
    side: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    line_type: int,
) -> None:
    """Square."""
    half = side // 2
    x, y = center
    cv2.rectangle(img, (x - half, y - half), (x + half, y + half), fill, -1)
    cv2.rectangle(img, (x - half, y - half), (x + half, y + half), outline, 2, line_type)


def triangle_down(
    img: np.ndarray,
    tip: Tuple[int, int],
    w: int,
    h: int,
    fill_bgr: Tuple[int, int, int],
    outline_bgr: Tuple[int, int, int],
    outline_thickness: int,
) -> None:
    """Triangle down."""
    x, y = tip
    pts = np.array([[x, y], [x - w // 2, y - h], [x + w // 2, y - h]], dtype=np.int32)
    cv2.fillPoly(img, [pts], fill_bgr, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, outline_bgr, outline_thickness, cv2.LINE_AA)


def arrow_points(
    center: Tuple[float, float],
    direction: np.ndarray,
    front: float,
    back: float,
    notch: float,
    half_width: float,
) -> np.ndarray:
    """Arrow points."""
    center_xy = np.asarray(center, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    side = np.array([-direction[1], direction[0]])
    pts = np.array(
        [
            center_xy + direction * front,
            center_xy - direction * back + side * half_width,
            center_xy - direction * notch,
            center_xy - direction * back - side * half_width,
        ]
    )
    return np.rint(pts).astype(np.int32)


def smoothstep(s: float) -> float:
    """Smoothstep."""
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def format_mmss(seconds: float) -> str:
    """Format mmss."""
    total = max(0, int(round(seconds)))
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"
