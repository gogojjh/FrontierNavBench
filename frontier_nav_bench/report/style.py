# Upstream: tools/viz/style.py
# Original SHA256: 6c12a54afcbe5a11502723f78a33af8145f3dca6d1f39e68028e82c9c4e5ad42.
"""Shared report typography and colors, adapted from RoboHike visualization tools."""

import functools
import os
import shutil
from typing import List, Sequence, Tuple

import numpy as np

PALETTE_SIZE = 60

_FIXED_COLORS = (
    (0.0, 152 / 255, 83 / 255),
    (228 / 255, 53 / 255, 39 / 255),
    (140 / 255, 3 / 255, 120 / 255),
    (0.0, 95 / 255, 129 / 255),
    (0.9290, 0.6940, 0.1250),
    (0.6350, 0.0780, 0.1840),
    (0.494, 0.184, 0.556),
    (0.850, 0.3250, 0.0980),
    (0.466, 0.674, 0.188),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
    (0.494, 0.184, 0.556),
    (0.850, 0.3250, 0.0980),
    (0.466, 0.674, 0.188),
)
_PALETTE_SEED = 0

_FONT_DIRS = ("/usr/share/fonts/opentype/urw-base35", "/usr/share/fonts/truetype/dejavu")
_FONT_FILES = {
    False: ("P052-Roman.otf", "DejaVuSerif.ttf"),
    True: ("P052-Bold.otf", "DejaVuSerif-Bold.ttf"),
}


class FontNotFoundError(RuntimeError):
    """Fontnotfounderror."""


def acquire_color_palette() -> np.ndarray:
    """Acquire color palette."""
    t = np.linspace(-510, 510, PALETTE_SIZE)
    palette = np.round(np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255))
    palette = palette.astype("float32") / 255
    palette[: len(_FIXED_COLORS)] = _FIXED_COLORS
    rng = np.random.RandomState(_PALETTE_SEED)
    palette[len(_FIXED_COLORS) :] = rng.random_sample((PALETTE_SIZE - len(_FIXED_COLORS), 3))
    return palette


def acquire_marker() -> List[str]:
    """Acquire marker."""
    return ["o", "s", "^", "D", "X", "*", "+", "o", "s", "^", "D", "*", "X"]


def acquire_linestyle() -> List[str]:
    """Acquire linestyle."""
    return ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def acquire_bar_style() -> List[str]:
    """Acquire bar style."""
    return ["", "///", "xx", "o", "O", ".", "*", "-", "+", "x"]


def setting_font(
    fontsize: float = 14,
    titlesize: float = 14,
    legend_fontsize: float = 14,
    font_family: str = "Palatino",
) -> None:
    """Setting font."""
    import matplotlib

    usetex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "P052", "DejaVu Serif"],
            "font.size": fontsize,
            "text.usetex": usetex,
            "axes.titlesize": titlesize,
            "legend.fontsize": legend_fontsize,
            "legend.numpoints": 1,
        }
    )


def color_rgb255(index: int) -> Tuple[int, int, int]:
    """Color rgb255."""
    r, g, b = np.round(acquire_color_palette()[index].astype(np.float64) * 255).astype(int)
    return int(r), int(g), int(b)


def color_bgr255(index: int) -> Tuple[int, int, int]:
    """Color bgr255."""
    r, g, b = color_rgb255(index)
    return b, g, r


def color_hex(index: int) -> str:
    """Color hex."""
    return "#{:02X}{:02X}{:02X}".format(*color_rgb255(index))


def lighten(color: Sequence[int], amount: float = 0.45) -> Tuple[int, int, int]:
    """Lighten."""
    r, g, b = (int(round(c + (255 - c) * amount)) for c in color)
    return r, g, b


def font_file(bold: bool = False) -> str:
    """Font file."""
    for name in _FONT_FILES[bold]:
        for folder in _FONT_DIRS:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    try:
        import matplotlib

        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", _FONT_FILES[bold][1])
        if os.path.isfile(path):
            return path
    except ImportError:
        pass
    raise FontNotFoundError("Install fonts-urw-base35 or fonts-dejavu-core for report typography.")


@functools.lru_cache(maxsize=64)
def pil_font(size: int, bold: bool = False):
    """Pil font."""
    from PIL import ImageFont

    return ImageFont.truetype(font_file(bold), size)
