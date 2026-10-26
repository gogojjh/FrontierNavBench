"""Shared method colors for paper figures, episode videos and comparison videos.

Colors use the tools/viz/style.py palette: B0 and M1-M5 take indices 0-5; M6 takes light blue at
index 9 (#4DBEEE). Index 6 resembles M2 purple, 7 resembles M1 red and 8 resembles B0 green.
Dark gray is reserved for top-down walls.
"""

from __future__ import annotations

from typing import Any, Tuple

CODE_COLOR_INDEX = {"B0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 9}
UNKNOWN_COLOR_INDEX = 10  # Fallback for unlisted codes, such as debug methods.


def _style() -> Any:
    from vnav_benchmark.report import style

    return style


def color_index(code: str) -> int:
    return CODE_COLOR_INDEX.get(code, UNKNOWN_COLOR_INDEX)


def code_color_hex(code: str) -> str:
    return str(_style().color_hex(color_index(code)))


def code_color_rgb(code: str) -> Tuple[int, int, int]:
    return tuple(_style().color_rgb255(color_index(code)))  # type: ignore[return-value]


def code_color_bgr(code: str) -> Tuple[int, int, int]:
    return tuple(_style().color_bgr255(color_index(code)))  # type: ignore[return-value]
