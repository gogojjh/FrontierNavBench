"""Verify the shared method-color table across figures/videos, including distinct M6 and M2 colors.
"""

from __future__ import annotations

from frontier_nav_bench.report import colors
from frontier_nav_bench.report.results_io import METHOD_CODES


def test_every_method_has_distinct_color() -> None:
    codes = list(METHOD_CODES.values())
    assert set(codes) <= set(colors.CODE_COLOR_INDEX)
    hexes = [colors.code_color_hex(c) for c in codes]
    assert len(set(hexes)) == len(codes)
    assert colors.code_color_hex("M6").upper() == "#4DBEEE"


def test_bgr_is_reversed_rgb() -> None:
    for code in ("B0", "M2", "M6"):
        assert colors.code_color_bgr(code) == tuple(reversed(colors.code_color_rgb(code)))
