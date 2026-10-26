"""Render campaign results and method-by-route trajectories after a run.

The implementation lives in frontier_nav_bench/campaign_view.py. Example:
    python scripts/campaign_view.py CAMPAIGN_DIR [--repeat 0]
Writes view/results.md and trajectories.png, with _r<k> suffixes when a repeat is
selected. Saved first-person images produce frames.jpg; recorded SRU speed
statistics are appended to the results table."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontier_nav_bench.campaign_view import (  # noqa: E402
    collect,
    draw,
    frame_sheet,
    markdown,
    sru_speed_markdown,
)
from frontier_nav_bench.oracle import build_truth_grid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--repeat", type=int, default=None, help="只看第几遍(从 0 起)")
    args = parser.parse_args()
    rows = collect(args.campaign, args.repeat)
    suffix = "" if args.repeat is None else f"_r{args.repeat}"
    out = args.campaign / "view"
    out.mkdir(parents=True, exist_ok=True)
    speed = sru_speed_markdown(rows)  # Only smoke methods record SRU speed statistics.
    table = markdown(rows) + ("\n" + speed if speed else "")
    (out / f"results{suffix}.md").write_text(table, encoding="utf-8")
    draw(rows, build_truth_grid(), out / f"trajectories{suffix}.png")
    if frame_sheet(rows, out / f"frames{suffix}.jpg"):  # Requires saved frames.
        print(f"frames: {out / f'frames{suffix}.jpg'}")
    print(table)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
