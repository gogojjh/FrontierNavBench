"""Build all campaign tables (R1) and figures (R2) from recorded results.

Example:
    python scripts/report_build.py CAMPAIGN_DIR
Outputs go to report/: per-table .md/.tex/.csv files and combined tables.md.
Figures are saved as PNG and PDF in report/figures/. Trajectory facets require
trace.csv in episode directories. Mixed VLM prompt fingerprints and excluded
development endpoints such as luna_dev are rejected."""

import json
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vnav_benchmark.report import aggregate, figures, tables  # noqa: E402
from vnav_benchmark.report.aggregate import ReportInputError  # noqa: E402
from vnav_benchmark.report.results_io import load_runs  # noqa: E402

REPORT_SUBDIR = "report"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    return build(Path(sys.argv[1]))


def build(campaign: Path, extra_notes: Sequence[str] = ()) -> int:
    """Build tables and figures; append extra_notes to the main table caption."""
    rows = load_runs(campaign)
    if not rows:
        print(f"没找到 runs/*/*/r*/run.json: {campaign}")
        return 1
    try:
        built = tables.build_all(rows, extra_notes)
    except ReportInputError as exc:
        print(f"拒绝出表: {exc}")
        return 1
    paths = tables.write_all(built, campaign / REPORT_SUBDIR)
    main_rows = aggregate.scored(rows)
    tracks = figures.load_tracks(campaign, main_rows)
    grid = None
    if tracks:
        from vnav_benchmark.oracle import build_truth_grid

        first = campaign / "runs" / tracks[0].method_id / tracks[0].episode_id / "r0" / "run.json"
        grid = build_truth_grid(json.loads(first.read_text())["episode"]["scene"])
    common = aggregate.common_episodes(main_rows)  # Use pairs shared by all configurations.
    shared = [r for r in main_rows if r.episode_id in common]
    figs = figures.build_all_figures(shared, campaign / REPORT_SUBDIR / "figures", tracks, grid)
    print(
        f"{len(rows)} runs → {len(paths)} tables, {len(figs)} figures in {campaign / REPORT_SUBDIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
