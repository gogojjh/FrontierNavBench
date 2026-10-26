#!/usr/bin/env python3
"""Run AM/AMem/AMM navigation using fixed published RGB map selections."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnav_benchmark.ablations import EPISODE_SETS, parse_variants, run_ablations  # noqa: E402
from vnav_benchmark.runner import CampaignStopped  # noqa: E402


def main() -> int:
    """Prepare with --max-jobs 0, or run/resume the requested variant campaigns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", choices=EPISODE_SETS, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--variants", type=parse_variants, default=parse_variants("AM,AMem,AMM"))
    parser.add_argument("--backend", choices=("kinematic", "isaac"), required=True)
    parser.add_argument("--slot", choices=("s1", "s2"), default="s1")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Maximum new runs across variants; 0 only checks inputs and plans")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        results = run_ablations(args.episodes, args.campaign, args.variants,
                                args.backend, args.slot, args.max_jobs)
    except (CampaignStopped, ValueError) as exc:
        parser.exit(3, f"ABLATION_STOPPED {exc}\n")
    print(json.dumps({variant: {key: value[key] for key in ("total", "done")}
                      for variant, value in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
