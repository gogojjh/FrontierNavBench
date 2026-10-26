#!/usr/bin/env python3
"""Run a benchmark campaign; execution lives in vnav_benchmark/runner.py (H7).

Kinematic rehearsal without occupying the simulator:
    python scripts/campaign_run.py --campaign drill_kinematic --episodes env0_dev_v1 \\
        --methods debug-oracle-walker,debug-stop-now --backend kinematic
Formal campaign after obtaining the slot lock and passing preflight:
    python scripts/campaign_run.py --campaign v1_main --episodes env0_test_v3 \\
        --methods v1 --backend isaac --slot s1
Rerun the same command to resume; cells with run.json are skipped and arguments
must match. Outputs go to <data_root>/campaigns/<name>/. Inspect coverage.json or
scripts/coverage_report.py for progress."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vnav_benchmark import fairness  # noqa: E402
from vnav_benchmark.ablations import CheckedProvider, reserve_slot  # noqa: E402
from vnav_benchmark.budget import BudgetGuard  # noqa: E402
from vnav_benchmark.episodes import load_episode_set  # noqa: E402
from vnav_benchmark.paths import data_root  # noqa: E402
from vnav_benchmark.runner import (  # noqa: E402
    BenchConfig,
    Campaign,
    CampaignStopped,
    KinematicProvider,
    MapsYamlSource,
    RunContext,
    v1_plan,
)

DATA_ROOT = data_root()
EPISODES_DIR = REPO / "configs" / "episodes"


def git_info() -> dict:
    out = {}
    for name, path in (("vnav_benchmark", REPO),):
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()  # fmt: skip
        status = ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]
        dirty = subprocess.run(status,
                               capture_output=True, text=True).stdout.strip()  # fmt: skip
        out[name] = {"head": head, "dirty": bool(dirty)}
    return out


def campaign_alert(message: str) -> None:
    """Print campaign budget and circuit-breaker alerts locally."""
    print(message, flush=True)


def parse_methods(text: str) -> List[str]:
    if text == "v1":
        return list(fairness.V1_METHOD_IDS)
    methods = [m.strip() for m in text.split(",") if m.strip()]
    for mid in methods:
        fairness.parse_method_id(mid)
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--campaign", required=True, help="Output campaign name under data/campaigns/"
    )
    parser.add_argument(
        "--episodes", required=True, help="Episode set name or path to a custom frozen set"
    )
    parser.add_argument(
        "--methods",
        required=True,
        help="Comma-separated method IDs, or v1 for all 12 configurations",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="Repeat count; v1 uses its fixed schedule"
    )
    parser.add_argument("--backend", choices=("kinematic", "isaac"), required=True)
    parser.add_argument(
        "--slot", choices=("s1", "s2"), default="s1", help="Exclusive Isaac simulator slot"
    )
    parser.add_argument(
        "--video", default="", help="Comma-separated episode tokens to record in third person"
    )
    parser.add_argument(
        "--max-jobs", type=int, default=None, help="Maximum number of new cells; zero prepares only"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    root = DATA_ROOT / "campaigns" / args.campaign
    root.mkdir(parents=True, exist_ok=True)
    episodes_dir = Path(args.episodes) if "/" in args.episodes else EPISODES_DIR / args.episodes
    es = load_episode_set(episodes_dir)
    methods = parse_methods(args.methods)
    video = {t for t in args.video.split(",") if t}
    if args.backend == "kinematic":
        sim = KinematicProvider(
            __import__("vnav_benchmark.oracle", fromlist=["x"]).build_truth_grid()
        )
    else:
        from vnav_benchmark.sim.orchestrator import IsaacOrchestrator

        sim = CheckedProvider(IsaacOrchestrator(args.slot), episodes_dir, BenchConfig.load().robot)
    needs_map = any(fairness.parse_method_id(m).map_variant for m in methods)
    ctx = RunContext(
        root=root,
        episode_set=es,
        sim=sim,
        maps=MapsYamlSource(root / "maps") if needs_map else None,
        guard=BudgetGuard(root / "ledger.jsonl"),
        bench=BenchConfig.load(),
        want_third=lambda job: job.token in video,
        git=git_info(),
    )
    # v1 uses its fixed job plan; other campaigns share episode/repeat counts.
    if args.methods == "v1":
        campaign = Campaign(ctx, tuple(methods), tuple(es.tokens()), 1, campaign_alert, v1_plan(es))
    else:
        campaign = Campaign(ctx, tuple(methods), tuple(es.tokens()), args.repeats, campaign_alert)
    try:
        with reserve_slot(args.slot, args.backend):
            cov = campaign.run(max_jobs=args.max_jobs)
    except CampaignStopped as exc:
        print(f"CAMPAIGN_STOPPED {exc}", flush=True)
        return 3
    print(json.dumps({k: cov[k] for k in ("total", "done", "by_method")}, ensure_ascii=False))
    print(f"CAMPAIGN_DONE {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
