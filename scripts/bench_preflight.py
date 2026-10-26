"""Run campaign preflight; checks live in frontier_nav_bench/preflight.py.

Examples without simulator GPU use, or with a temporary server identity check:
    python scripts/bench_preflight.py
    python scripts/bench_preflight.py --with-server --slot s1 \\
        --episodes configs/episodes/env0_test_v2/episodes.json
The server option requires the slot lock and compares scene MD5, USD SHA, pins
and Git versions. Remote execution requires SSH access. Reports are printed as a
table and saved to <data_root>/preflight/<timestamp>.json. Any failed check
returns exit code 1."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontier_nav_bench.oracle import SCENE_ENV0  # noqa: E402
from frontier_nav_bench.preflight import run_preflight  # noqa: E402
from frontier_nav_bench.sim.orchestrator import (  # noqa: E402
    IsaacOrchestrator,
    SubprocessRunner,
    load_ports_config,
)

# Default env0 spawn for a hello-only check: (12.5, 2.6), yaw 90 degrees.
PROBE_POSE = ((12.5, 2.6), 1.5708)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default=SCENE_ENV0)
    parser.add_argument("--with-server", action="store_true", help="起一次服务取 hello 比对")
    parser.add_argument("--slot", default="s1", help="Simulator slot s1 / s2(--with-server 时用)")
    parser.add_argument("--episodes", type=Path, help="冻结起终点集 episodes.json(比对 USD sha)")
    args = parser.parse_args()
    config = load_ports_config()
    runner = SubprocessRunner()
    out_dir = config.data_root / "preflight"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    hello = None
    if args.with_server:
        orch = IsaacOrchestrator(args.slot, config, runner)
        try:
            orch.start(args.scene, *PROBE_POSE, out_dir / f"{stamp}_server.log")
            hello = orch.identity()
        finally:
            orch.stop()
    report = run_preflight(config, runner, args.scene, hello, args.episodes)
    print(report.markdown())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stamp}.json").write_text(
        json.dumps(report.to_json(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
