#!/usr/bin/env python3
"""Print campaign coverage by method, status, pending work and paired cells (H7).

Example:
    python scripts/coverage_report.py <data_root>/campaigns/v1_main
Budget-blocked cells are included in the unfinished-work list."""

from __future__ import annotations

import json
import sys
from pathlib import Path

STATUS_ORDER = (
    "scored",
    "unscored_infra",
    "unscored_guard",
    "excluded_debug",
    "budget_blocked",
    "pending",
)


def render(cov: dict) -> str:
    lines = [
        f"共 {cov['total']} 格,已结束 {cov['done']} 格;"
        f"所有方法都计分的配对格 {len(cov['paired_cells'])} 个",
        "",
    ]
    lines.append("| 方法 | " + " | ".join(STATUS_ORDER) + " |")
    lines.append("|---" * (len(STATUS_ORDER) + 1) + "|")
    for method, counts in cov["by_method"].items():
        lines.append(
            f"| `{method}` | " + " | ".join(str(counts.get(s, 0)) for s in STATUS_ORDER) + " |"
        )
    todo = [c for c in cov["cells"] if c["status"] in ("pending", "budget_blocked")]
    if todo:
        lines += ["", "没跑完的格子:"]
        lines += [f"- {c['method_id']} / {c['token']} / r{c['repeat']}:{c['status']}" for c in todo]
    for key, why in cov.get("blocked", {}).items():
        lines.append(f"- 花费闸:{key}:{why}")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cov = json.loads((Path(sys.argv[1]) / "coverage.json").read_text(encoding="utf-8"))
    print(render(cov))
    return 0


if __name__ == "__main__":
    sys.exit(main())
