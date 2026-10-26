"""Classify failed episodes from run.json outcomes and method summaries, without images.

Categories are checked in order, and the first match wins:
1. said_done: the method declares arrival outside 0.5 m / 45 deg, or M2/M3 times out after pose
checks reject a VLM done declaration (vlm_done_ignored > 0).
2. wall_subgoal: a B0/M1 point-goal timeout, attributed to SRU pursuing coordinates behind a
wall that the walking policy cannot navigate around.
3. not_recognized: an image-goal timeout within the loose 2 m goal radius.
4. lost: an image-goal timeout farther away, consistent with losing the route or goal
photographs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from frontier_nav_bench import metrics
from frontier_nav_bench.fairness import parse_method_id
from frontier_nav_bench.report import aggregate as agg
from frontier_nav_bench.report import figures
from frontier_nav_bench.report.results_io import REPORT_METHOD_IDS
from frontier_nav_bench.report.tables import Table, _label

FAILURE_TYPES = ("said_done", "wall_subgoal", "not_recognized", "lost")
FAILURE_LABELS = {
    "said_done": "Said done too early",
    "wall_subgoal": "Subgoal behind a wall",
    "not_recognized": "Near goal, not recognized",
    "lost": "Lost the route",
}
# (Method, episode, failure type, outcome).
FailureRow = Tuple[str, str, Optional[str], Dict[str, Any]]


def classify(run: Dict[str, Any], summary: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return None for successful episodes, otherwise a FAILURE_TYPES category."""
    out = run["outcome"]
    if out["success"]:
        return None
    if out["termination"] == "done" or (summary or {}).get("vlm_done_ignored", 0) > 0:
        return "said_done"
    if parse_method_id(run["method_id"]).planner == "ptgoal":
        return "wall_subgoal"
    if out["final_dist_m"] is not None and out["final_dist_m"] <= metrics.SUCCESS_DIST_LOOSE_M:
        return "not_recognized"
    return "lost"


def load_failures(campaign_dir: Path) -> List[FailureRow]:
    """Return (method, episode, optional failure type, outcome) for scored first runs (r0)."""
    rows = []
    for path in sorted(campaign_dir.glob(f"runs/*/*/r{agg.MAIN_REPEAT}/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") != "scored":
            continue
        summary = (run.get("method_stats") or {}).get("summary")
        kind = classify(run, summary)
        rows.append((run["method_id"], run["episode_token"], kind, run["outcome"]))
    return rows


def failure_counts(rows: Sequence[FailureRow]) -> Dict[str, Dict[str, int]]:
    """Count each failure type and total episodes per method, in table order."""
    present = {m for m, _, _, _ in rows}
    out: Dict[str, Dict[str, int]] = {}
    for mid in [m for m in REPORT_METHOD_IDS if m in present]:
        mine = [t for m, _, t, _ in rows if m == mid]
        out[mid] = {t: mine.count(t) for t in FAILURE_TYPES}
        out[mid]["n"] = len(mine)
    return out


def failure_table(counts: Dict[str, Dict[str, int]]) -> Table:
    out = [
        [_label(mid), mid, str(c["n"]), str(c["n"] - sum(c[t] for t in FAILURE_TYPES))]
        + [str(c[t]) for t in FAILURE_TYPES]
        for mid, c in counts.items()
    ]
    return Table(
        "failures",
        "Failure types (episodes, first run)",
        ["ID", "Method", "Episodes", "Success"] + [FAILURE_LABELS[t] for t in FAILURE_TYPES],
        out,
        [
            "Said done too early: the method declared done outside 0.5 m / 45°, or (M2 / M3) the "
            "VLM said done, the pose check did not accept it and time ran out.",
            "Subgoal behind a wall: point-goal methods (B0 / M1) timed out while SRU pushed "
            "toward a subgoal it could not walk around to.",
            f"Near goal, not recognized: VLM method timed out within "
            f"{metrics.SUCCESS_DIST_LOOSE_M:g} m of the goal. Lost the route: timed out farther "
            "away.",
            "Classified from logged outcomes only (termination, final distance, ignored VLM done); "
            "spot-checked on the episode stills.",
        ],
    )


def failure_bars(counts: Dict[str, Dict[str, int]], out_stem: Path) -> Path:
    """Draw one stacked bar per configuration for successes and four failure categories."""
    style = figures._style()
    import matplotlib.pyplot as plt

    mids = list(counts)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(mids) + 1.2))
    left = [0.0] * len(mids)
    parts = [("success", "Success", "#BBBBBB")] + [
        (t, FAILURE_LABELS[t], style.color_hex(i + 5)) for i, t in enumerate(FAILURE_TYPES)
    ]
    for key, label, color in parts:
        vals = [
            counts[m]["n"] - sum(counts[m][t] for t in FAILURE_TYPES) if key == "success"
            else counts[m][key]
            for m in mids
        ]  # fmt: skip
        ax.barh(range(len(mids)), vals, left=left, color=color, label=label, edgecolor="white")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(mids)))
    ax.set_yticklabels([_label(m) for m in mids])
    ax.set_xlabel("Episodes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(mids)), ncol=3,
              frameon=False)  # fmt: skip
    ax.set_xlim(0, max(c["n"] for c in counts.values()))
    return figures._save(fig, out_stem)
