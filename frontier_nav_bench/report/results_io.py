"""Flatten each run.json and optional VLM summary into one RunRow.

The run.json schema is vnav.run/v1. All field paths are centralized in row_from_run so format
changes have one update point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

from frontier_nav_bench.fairness import parse_method_id

# Method prefixes map to B0-M6 codes; order also defines table rows and colors.
METHOD_CODES = {
    "dense-contloc-ptgoal": "B0",
    "sparse-contloc-ptgoal": "M1",
    "dense-contloc-imggoal": "M2",
    "sparse-contloc-imggoal": "M3",
    "dense-initloc-imggoal": "M4",
    "sparse-initloc-imggoal": "M5",
    "nomap-noloc-imggoal": "M6",
}
VLM_ORDER = ("astra", "qwen")
# All report configurations, ordered B0, M1, M2-Astra, M2-Qwen through M6-Qwen.
REPORT_METHOD_IDS = tuple(
    base if base.endswith("ptgoal") else f"{base}-{vlm}"
    for base in METHOD_CODES
    for vlm in ((None,) if base.endswith("ptgoal") else VLM_ORDER)
)
# Image-goal method summary within an episode directory.
METHOD_SUMMARY_REL = Path("method") / "vlm" / "summary.json"


@dataclass(frozen=True)
class RunRow:
    method_id: str
    code: str  # B0 / M1 / M2 / M3 / M4
    vlm: Optional[str]  # "astra" | "qwen" | None
    episode_id: str
    repeat: int
    status: str  # scored / unscored_infra / excluded_debug
    termination: str
    success: bool
    success_loose: bool
    spl: float
    plr: Optional[float]
    path_m: float
    shortest_m: float
    sim_time_s: float
    budget_s: float  # Episode simulation-time budget from run.json settings.budget_s.
    decisions: int
    act_med_s: Optional[float]
    act_p95_s: Optional[float]
    act_max_s: Optional[float]
    compute_per_sim: Optional[float]
    # Difficulty tier from run.json episode.tier; shortest paths use 3-6 / 6-12 / 12-25 m bins.
    band: str
    same_room: Optional[bool]
    ever_within: bool
    premature_done: bool
    falls: int
    # StuckMonitor counts displacement below 0.3 m over 15 s without declaring failure.
    stuck_events: int
    # is_proximity counts median depth below 0.3 m in the center third of the image.
    proximity_events: int
    list_usd: float
    billed_usd: float
    calls: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    endpoint: str  # Image-goal endpoint name; empty for other methods.
    service_tier: str  # "flex" | "default" | ""
    fingerprint: str  # Prompt fingerprint; empty for other methods.
    vlm_api_med_s: Optional[float]
    server_ms_med: Optional[float]
    invalid_outputs: int
    subgoal_advances: int
    done_judge: str  # Arrival judge: pose for B0/M1/M2/M3, otherwise vlm.
    vlm_done_ignored: int  # VLM done declarations rejected by the pose-based arrival judge.

    @property
    def scored(self) -> bool:
        return self.status == "scored"


def _get(d: Optional[Mapping[str, Any]], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def row_from_run(run: Mapping[str, Any], summary: Optional[Mapping[str, Any]] = None) -> RunRow:
    """Convert a run.json dictionary and optional image-goal summary into a RunRow."""
    method_id = str(run["method_id"])
    profile = parse_method_id(method_id)
    code = METHOD_CODES["-".join(method_id.split("-")[:3])]
    out = run["outcome"]
    usage = run.get("usage") or {}
    s = summary if summary is not None else _get(run, "method_stats", "summary")
    act = out.get("act_wall_s") or {}
    endpoint = _get(s, "endpoint", "endpoint", default="") or ""
    return RunRow(
        method_id=method_id,
        code=code,
        vlm=profile.vlm,
        episode_id=str(run["episode_token"]),
        repeat=int(run.get("repeat_index", 0)),
        status=str(run.get("status", "scored")),
        termination=str(out["termination"]),
        success=bool(out["success"]),
        success_loose=bool(out["success_loose"]),
        spl=float(out["spl"]),
        plr=None if out.get("plr") is None else float(out["plr"]),
        path_m=float(out["path_length_m"]),
        shortest_m=float(out["shortest_path_m"]),
        sim_time_s=float(out["sim_time_s"]),
        budget_s=float(_get(run, "settings", "budget_s", default=0.0) or 0.0),
        decisions=int(out["decisions"]),
        act_med_s=act.get("median"),
        act_p95_s=act.get("p95"),
        act_max_s=act.get("max"),
        compute_per_sim=out.get("compute_per_sim_s"),
        band=str(_get(run, "episode", "tier", default="")),
        same_room=_get(run, "episode", "same_room"),
        ever_within=bool(out.get("ever_within_pose", False)),
        premature_done=bool(out.get("premature_done", False)),
        falls=int(out.get("falls", 0)),
        stuck_events=int(out.get("stuck_events", 0)),
        proximity_events=int(out.get("proximity_events", 0)),
        list_usd=float(usage.get("list_usd", _get(s, "usage", "list_usd", default=0.0))),
        billed_usd=float(usage.get("billed_usd", _get(s, "usage", "billed_usd", default=0.0))),
        calls=int(usage.get("calls", _get(s, "usage", "calls", default=0))),
        input_tokens=int(usage.get("input_tokens", _get(s, "usage", "input_tokens", default=0))),
        cached_tokens=int(usage.get("cached_tokens", _get(s, "usage", "cached_tokens", default=0))),
        output_tokens=int(usage.get("output_tokens", _get(s, "usage", "output_tokens", default=0))),
        reasoning_tokens=int(
            usage.get("reasoning_tokens", _get(s, "usage", "reasoning_tokens", default=0))
        ),
        endpoint=str(endpoint),
        service_tier=str(_get(s, "endpoint", "service_tier", default="") or ""),
        fingerprint=str(_get(s, "fingerprint", default="") or ""),
        vlm_api_med_s=_get(s, "latency", "vlm_api_s", "median"),
        server_ms_med=_get(s, "latency", "server_ms", "median"),
        invalid_outputs=int(_get(s, "policy_stats", "invalid", default=0)),
        subgoal_advances=int(_get(s, "subgoal_advances", default=0)),
        done_judge=profile.done_judge,
        vlm_done_ignored=int(_get(s, "vlm_done_ignored", default=0)),
    )


def load_runs(campaign_dir: Path) -> List[RunRow]:
    """Load campaign runs/<method>/<episode>/r<k>/run.json files."""
    rows: List[RunRow] = []
    for path in sorted(campaign_dir.glob("runs/*/*/r*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        summary_path = path.parent / METHOD_SUMMARY_REL
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        )
        rows.append(row_from_run(run, summary))
    return rows
