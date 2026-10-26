"""Shared report-test fixtures: synthetic run.json records and VLM summaries, without real data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frontier_nav_bench.fairness import parse_method_id
from frontier_nav_bench.report.results_io import REPORT_METHOD_IDS

FINGERPRINT = "sha256:test-fingerprint"
EPISODES = [f"ep{i}" for i in range(8)]
BANDS = ["short"] * 3 + ["medium"] * 3 + ["long"] * 2
BUDGET_S = 90.0  # Approved per-episode simulation budget: 90 s
SHORTEST = [4.0, 5.0, 5.5, 7.0, 9.0, 11.0, 15.0, 20.0]
# Per-method synthetic success rates distinguish table entries
SUCCESS_P = dict(zip(REPORT_METHOD_IDS, np.linspace(0.95, 0.1, len(REPORT_METHOD_IDS))))


def make_run(
    method_id: str,
    ep: int,
    repeat: int,
    success: bool,
    rng: np.random.Generator,
    endpoint: Optional[str] = None,
    fingerprint: str = FINGERPRINT,
    status: str = "scored",
    termination: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    profile = parse_method_id(method_id)
    short = SHORTEST[ep]
    p = short * float(rng.uniform(1.05, 1.6))
    t = p / 0.45
    out = {
        "termination": termination or ("done" if success or rng.random() < 0.5 else "timeout"),
        "success": success,
        "success_loose": success or bool(rng.random() < 0.3),
        "final_dist_m": 0.3 if success else 2.0,
        "final_yaw_err_deg": 10.0,
        "ever_within_pose": success or bool(rng.random() < 0.2),
        "premature_done": (not success) and bool(rng.random() < 0.3),
        "falls": 0,
        "stuck_events": 0,
        "proximity_events": 1,
        "path_length_m": p,
        "shortest_path_m": short,
        "spl": short / max(p, short) if success else 0.0,
        "plr": p / short if success else None,
        "sim_time_s": min(t, BUDGET_S) if success else BUDGET_S,
        "decisions": int(min(t, BUDGET_S) if success else BUDGET_S),
        "act_wall_s": {
            "mean": 8.0,
            "median": float(rng.uniform(5, 12)),
            "p95": float(rng.uniform(15, 30)),
            "max": 40.0,
        },
        "compute_per_sim_s": 7.5,
    }
    run: Dict[str, Any] = {
        "method_id": method_id,
        "episode_token": EPISODES[ep],
        "repeat_index": repeat,
        "status": status,
        "settings": {"budget_s": BUDGET_S},
        "episode": {"tier": BANDS[ep], "same_room": ep < 2, "shortest_path_m": short},
        "outcome": out,
    }
    summary = None
    if profile.vlm is not None:
        calls = out["decisions"]
        ep_name = endpoint or ("astra_flex" if profile.vlm == "astra" else "qwen")
        summary = {
            "endpoint": {
                "endpoint": ep_name,
                "service_tier": "flex" if ep_name == "astra_flex" else "default",
            },
            "fingerprint": fingerprint,
            "subgoal_advances": 3,
            "vlm_done_ignored": 2 if profile.done_judge == "pose" else 0,
            "policy_stats": {"invalid": 1},
            "usage": {
                "calls": calls,
                "input_tokens": 3000 * calls,
                "cached_tokens": 1800 * calls,
                "output_tokens": 450 * calls,
                "reasoning_tokens": 300 * calls,
                "list_usd": 0.037 * calls,
                "billed_usd": 0.018 * calls,
            },
            "latency": {"vlm_api_s": {"median": 7.0}, "server_ms": {"median": 5000.0}},
        }
    return run, summary


def synthetic(seed: int = 0) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """All 12 configurations run eight routes once; B0/M1 add two repeats to compare repeat
    variation.
    """
    rng = np.random.default_rng(seed)
    out = []
    for mid in REPORT_METHOD_IDS:
        reps = 3 if mid.endswith("ptgoal") else 1
        for rep in range(reps):
            for ep in range(8):
                out.append(make_run(mid, ep, rep, bool(rng.random() < SUCCESS_P[mid]), rng))
    return out


def write_campaign(root: Path, runs: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]) -> Path:
    """Write runs/<method>/<episode>/r<k>/run.json and method/vlm/summary.json."""
    for run, summary in runs:
        d = root / "runs" / run["method_id"] / run["episode_token"] / f"r{run['repeat_index']}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps(run))
        if summary is not None:
            (d / "method" / "vlm").mkdir(parents=True, exist_ok=True)
            (d / "method" / "vlm" / "summary.json").write_text(json.dumps(summary))
    return root
