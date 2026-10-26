"""Failure classification uses four ordered categories with matching counts/table rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from vnav_benchmark.report import failures as fl


def _run(mid: str, success: bool, term: str, dist: float) -> Dict[str, Any]:
    return {"method_id": mid, "outcome": {"success": success, "termination": term,
                                          "final_dist_m": dist}}  # fmt: skip


def test_classify_order() -> None:
    img, pt = "dense-contloc-imggoal-astra", "sparse-contloc-ptgoal"
    ignored: Optional[Dict[str, Any]] = {"vlm_done_ignored": 3}
    assert fl.classify(_run(img, True, "done", 0.2), None) is None
    assert fl.classify(_run(img, False, "done", 1.0), None) == "said_done"
    assert fl.classify(_run(img, False, "timeout", 0.6), ignored) == "said_done"
    assert fl.classify(_run(pt, False, "timeout", 1.5), {"vlm_done_ignored": 0}) == "wall_subgoal"
    assert fl.classify(_run(img, False, "timeout", 1.9), None) == "not_recognized"
    assert fl.classify(_run(img, False, "timeout", 6.0), None) == "lost"


def test_counts_table_and_bars(tmp_path: Path) -> None:
    rows = [
        ("dense-contloc-ptgoal", "ep_a", None, {}),
        ("dense-contloc-ptgoal", "ep_b", "wall_subgoal", {}),
        ("nomap-noloc-imggoal-qwen", "ep_a", "lost", {}),
    ]
    counts = fl.failure_counts(rows)
    assert list(counts) == ["dense-contloc-ptgoal", "nomap-noloc-imggoal-qwen"]
    t = fl.failure_table(counts)
    assert t.rows[0][:5] == ["B0", "dense-contloc-ptgoal", "2", "1", "0"]
    assert t.rows[0][5] == "1" and t.rows[1][-1] == "1"
    assert fl.failure_bars(counts, tmp_path / "f").exists()
