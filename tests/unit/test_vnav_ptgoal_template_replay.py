"""Verify B0/M1 nav_template refactoring against point-goal replay behavior.

golden/method_trace_ptgoal.json was generated at commit 73d74ad by vnav_method_trace_ptgoal.py,
before ptgoal_sru.py adopted the template. Compare actions, debug fields, summaries, output-file
contents and fake-SRU subgoals. Regenerate the golden after relaxing arrival tolerance from 0.35
m to 0.5 m; temporarily restoring 0.35 m reproduced the old golden exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest
import vnav_method_trace_ptgoal as trace

from vnav_benchmark.methods.nav_template import NavMethod
from vnav_benchmark.methods.ptgoal_sru import PtGoalSruMethod

GOLDEN = Path(__file__).resolve().parent / "golden" / "method_trace_ptgoal.json"


def _assert_json_matches(got: Any, want: Any, path: str = "replay") -> None:
    """Require identical JSON structure and values, allowing only tiny float drift."""
    assert type(got) is type(want), (path, type(got), type(want))
    if isinstance(want, dict):
        assert got.keys() == want.keys(), path
        for key in want:
            _assert_json_matches(got[key], want[key], f"{path}.{key}")
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for index, (actual, expected) in enumerate(zip(got, want)):
            _assert_json_matches(actual, expected, f"{path}[{index}]")
    elif isinstance(want, float):
        assert math.isfinite(got) and math.isfinite(want), path
        # Python/libm versions can differ by one ULP while replaying the same actions.
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12), (path, got, want)
    else:
        assert got == want, (path, got, want)


def _parse_recorded_files(files: Dict[str, str]) -> Dict[str, Any]:
    """Decode file contents so serialized floats receive the same strict comparison."""
    decoded = {}
    for filename, contents in files.items():
        if filename.endswith(".jsonl"):
            decoded[filename] = [json.loads(line) for line in contents.splitlines()]
        else:
            assert filename.endswith(".json"), filename
            decoded[filename] = json.loads(contents)
    return decoded


def test_ptgoal_replay_identical_to_pre_template() -> None:
    want = json.loads(GOLDEN.read_text())
    got = json.loads(json.dumps(trace.run_all(), sort_keys=True))
    for replay in (got, want):
        for scenario in replay.values():
            scenario["files"] = _parse_recorded_files(scenario["files"])
    _assert_json_matches(got, want)


def test_replay_comparison_accepts_only_tiny_nested_float_drift() -> None:
    _assert_json_matches(
        {"actions": [{"vx": 0.6500746017716331, "done": False}]},
        {"actions": [{"vx": 0.650074601771633, "done": False}]},
    )


@pytest.mark.parametrize("got,want", [
    (1.0 + 1e-9, 1.0),
    (True, 1),
    (2, 1),
    ("changed", "original"),
    ({"key": 1}, {"other": 1}),
    ([1], [1, 2]),
])
def test_replay_comparison_rejects_behavior_or_structure_changes(got: Any, want: Any) -> None:
    with pytest.raises(AssertionError):
        _assert_json_matches(got, want)


def test_ptgoal_uses_template() -> None:
    assert issubclass(PtGoalSruMethod, NavMethod)
