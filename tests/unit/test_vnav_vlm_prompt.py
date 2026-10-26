"""Test full prompts, request assembly, byte-stable cache prefixes and pinned fingerprints."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Sequence

from vnav_benchmark import fairness
from vnav_benchmark.vlm import prompt
from vnav_benchmark.vlm.client import (
    ENDPOINTS,
    ImagePart,
    TextPart,
    build_chat_body,
    build_responses_body,
)
from vnav_benchmark.vlm.memory import LogEntry
from vnav_benchmark.vlm.pricing import estimate_text_tokens
from vnav_benchmark.vlm.prompt import StepInputs, SubgoalView, build_request

GOAL = b"\xff\xd8goal"
SUB = b"\xff\xd8sub1"


def _inputs(decision: int, subgoal: bool, feedback: Sequence[str] = ()) -> StepInputs:
    hist = [(k, b"\xff\xd8h%d" % (decision - k)) for k in (8, 4, 2, 1) if decision - k >= 0]
    log = [LogEntry(d, 0.5, 0.0, -0.7, f"obs {d}") for d in range(max(0, decision - 15), decision)]
    return StepInputs(
        decision=decision,
        goal_jpeg=GOAL,
        current_jpeg=b"\xff\xd8cur%d" % decision,
        subgoal=SubgoalView(SUB, 2, 7) if subgoal else None,
        history=hist,
        log=log,
        memory_note=f"note {decision}" if decision else "",
        feedback=list(feedback),
    )


def _cached_prefix_responses(body: Dict[str, Any]) -> str:
    """Serialize the developer block and user content through the last cache breakpoint."""
    user = body["input"][1]["content"]
    last = max(i for i, p in enumerate(user) if "prompt_cache_breakpoint" in p)
    return json.dumps([body["input"][0], user[: last + 1]], ensure_ascii=False)


def test_system_prompt_rendered_and_sized() -> None:
    sp = prompt.SYSTEM_PROMPT
    assert "{" not in sp.replace('{"observation"', "") and "}" not in sp.replace(
        '"wz": number}', ""
    )
    assert f"from {fairness.VX_MIN:.1f} to {fairness.VX_MAX:.1f}" in sp
    assert f"from -{fairness.WZ_MAX:.1f} to {fairness.WZ_MAX:.1f}" in sp
    assert "d-8 ... d-1" in sp and "8, 4, 2 and 1 decisions" in sp
    # System prompt plus the 360-token goal image must reach the 1,024-token cache threshold.
    assert 800 <= estimate_text_tokens(sp) <= 1400


def test_system_prompt_has_no_budget_or_pose_leak_words() -> None:
    text = prompt.SYSTEM_PROMPT.lower() + prompt.step_text(_inputs(20, True)).lower()
    for word in ("remaining time", "budget", "odom", "seconds left", "coordinates"):
        assert word not in text


def test_same_system_prompt_and_schema_for_all_imggoal_methods() -> None:
    ids = [m for m in fairness.V1_METHOD_IDS if "imggoal" in m]
    assert len(ids) == 10
    reqs = [
        build_request(_inputs(3, fairness.parse_method_id(m).map_variant is not None)) for m in ids
    ]
    assert len({r.system for r in reqs}) == 1
    assert len({json.dumps(r.schema) for r in reqs}) == 1


def test_part_order_with_and_without_subgoal() -> None:
    def kinds(req: Any) -> str:
        return "".join("I" if isinstance(p, ImagePart) else "T" for p in req.parts)

    m4 = build_request(_inputs(9, subgoal=False))
    m2 = build_request(_inputs(9, subgoal=True))
    # Goal(T I T), optional subgoal(T I T), history(T + 4*(T I)), current(T I), per-step text(T).
    assert kinds(m4) == "TIT" + "T" + "TI" * 4 + "TI" + "T"
    assert kinds(m2) == "TIT" + "TIT" + "T" + "TI" * 4 + "TI" + "T"
    assert [p.cache_breakpoint for p in m2.parts if isinstance(p, TextPart)].count(True) == 2
    assert m2.parts[1].jpeg == GOAL and m2.parts[4].jpeg == SUB
    sizes = [(p.width, p.height) for p in m2.parts if isinstance(p, ImagePart)]
    assert sizes == [(640, 480), (640, 480)] + [(320, 240)] * 4 + [(640, 480)]


def test_first_decision_has_no_history() -> None:
    req = build_request(_inputs(0, subgoal=False))
    texts = [p.text for p in req.parts if isinstance(p, TextPart)]
    assert prompt.PAST_NONE in texts
    assert "RECENT LOG: none yet." in texts[-1] and "MEMORY: none yet." in texts[-1]


def test_cached_prefix_byte_identical_across_steps_responses() -> None:
    for subgoal in (False, True):
        prefixes = {
            _cached_prefix_responses(
                build_responses_body(ENDPOINTS["astra_flex"], build_request(_inputs(d, subgoal)))
            )
            for d in range(0, 30)
        }
        assert len(prefixes) == 1


def test_chat_prefix_byte_identical_across_steps() -> None:
    def head(d: int) -> str:
        body = build_chat_body(ENDPOINTS["qwen"], build_request(_inputs(d, True)))
        return json.dumps([body["messages"][0], body["messages"][1]["content"][:6]])

    assert len({head(d) for d in range(30)}) == 1


def test_step_text_log_format_and_feedback() -> None:
    inp = _inputs(20, True, feedback=[prompt.FEEDBACK_SUBGOAL_NEW])
    inp = StepInputs(
        **{
            **inp.__dict__,
            "log": list(inp.log[:-1]) + [LogEntry(19, 0.0, -0.0, 0.0, "", "invalid")],
        }
    )
    text = prompt.step_text(inp)
    lines = text.splitlines()
    assert lines[0] == "DECISION 20"
    assert lines[2].startswith("d-15: vx=0.50 vy=0.00 wz=-0.70 | obs: obs 5")
    assert "d-1: vx=0.00 vy=0.00 wz=0.00 | (reply could not be used" in text
    assert lines[-2] == "FEEDBACK: " + prompt.FEEDBACK_SUBGOAL_NEW
    assert lines[-1] == prompt.REPLY_LINE
    assert len([ln for ln in lines if re.match(r"d-\d+:", ln)]) == 15


def test_reask_appends_one_line_and_keeps_prefix() -> None:
    a = build_request(_inputs(5, True))
    b = build_request(_inputs(5, True), reask_error="missing fields: vx")
    assert b.parts[:-1] == a.parts
    assert b.parts[-1].text.startswith("Your reply above could not be used (missing fields: vx)")


def test_fingerprint_pinned() -> None:
    assert prompt.prompt_fingerprint() == prompt.PROMPT_FINGERPRINT_V1
