"""Test ImageGoalVlmPolicy clipping, NaN retries, done/error zeroing, odometry isolation and
restore.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytest
from vnav_fake_client import FakeVlmClient

from vnav_benchmark.interfaces import FatalInfraError, InfraError, Observation, UsageRecord
from vnav_benchmark.vlm import prompt
from vnav_benchmark.vlm.client import TextPart, VlmFatalError, VlmTransientError
from vnav_benchmark.vlm.policy import ImageGoalVlmPolicy, ProviderFatalError, RgbView
from vnav_benchmark.vlm.prompt import SubgoalView


class _Sink:
    def __init__(self) -> None:
        self.records: List[UsageRecord] = []

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    def episode_billed_usd(self) -> float:
        return sum(r.billed_usd for r in self.records)


class _Bomb:
    """Reject all attribute access to prove the policy ignores odometry, depth and remaining
    budget.
    """

    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"策略读了不该读的字段: {name}")

    def __float__(self) -> float:
        raise AssertionError("策略读了剩余预算")


def _obs(step: int) -> Observation:
    rgb = np.full((480, 640, 3), step % 250, dtype=np.uint8)
    return Observation(
        step_index=step,
        sim_time_s=float(step),
        budget_left_s=_Bomb(),  # type: ignore[arg-type]
        rgb=rgb,
        rgb_jpeg=b"\xff\xd8cur%d" % step,
        depth_m=_Bomb(),  # type: ignore[arg-type]
        K=(295.48, 0.0, 314.0, 0.0, 295.66, 242.41, 0.0, 0.0, 1.0),
        odom=_Bomb(),  # type: ignore[arg-type]
    )


def _reply(**over: Any) -> str:
    data = {
        "observation": "hallway",
        "target_visible": False,
        "subgoal_reached": False,
        "done": False,
        "memory": "explored hallway",
        "vx": 0.5,
        "vy": 0.0,
        "wz": 0.0,
    }
    data.update(over)
    return json.dumps(data)


def _policy(script: List[Any], tmp: Optional[Path] = None) -> ImageGoalVlmPolicy:
    return ImageGoalVlmPolicy(FakeVlmClient(script), _Sink(), b"\xff\xd8goal", tmp)


def test_valid_step_clamps_and_records_usage() -> None:
    pol = _policy([_reply(vx=3.0, wz=-5.0, vy=0.1)])
    step = pol.decide(_obs(0))
    assert step.status == "ok" and step.calls == 1
    assert (step.action.vx, step.action.vy, step.action.wz) == (0.8, 0.1, -0.7)
    assert step.action.hold_s == 1.0 and step.clamp.clamped_vx and step.clamp.clamped_wz
    assert len(pol._usage.records) == 1 and pol._usage.records[0].billed_usd > 0
    assert pol.memory.note == "explored hallway"
    assert pol.memory.log()[0].vx == 0.8  # Log velocities after clipping.


def test_nan_reply_reasks_once_then_ok() -> None:
    pol = _policy([_reply(vx=float("nan")), _reply()])
    step = pol.decide(_obs(0))
    assert step.status == "ok" and step.calls == 2 and pol.stats["reasks"] == 1
    reqs = pol._client.requests
    assert reqs[1].parts[:-1] == reqs[0].parts
    assert "must be a finite number" in reqs[1].parts[-1].text


def test_two_bad_replies_give_zero_velocity_and_feedback_next_step() -> None:
    pol = _policy(["garbage", "{}", _reply()])
    step = pol.decide(_obs(0))
    assert step.status == "invalid" and step.action.vx == 0.0 and step.action.wz == 0.0
    assert pol.memory.log()[0].status == "invalid"
    pol.decide(_obs(1))
    last_text = pol._client.requests[-1].parts[-1].text
    assert "FEEDBACK: " + prompt.FEEDBACK_PREV_INVALID in last_text


def test_done_zeroes_velocity() -> None:
    step = _policy([_reply(done=True, vx=0.6, wz=0.5)]).decide(_obs(0))
    assert step.action.done and (step.action.vx, step.action.vy, step.action.wz) == (0, 0, 0)


def test_call_failure_zero_velocity_and_consecutive_count() -> None:
    err = VlmTransientError("503 after retries")
    pol = _policy([err, err, _reply()])
    for t in range(2):
        step = pol.decide(_obs(t))
        assert step.status == "call_failed" and step.action.vx == 0.0
    assert pol.stats["consecutive_call_failed"] == 2
    pol.decide(_obs(2))
    assert pol.stats["consecutive_call_failed"] == 0
    assert prompt.FEEDBACK_PREV_CALL_FAILED in pol._client.requests[-1].parts[-1].text


def test_fatal_provider_error_is_infra() -> None:
    pol = _policy([VlmFatalError("insufficient_quota")])
    with pytest.raises(ProviderFatalError) as info:
        pol.decide(_obs(0))
    assert isinstance(info.value, FatalInfraError) and isinstance(info.value, InfraError)


def test_rgb_view_blocks_odom_depth_budget() -> None:
    view = RgbView(_obs(3))
    assert view.step_index == 3 and view.rgb.shape == (480, 640, 3)
    for name in ("odom", "depth_m", "budget_left_s", "sim_time_s", "frames"):
        with pytest.raises(AttributeError):
            getattr(view, name)


def test_subgoal_and_feedback_reach_request() -> None:
    pol = _policy([_reply()])
    pol.decide(_obs(0), SubgoalView(b"\xff\xd8kf", 1, 4), [prompt.FEEDBACK_SUBGOAL_NEW])
    req = pol._client.requests[0]
    assert any(isinstance(p, TextPart) and "subgoal 1 of 4" in p.text for p in req.parts)
    assert prompt.FEEDBACK_SUBGOAL_NEW in req.parts[-1].text


def test_history_grows_with_decisions() -> None:
    pol = _policy([_reply()] * 10)
    for t in range(10):
        pol.decide(_obs(t))
    counts = [
        sum(1 for p in r.parts if getattr(p, "width", 0) == 320) for r in pol._client.requests
    ]
    assert counts == [0, 1, 2, 2, 3, 3, 3, 3, 4, 4]


def test_state_roundtrip_gives_identical_next_request() -> None:
    a = _policy([_reply(observation=f"o{i}", memory=f"m{i}") for i in range(7)])
    for t in range(6):
        a.decide(_obs(t))
    b = _policy([_reply()])
    b.load_state(json.loads(json.dumps(a.to_state())))
    a.decide(_obs(6))
    b.decide(_obs(6))
    assert a._client.requests[-1] == b._client.requests[-1]


def test_load_state_rejects_other_fingerprint() -> None:
    a = _policy([])
    state = a.to_state()
    state["fingerprint"] = "sha256:other"
    with pytest.raises(ValueError):
        _policy([]).load_state(state)


def test_calls_log_and_images_on_disk(tmp_path: Path) -> None:
    pol = _policy(["bad", _reply(), _reply()], tmp_path)
    pol.decide(_obs(0), SubgoalView(b"\xff\xd8kf", 1, 2))
    pol.decide(_obs(1), SubgoalView(b"\xff\xd8kf", 1, 2))
    rows = [json.loads(x) for x in (tmp_path / "vlm" / "calls.jsonl").read_text().splitlines()]
    assert [(r["decision"], r["attempt"], r["status"]) for r in rows] == [
        (0, 0, "invalid"),
        (0, 1, "ok"),
        (1, 0, "ok"),
    ]
    assert rows[0]["fingerprint"] == pol.fingerprint and rows[1]["usage"]["billed_usd"] > 0
    shas = {p["img"] for r in rows for p in r["parts"] if "img" in p}
    on_disk = {p.stem for p in (tmp_path / "vlm" / "images").glob("*.jpg")}
    assert shas == on_disk  # Store each goal, subgoal, current and history image once.
