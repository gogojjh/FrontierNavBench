"""VLM memory tests for history sampling, truncation, summary feedback, restore and output parsing.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from vnav_benchmark import fairness
from vnav_benchmark.vlm import schema
from vnav_benchmark.vlm.memory import HISTORY_SIZE, LOG_STEPS, VlmMemory, shrink_rgb


def _rgb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (480, 640, 3), dtype=np.uint8)


def _valid(**over: object) -> str:
    data = {
        "observation": "door ahead",
        "target_visible": False,
        "subgoal_reached": False,
        "done": False,
        "memory": "checked corridor",
        "vx": 0.5,
        "vy": 0.0,
        "wz": -0.2,
    }
    data.update(over)
    return json.dumps(data)


def test_history_counts_for_first_ten_decisions() -> None:
    mem = VlmMemory()
    counts = []
    for t in range(10):
        counts.append(len(mem.history(t)))
        mem.add_frame(t, _rgb(t))
    assert counts == [0, 1, 2, 2, 3, 3, 3, 3, 4, 4]


def test_history_offsets_oldest_first_and_right_frames() -> None:
    mem = VlmMemory()
    for t in range(12):
        mem.add_frame(t, _rgb(t))
    hist = mem.history(12)
    assert [k for k, _ in hist] == [8, 4, 2, 1]
    assert hist[0][1] == shrink_rgb(_rgb(4))
    assert hist[-1][1] == shrink_rgb(_rgb(11))
    assert len(mem._frames) <= 8  # Retain only the latest eight steps


def test_shrink_is_deterministic_and_320x240() -> None:
    a = shrink_rgb(_rgb(3))
    assert a == shrink_rgb(_rgb(3))
    img = cv2.imdecode(np.frombuffer(a, np.uint8), cv2.IMREAD_COLOR)
    assert (img.shape[1], img.shape[0]) == HISTORY_SIZE


def test_log_keeps_last_15_and_truncates_observation() -> None:
    mem = VlmMemory()
    for d in range(20):
        mem.record(d, 0.5, 0.0, 0.1, " ".join(f"w{i}" for i in range(20)))
    log = mem.log()
    assert len(log) == LOG_STEPS
    assert log[0].decision == 5
    assert len(log[0].observation.split()) == schema.OBS_MAX_WORDS
    assert mem.obs_truncations == 20


def test_note_roundtrip_and_truncation() -> None:
    mem = VlmMemory()
    mem.set_note("  kitchen   is east  ")
    assert mem.note == "kitchen is east"
    mem.set_note(" ".join(["x"] * 60))
    assert len(mem.note.split()) == schema.MEMORY_MAX_WORDS
    assert mem.note_truncations == 1


def test_state_roundtrip_is_json_and_equal() -> None:
    mem = VlmMemory()
    for t in range(9):
        mem.add_frame(t, _rgb(t))
        mem.record(t, 0.1 * t, 0.0, 0.0, f"obs {t}", "ok" if t % 3 else "invalid")
    mem.set_note("plan")
    state = json.loads(json.dumps(mem.to_state()))
    back = VlmMemory.from_state(state)
    assert back.history(9) == mem.history(9)
    assert back.log() == mem.log()
    assert back.to_state() == mem.to_state()


def test_schema_field_order_and_ranges_from_fairness() -> None:
    props = schema.ACTION_SCHEMA["properties"]
    assert tuple(props) == schema.FIELD_ORDER
    assert schema.ACTION_SCHEMA["required"] == list(schema.FIELD_ORDER)
    assert (props["vx"]["minimum"], props["vx"]["maximum"]) == (fairness.VX_MIN, fairness.VX_MAX)
    assert props["vy"]["maximum"] == fairness.VY_MAX and props["vy"]["minimum"] == -fairness.VY_MAX
    assert props["wz"]["maximum"] == fairness.WZ_MAX and props["wz"]["minimum"] == -fairness.WZ_MAX


def test_parse_valid_and_fenced() -> None:
    d, err = schema.parse_decision(_valid())
    assert err == "" and d is not None and d.vx == 0.5 and d.wz == -0.2
    d2, _ = schema.parse_decision("```json\n" + _valid(vx=1) + "\n```")
    assert d2 is not None and d2.vx == 1.0


def test_parse_out_of_range_is_accepted_for_clamping() -> None:
    d, err = schema.parse_decision(_valid(vx=3.0, wz=-9))
    assert err == "" and d is not None and d.vx == 3.0


def test_parse_rejects_bad_outputs() -> None:
    bad = [
        "not json",
        "[1, 2]",
        json.dumps({"observation": "x"}),
        _valid(done="yes"),
        _valid(vx=True),
        _valid(memory=5),
        _valid(vx="0.5"),
        _valid().replace("0.5", "NaN"),
    ]
    for text in bad:
        d, err = schema.parse_decision(text)
        assert d is None and err, text


def test_parse_ignores_extra_fields() -> None:
    d, err = schema.parse_decision(_valid(note="extra"))
    assert d is not None and err == ""
