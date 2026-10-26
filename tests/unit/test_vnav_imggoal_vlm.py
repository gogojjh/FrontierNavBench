"""VLM method state-machine tests with fake model and simulation over complete episodes."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import pytest
from vnav_fake_client import FakeVlmClient
from vnav_t_fixtures import CAMERA_FORWARD_M, FakeWorld, TruthLocalizer, make_map

from frontier_nav_bench import fairness
from frontier_nav_bench import mapview as mv
from frontier_nav_bench.interfaces import (
    EpisodeSpec,
    InfraError,
    LocResult,
    MethodContext,
    Observation,
    Pose2D,
    UsageRecord,
)
from frontier_nav_bench.methods import imggoal_vlm as iv
from frontier_nav_bench.methods import make_method
from frontier_nav_bench.methods.base_topo import LocalizationFailed
from frontier_nav_bench.vlm import prompt
from frontier_nav_bench.vlm.client import ImagePart, TextPart, VlmRequest, VlmTransientError

GOAL_JPEG = b"\xff\xd8goal-image"
START = (0.0, 0.0, 0.0)
GOAL = (4.0, 4.0, math.pi / 2)


class _Sink:
    def __init__(self) -> None:
        self.records: List[UsageRecord] = []

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    def episode_billed_usd(self) -> float:
        return sum(r.billed_usd for r in self.records)


class _Bomb:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"方法读了里程计: {name}")


def _reply(**over: Any) -> str:
    data = {
        "observation": "x",
        "target_visible": True,
        "subgoal_reached": False,
        "done": False,
        "memory": "m",
        "vx": 0.3,
        "vy": 0.0,
        "wz": 0.0,
    }
    data.update(over)
    return json.dumps(data)


def _subgoal_label(req: VlmRequest) -> Optional[str]:
    for p in req.parts:
        if isinstance(p, TextPart) and p.text.startswith("CURRENT SUBGOAL IMAGE"):
            return p.text
    return None


def _scripted() -> Callable[[VlmRequest], str]:
    """Report subgoal arrival every two steps, or done on the second step without a subgoal."""
    state = {"since": 0}

    def reply(req: VlmRequest) -> str:
        state["since"] += 1
        if _subgoal_label(req) is not None:
            return _reply(subgoal_reached=state["since"] % 2 == 0)
        return _reply(done=state["since"] >= 2, vx=0.2)

    return reply


class _Episode:
    def __init__(
        self,
        tmp: Path,
        method_id: str,
        script: List[Any],
        localizer_ok: bool = True,
        config: Optional[dict] = None,
    ) -> None:
        profile = fairness.parse_method_id(method_id)
        self.world = FakeWorld(START)
        self.fake: Optional[FakeVlmClient] = None
        self.endpoints: List[str] = []

        def factory(endpoint: str) -> FakeVlmClient:
            self.endpoints.append(endpoint)
            self.fake = FakeVlmClient(script, endpoint)
            return self.fake

        handle = None
        self.loc: Any = None
        if profile.map_variant is not None:
            src = make_map(tmp / "src")
            mv.build_sanitized_view(src, tmp / "view")
            handle = mv.SanitizedMap(
                tmp / "view", "m", profile.map_variant, "trav", CAMERA_FORWARD_M
            )
            inner = TruthLocalizer(self.world, GOAL)
            if not localizer_ok:
                inner.localize = lambda *a, purpose: LocResult(False, None, None, "lost")  # type:
                # ignore
            self.inner_loc = inner
            self.loc = fairness.QuotaLocalizer(inner, profile)
        self.sink = _Sink()
        self.ctx = MethodContext(
            profile=profile,
            config=config or {},
            map=handle,
            localizer=self.loc,
            rng=np.random.default_rng(0),
            log=logging.getLogger("t"),
            usage=self.sink,
            artifact_dir=tmp / "method",
        )
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        self.spec = EpisodeSpec(
            "tok", 0, "env0", img, GOAL_JPEG, np.zeros((48, 64), np.float32), (1.0,), 180.0
        )
        self.method = iv.ImgGoalVlmMethod(method_id, client_factory=factory)

    def obs(self) -> Observation:
        return dataclasses.replace(self.world.observation(), odom=_Bomb())

    def run(self, max_steps: int = 60) -> int:
        self.method.reset(self.spec, self.world.observation(), self.ctx)
        for t in range(max_steps):
            action = self.method.act(self.obs())
            self.world.step(action.vx, action.vy, action.wz, action.hold_s)
            if action.done:
                self.method.close()
                return t + 1
        self.method.close()
        return max_steps

    def rows(self, name: str) -> List[dict]:
        path = self.ctx.artifact_dir / "vlm" / name
        return [json.loads(x) for x in path.read_text().splitlines()]


def test_m2_full_episode_walks_subgoals_then_done(tmp_path: Path) -> None:
    ep = _Episode(tmp_path, "dense-initloc-imggoal-astra", [_scripted()] * 200)
    steps = ep.run()
    assert ep.endpoints == ["astra_flex"]
    assert ep.inner_loc.calls == ["self", "goal"]
    s = json.loads((tmp_path / "method" / "vlm" / "summary.json").read_text())
    n = len(s["route"]["kf_ids"])
    assert n >= 2
    assert s["subgoal_advances"] == n and s["termination"] == "done"
    assert [p["phase"] for p in s["phase_changes"]] == ["NAVIGATE", "FINAL_APPROACH", "DONE"]
    labels = [_subgoal_label(r) for r in ep.fake.requests]
    assert labels[0] == f"CURRENT SUBGOAL IMAGE (subgoal 1 of {n}, from the earlier mapping run):"
    assert labels[2].startswith(f"CURRENT SUBGOAL IMAGE (subgoal 2 of {n}")
    assert labels[-1] is None  # Final approach includes only the goal image
    # Subgoals use map keyframe images; FEEDBACK reports progression and final approach
    imgs = [p.jpeg for p in ep.fake.requests[0].parts if isinstance(p, ImagePart)]
    assert imgs[0] == GOAL_JPEG and imgs[1] == ep.ctx.map.keyframe_rgb_jpeg(s["route"]["kf_ids"][0])
    assert prompt.FEEDBACK_SUBGOAL_NEW in ep.fake.requests[2].parts[-1].text
    assert any(prompt.FEEDBACK_SUBGOAL_FINAL in r.parts[-1].text for r in ep.fake.requests)
    rows = ep.rows("steps.jsonl")
    assert len(rows) == steps
    need = {"decision", "phase", "subgoal", "subgoal_kf", "status", "reply", "cmd", "hold_s",
            "done", "latency"}  # fmt: skip
    kf_ids = s["route"]["kf_ids"]
    assert rows[0]["subgoal_kf"] == kf_ids[0] and rows[2]["subgoal_kf"] == kf_ids[1]
    assert all(need <= set(r) for r in rows)
    assert len(ep.sink.records) == len(ep.fake.requests)
    assert s["usage"]["calls"] == len(ep.fake.requests) and s["usage"]["billed_usd"] > 0
    assert s["fingerprint"] == prompt.prompt_fingerprint()
    assert s["endpoint"]["endpoint"] == "astra_flex"
    assert s["latency"]["vlm_api_s"]["median"] is not None


def test_m4_nomap_never_gets_subgoal(tmp_path: Path) -> None:
    ep = _Episode(tmp_path, "nomap-noloc-imggoal-qwen", [_scripted()] * 200)
    ep.run()
    assert ep.endpoints == ["qwen"] and ep.ctx.localizer is None
    assert all(_subgoal_label(r) is None for r in ep.fake.requests)
    s = ep.method.summary()
    assert s["route"]["has_map"] is False
    assert s["phase_changes"][0]["phase"] == "FINAL_APPROACH" and s["final_phase"] == "DONE"


def test_initloc_keeps_subgoal_until_vlm_says_reached(tmp_path: Path) -> None:
    # Keep strict sequence despite invisible subgoals; target_visible is logged only.
    script = [_reply(target_visible=False)] * 14 + [_reply(subgoal_reached=True)]
    ep = _Episode(tmp_path, "sparse-initloc-imggoal-astra", script)
    ep.run(max_steps=15)
    labels = [_subgoal_label(r) for r in ep.fake.requests]
    assert len(set(labels[:15])) == 1 and labels[0] is not None
    s = ep.method.summary()
    assert s["subgoal_advances"] == 1 and "lost_fallback" not in s
    texts = " ".join(p.text for r in ep.fake.requests for p in r.parts if hasattr(p, "text"))
    assert "not seen for many decisions" not in texts


def test_init_loc_failure_is_method_error(tmp_path: Path) -> None:
    # Initial localization failure has no fallback and counts as a method failure.
    ep = _Episode(tmp_path, "dense-initloc-imggoal-astra", [_reply()], localizer_ok=False)
    with pytest.raises(LocalizationFailed):
        ep.method.reset(ep.spec, ep.world.observation(), ep.ctx)


def test_three_consecutive_call_failures_raise_infra(tmp_path: Path) -> None:
    err = VlmTransientError("boom")
    ep = _Episode(tmp_path, "nomap-noloc-imggoal-astra", [_reply(), err, err, err])
    with pytest.raises(InfraError):
        ep.run()
    s = json.loads((tmp_path / "method" / "vlm" / "summary.json").read_text())
    assert s["termination"] == "infra_error"


def test_two_failures_then_ok_keeps_going(tmp_path: Path) -> None:
    err = VlmTransientError("boom")
    ep = _Episode(tmp_path, "nomap-noloc-imggoal-astra", [err, err, _reply(), _reply(done=True)])
    assert ep.run() == 4


def test_endpoint_rules(tmp_path: Path) -> None:
    ep = _Episode(tmp_path, "nomap-noloc-imggoal-astra", [], config={"endpoint": "qwen"})
    with pytest.raises(ValueError):
        ep.method.reset(ep.spec, ep.world.observation(), ep.ctx)
    ep2 = _Episode(tmp_path / "b", "nomap-noloc-imggoal-astra", [], config={"endpoint": "luna_dev"})
    ep2.method.reset(ep2.spec, ep2.world.observation(), ep2.ctx)
    assert ep2.endpoints == ["luna_dev"]


def _pose_episode(tmp_path: Path, script: List[Any]) -> _Episode:
    ep = _Episode(tmp_path, "dense-contloc-imggoal-astra", script)
    assert ep.ctx.profile.progress_judge == "pose" and ep.ctx.profile.done_judge == "pose"
    return ep


def _run_teleport(ep: _Episode, poses: List[tuple]) -> List[Any]:
    """Teleport the fake robot to each supplied pose and return the resulting Actions."""
    ep.method.reset(ep.spec, ep.world.observation(), ep.ctx)
    actions = []
    for step, pose in enumerate(poses):
        ep.world.pose = Pose2D(*pose)
        ep.world.step_index = step
        ep.loc.on_frame(step)  # Match runner by granting one localization call per frame
        act = ep.method.act(ep.obs())
        actions.append(act)
        if act.done:
            break
    ep.method.close()
    return actions


def test_pose_mode_advances_by_pose_and_declares_done(tmp_path: Path) -> None:
    # Ignore/count VLM subgoal/done reports for pose-driven progression and arrival.
    ep = _pose_episode(tmp_path, [_reply(subgoal_reached=True, done=True)] * 40)
    path = [(0.5, 0.0, 0.0), (1.2, 0.0, 0.0), (2.2, 0.0, 0.0), (3.2, 0.0, 0.0), (4.0, 0.3, 1.57),
            (4.0, 1.5, 1.57), (4.0, 2.5, 1.57), (4.0, 3.4, 1.57), (4.0, 3.9, 1.57),
            (4.0, 4.0, math.pi / 2), (4.0, 4.0, math.pi / 2)]  # fmt: skip
    actions = _run_teleport(ep, path)
    s = ep.method.summary()
    assert actions[-1].done and actions[-1].debug["status"] == "system_done"
    assert all(not a.done for a in actions[:-1])  # The VLM done report did not end the episode
    assert s["termination"] == "done_by_pose" and s["done_judge"] == "pose"
    assert s["vlm_done_ignored"] == len(ep.fake.requests) == len(actions) - 1
    assert s["subgoal_advances"] >= 2
    assert all(a.vx == 0.0 and a.wz == 0.0 for a in actions[:-1])  # A rejected VLM done decision
    # holds the robot still
    labels = [_subgoal_label(r) for r in ep.fake.requests]
    idx = [int(re.search(r"subgoal (\d+) of", lb).group(1)) for lb in labels if lb]
    assert idx == sorted(idx) and idx[-1] > idx[0]  # Monotonic pose-based progression
    assert labels[-1] is None  # Final approach includes only the goal image
    assert prompt.FEEDBACK_SUBGOAL_NEW in " ".join(r.parts[-1].text for r in ep.fake.requests)
    assert ep.inner_loc.calls.count("self") == 1 + len(actions)  # One startup call plus one per
    # step


def test_pose_mode_ignores_vlm_subgoal_claims(tmp_path: Path) -> None:
    # Identical poses yield identical subgoal images/progression regardless of VLM arrival reports.
    runs = []
    for claim in (True, False):
        ep = _pose_episode(tmp_path / str(claim), [_reply(subgoal_reached=claim)] * 5)
        actions = _run_teleport(ep, [(0.0, 0.0, 0.0)] * 5)  # Keep the robot at the start
        labels = [_subgoal_label(r) for r in ep.fake.requests]
        runs.append((labels, ep.method.summary()["subgoal_advances"]))
        assert not any(a.done for a in actions) and labels[-1] is not None
    assert runs[0] == runs[1]
    assert len(set(runs[0][0][1:])) == 1  # Without motion, the subgoal stops changing after the
    # first step


def test_pose_mode_prompt_identical_to_initloc(tmp_path: Path) -> None:
    a = _pose_episode(tmp_path / "a", [_reply()])
    _run_teleport(a, [(0.0, 0.0, 0.0)])
    b = _Episode(tmp_path / "b", "dense-initloc-imggoal-astra", [_reply()])
    b.method.reset(b.spec, b.world.observation(), b.ctx)
    b.method.act(b.obs())
    ra, rb = a.fake.requests[0], b.fake.requests[0]
    assert ra.system == rb.system and ra.schema == rb.schema
    assert [type(p).__name__ for p in ra.parts] == [type(p).__name__ for p in rb.parts]


def test_registry_builds_all_six() -> None:
    ids = [m for m in fairness.V1_METHOD_IDS if "-imggoal-" in m]
    for mid in ids:
        method = make_method(mid)
        assert isinstance(method, iv.ImgGoalVlmMethod) and method.method_id == mid


def test_state_roundtrip_mid_episode(tmp_path: Path) -> None:
    script = [_reply(subgoal_reached=i % 2 == 1, observation=f"o{i}") for i in range(8)]
    a = _Episode(tmp_path / "a", "dense-initloc-imggoal-astra", script)
    a.method.reset(a.spec, a.world.observation(), a.ctx)
    for _ in range(5):
        act = a.method.act(a.obs())
        a.world.step(act.vx, act.vy, act.wz, act.hold_s)
    b = _Episode(tmp_path / "b", "dense-initloc-imggoal-astra", script[5:])
    b.method.reset(b.spec, b.world.observation(), b.ctx)
    b.method.load_state_dict(json.loads(json.dumps(a.method.state_dict())))
    b.world = a.world
    a.method.act(a.obs())
    b.method.act(b.obs())
    assert a.fake.requests[-1] == b.fake.requests[-1]


def test_no_budget_or_numbers_from_odom_in_requests(tmp_path: Path) -> None:
    ep = _Episode(tmp_path, "dense-initloc-imggoal-astra", [_scripted()] * 200)
    ep.run()
    for req in ep.fake.requests:
        text = " ".join(p.text for p in req.parts if isinstance(p, TextPart))
        assert "budget" not in text.lower() and not re.search(r"\b180\b", text)


def test_endpoint_override_for_dev_runs(tmp_path: Path) -> None:
    ep = _Episode(tmp_path, "dense-initloc-imggoal-astra", [])
    ep.method = iv.ImgGoalVlmMethod(
        "dense-initloc-imggoal-astra",
        client_factory=lambda e: (ep.endpoints.append(e), FakeVlmClient([], e))[1],
        endpoint_override="luna_dev",
    )
    ep.method.reset(ep.spec, ep.world.observation(), ep.ctx)
    assert ep.endpoints == ["luna_dev"]
    with pytest.raises(ValueError):
        bad = iv.ImgGoalVlmMethod("nomap-noloc-imggoal-qwen", endpoint_override="luna_dev")
        bad.reset(
            ep.spec,
            ep.world.observation(),
            dataclasses.replace(
                ep.ctx,
                profile=fairness.parse_method_id("nomap-noloc-imggoal-qwen"),
                map=None,
                localizer=None,
            ),
        )


def test_pose_done_method_tells_vlm_not_at_goal_yet(tmp_path: Path) -> None:
    # M2/M3 pose-based arrival rejection sends feedback on the next step.
    ep = _pose_episode(tmp_path, [_reply(done=True), _reply(), _reply(done=True), _reply()])
    _run_teleport(ep, [(0.0, 0.0, 0.0)] * 4)  # Remain at the start, far from the goal
    last = [r.parts[-1].text for r in ep.fake.requests]
    assert prompt.FEEDBACK_NOT_AT_GOAL not in last[0]
    assert prompt.FEEDBACK_NOT_AT_GOAL in last[1] and prompt.FEEDBACK_NOT_AT_GOAL not in last[2]
    assert prompt.FEEDBACK_NOT_AT_GOAL in last[3]
    assert ep.method.summary()["vlm_done_ignored"] == 2


def test_vlm_done_methods_never_get_not_at_goal_line(tmp_path: Path) -> None:
    ep = _Episode(tmp_path, "dense-initloc-imggoal-astra", [_reply()] * 3 + [_reply(done=True)])
    ep.run(max_steps=4)
    texts = " ".join(r.parts[-1].text for r in ep.fake.requests)
    assert prompt.FEEDBACK_NOT_AT_GOAL not in texts
