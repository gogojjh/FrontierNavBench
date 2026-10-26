"""Image-goal VLM methods M2-M6 within the seven-method benchmark.

M2/M3 use dense/sparse-contloc-imggoal with pose-based progression and arrival.
M4/M5 use dense/sparse-initloc-imggoal with VLM-based progression and arrival.
M6 uses nomap-noloc-imggoal and receives only the final goal image. All five share
byte-identical prompts, image handling and output fields.

INIT_LOC runs during reset: localize self/goal and plan the subgoal-image sequence.
Then NAVIGATE traverses subgoals, FINAL_APPROACH uses only the goal image, and
DONE ends the episode. M4/M5 advance when the VLM reports subgoal_reached.
M2/M3 obtain system-only poses each step and use PoseSubgoalFeed for progression
and deviation replanning; VLM subgoal_reached values are logged only. For their
pose-based arrival checks, the system declares done within 0.5 m / 30 degrees
without another VLM call. A VLM done report is counted but holds the robot still
and continues the episode. Other methods accept VLM arrival decisions. Poses
and coordinates are never exposed to the VLM. M6 starts in FINAL_APPROACH.
Initial ideal localization is assumed to succeed in v1.

The nav_template flow is subgoal (progression/system arrival), decide (one VLM
step), after_decide (accept completion/progression and prepare feedback), then action.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vnav_benchmark.interfaces import (
    EpisodeSpec,
    InfraError,
    MethodContext,
    Observation,
    UsageRecord,
    UsageSink,
)
from vnav_benchmark.methods.base_topo import (
    PHASE_DONE,
    InitLocProcedure,
    LocalizationFailed,
    PoseSubgoalFeed,
    Subgoal,
    TopoSubgoalFeed,
    localize_goal,
    localize_goal_result,
)
from vnav_benchmark.methods.nav_template import Decision, NavMethod, SubgoalInfo
from vnav_benchmark.vlm.client import ENDPOINTS, VlmClient, VlmClientLike, endpoint_info
from vnav_benchmark.vlm.policy import ImageGoalVlmPolicy, PolicyStep
from vnav_benchmark.vlm.prompt import (
    FEEDBACK_NOT_AT_GOAL,
    FEEDBACK_SUBGOAL_FINAL,
    FEEDBACK_SUBGOAL_NEW,
    HOLD_S,
    SubgoalView,
)

INFRA_CONSECUTIVE_FAILS = 3  # Consecutive API failures before InfraError; leave unscored and retry

DEFAULT_ENDPOINT = {"astra": "astra_flex", "qwen": "qwen"}  # Official evaluation endpoints
ALLOWED_ENDPOINTS = {"astra": ("astra_flex", "astra", "luna_dev"), "qwen": ("qwen",)}

INIT_LOC, NAVIGATE, FINAL_APPROACH, DONE = "INIT_LOC", "NAVIGATE", "FINAL_APPROACH", "DONE"

ClientFactory = Callable[[str], VlmClientLike]


class _CountingSink:
    """Forward usage to the framework UsageSink and accumulate episode totals for the summary."""

    def __init__(self, inner: UsageSink) -> None:
        self._inner = inner
        self.total: Dict[str, float] = {
            "calls": 0,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "list_usd": 0.0,
            "billed_usd": 0.0,
        }

    def add(self, record: UsageRecord) -> None:
        self._inner.add(record)
        t = self.total
        t["calls"] += 1
        for key in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
            t[key] += getattr(record, key)
        t["list_usd"] += record.list_usd
        t["billed_usd"] += record.billed_usd

    def episode_billed_usd(self) -> float:
        return self._inner.episode_billed_usd()


class ImgGoalVlmMethod(NavMethod):
    uses_truth = False
    LOG_ENSURE_ASCII = False  # Keep the existing vlm/steps.jsonl format
    SUMMARY_JSON_KW = {"ensure_ascii": False, "indent": 2}

    def __init__(
        self,
        method_id: str,
        client_factory: Optional[ClientFactory] = None,
        endpoint_override: Optional[str] = None,
    ) -> None:
        """endpoint_override selects luna_dev for development end-to-end runs; exclude results from
        tables.
        """
        self.method_id = method_id
        self._endpoint_override = endpoint_override
        self._client_factory: ClientFactory = client_factory or VlmClient
        self._phase = INIT_LOC
        self._feed: Optional[TopoSubgoalFeed] = None
        self._policy: Optional[ImageGoalVlmPolicy] = None
        self._dir: Optional[Path] = None
        self._feedback: List[str] = []
        self._pose_mode = False
        self._done_by_pose = False
        self._pose_feed: Optional[PoseSubgoalFeed] = None
        self._pose_sg: Optional[Subgoal] = None  # Current subgoal for pose-based progression
        self._localizer: Any = None
        self._summary: Dict[str, Any] = {}
        self._latency: List[Dict[str, float]] = []
        self._decision = 0  # Decision index captured in subgoal and shared by later hooks

    # ---- Hook 1: episode startup / INIT_LOC ----
    def setup(self, spec: EpisodeSpec, obs: Observation, ctx: MethodContext) -> None:
        profile = ctx.profile
        if profile.planner != "imggoal" or profile.vlm not in DEFAULT_ENDPOINT:
            raise ValueError(f"{self.method_id} 不是看图大模型方法")
        if profile.progress_judge not in ("vlm", "pose"):
            raise ValueError(f"{self.method_id} 的子目标判定 {profile.progress_judge!r} 不认识")
        self._pose_mode = profile.progress_judge == "pose"
        self._done_by_pose = profile.done_judge == "pose"
        self._localizer = ctx.localizer
        endpoint = self._endpoint_override or str(
            ctx.config.get("endpoint", DEFAULT_ENDPOINT[profile.vlm])
        )
        if endpoint not in ALLOWED_ENDPOINTS[profile.vlm]:
            raise ValueError(f"{self.method_id} 不能用端点 {endpoint!r}")
        self._dir = ctx.artifact_dir
        self.steps_log_path = Path(ctx.artifact_dir) / "vlm" / "steps.jsonl"
        self.summary_path = Path(ctx.artifact_dir) / "vlm" / "summary.json"
        self._sink = _CountingSink(ctx.usage)
        client = self._client_factory(endpoint)
        self._policy = ImageGoalVlmPolicy(client, self._sink, spec.goal_rgb_jpeg, ctx.artifact_dir)

        # v1 ideal localization has no fallback; an unexpected failure counts as a method failure.
        route_info: Dict[str, Any] = {"has_map": ctx.map is not None}
        if ctx.map is not None:
            if ctx.localizer is None:
                raise LocalizationFailed("有图方法没有定位器")
            initial = InitLocProcedure(ctx.localizer)
            start = initial.localize_start(obs)
            if self._pose_mode:
                goal_loc = localize_goal_result(ctx.localizer, spec)
                goal = goal_loc.pose_map
                assert goal is not None
                self._pose_feed = PoseSubgoalFeed(
                    ctx.map, start, goal, start_floor=initial.floor, goal_floor=goal_loc.floor
                )
                # Select the first subgoal using the initial pose
                self._pose_sg = self._pose_feed.current_subgoal_image(
                    start, obs.sim_time_s, floor=initial.floor
                )
                route = self._pose_feed.route
                assert route is not None
                self._phase = NAVIGATE
            else:
                goal = localize_goal(ctx.localizer, spec)
                self._feed = TopoSubgoalFeed.plan(ctx.map, start, goal)
                route = self._feed.route
                self._phase = NAVIGATE if not self._feed.finished else FINAL_APPROACH
            kf_ids = list(route.kf_path) if self._feed is None else list(self._feed.kf_ids)
            route_info.update(kf_ids=kf_ids, route_length_m=float(route.length_m))  # Subgoal
            # sequence
            self.configure_transit(ctx, route.kf_path, start, obs)
        else:
            self._phase = FINAL_APPROACH
        self._summary = {
            "method_id": self.method_id,
            "endpoint": endpoint_info(ENDPOINTS[endpoint]),
            "fingerprint": self._policy.fingerprint,
            "route": route_info,
            "subgoal_advances": 0,
            "progress_judge": profile.progress_judge,
            "replans": 0,
            "done_judge": profile.done_judge,
            "vlm_done_ignored": 0,
            "self_loc_failed": 0,
            "phase_changes": [{"decision": 0, "phase": self._phase}],
            "termination": None,
        }

    def transit_completed(self, consumed: Sequence[int]) -> None:
        """Resume the original pose or image sequence after the shared facility segment."""
        if self._pose_feed is not None:
            self._pose_feed.core.consume_prefix(consumed)
            self._pose_sg = None
        elif self._feed is not None:
            covered = set(consumed)
            advanced = False
            while self._feed.current() is not None and self._feed.current().kf_id in covered:
                self._feed.advance()
                advanced = True
            if advanced:
                self._set_phase(FINAL_APPROACH if self._feed.finished else NAVIGATE, self._decision)
                self._feedback.append(
                    FEEDBACK_SUBGOAL_FINAL if self._feed.finished else FEEDBACK_SUBGOAL_NEW
                )

    # ---- Per-step processing ----
    def _pose_progress(self, obs: Observation, decision: int) -> None:
        """M2/M3 obtain system-only poses to advance/replan subgoals and check final arrival."""
        if self._phase not in (NAVIGATE, FINAL_APPROACH) or self._pose_feed is None:
            return
        loc = self._localizer.localize(obs.rgb_jpeg, obs.depth_m, obs.K, purpose="self")
        if not loc.ok or loc.pose_map is None:
            self._summary["self_loc_failed"] += 1  # Keep the previous subgoal image if localization
            # is unavailable
            return
        prev = self._pose_sg
        sg = self._pose_feed.current_subgoal_image(loc.pose_map, obs.sim_time_s, floor=loc.floor)
        self._pose_sg = sg
        if self._phase == FINAL_APPROACH:
            return  # The final segment only checks arrival; act reads feed.phase
        if sg is None:
            if prev is not None:
                self._summary["subgoal_advances"] += 1
            self._feedback.append(FEEDBACK_SUBGOAL_FINAL)
            self._set_phase(FINAL_APPROACH, decision)
            return
        if prev is None or sg.kf_id == prev.kf_id:
            return
        if (
            sg.index > prev.index
        ):  # More visited keyframes means progression; every advance also replans, so version alone
        # cannot tell.
            self._summary["subgoal_advances"] += 1
            self._feedback.append(FEEDBACK_SUBGOAL_NEW)
        else:
            self._summary["replans"] += 1  # Reorder for deviation/stall by changing images only;
            # preserve the prompt text.

    def _subgoal_view(self) -> Optional[SubgoalView]:
        if self._pose_mode:
            if self._phase != NAVIGATE or self._pose_sg is None or self._pose_feed is None:
                return None
            sg = self._pose_sg
            core = self._pose_feed.core
            total = len(core.visited) + len(core.remaining)  # Visited plus pending keyframes
            return SubgoalView(sg.rgb_jpeg, sg.index + 1, total)
        if self._phase != NAVIGATE or self._feed is None:
            return None
        sg = self._feed.current()
        assert sg is not None
        return SubgoalView(sg.rgb_jpeg, sg.index + 1, len(self._feed.kf_ids))

    def _subgoal_kf(self) -> Optional[int]:
        """Current subgoal keyframe ID, for logs/video markers only; never supplied to the VLM."""
        if self._phase != NAVIGATE:
            return None
        if self._pose_mode:
            return None if self._pose_sg is None else int(self._pose_sg.kf_id)
        sg = self._feed.current() if self._feed is not None else None
        return None if sg is None else int(sg.kf_id)

    def _set_phase(self, phase: str, decision: int) -> None:
        if phase != self._phase:
            self._phase = phase
            self._summary["phase_changes"].append({"decision": decision, "phase": phase})

    def _advance_progress(self, step: PolicyStep, decision: int) -> None:
        """Advance only in NAVIGATE with a valid VLM reply; pose-driven methods do not use this
        path.
        """
        if self._pose_mode or self._phase != NAVIGATE or step.decision is None:
            return
        if self._feed is None:
            return
        if step.decision.subgoal_reached:  # Advance strictly in sequence, only when the VLM reports
        # arrival
            self._summary["subgoal_advances"] += 1
            if self._feed.advance() is None:
                self._feedback.append(FEEDBACK_SUBGOAL_FINAL)
                self._set_phase(FINAL_APPROACH, decision + 1)
            else:
                self._feedback.append(FEEDBACK_SUBGOAL_NEW)

    # ---- Hook 2: subgoal progression / system pose-based arrival ----
    def subgoal(self, obs: Observation) -> SubgoalInfo:
        assert self._policy is not None, "先 reset"
        if self._phase == DONE:  # If called after completion, remain still and continue reporting
        # done
            return SubgoalInfo(DONE, system_done=True, extra={"ended": True})
        self._decision = self._policy.stats["decisions"]
        if self._pose_mode:
            self._pose_progress(obs, self._decision)
        if self._done_by_pose and self._pose_feed is not None:
            if self._pose_feed.phase == PHASE_DONE:  # System pose-based arrival skips further VLM
            # calls
                return SubgoalInfo(DONE, system_done=True)
        view, kf = self._subgoal_view(), self._subgoal_kf()
        feedback, self._feedback = self._feedback, []
        return SubgoalInfo(
            self._phase,
            kf_id=kf,
            index=None if view is None else view.index,
            total=None if view is None else view.total,
            image_jpeg=None if view is None else view.jpeg,
            feedback=tuple(feedback),
            extra={"view": view},
        )

    def system_done_decision(self, obs: Observation, sub: SubgoalInfo) -> Decision:
        if sub.extra.get("ended"):
            return Decision(0.0, 0.0, 0.0, 1.0, done=True)
        self._set_phase(DONE, self._decision)
        self._summary["termination"] = "done_by_pose"
        return Decision(0.0, 0.0, 0.0, HOLD_S, done=True, note="pose: within tolerance",
                        debug={"phase": DONE, "status": "system_done"})  # fmt: skip

    # ---- Hook 3: one VLM decision ----
    def decide(self, obs: Observation, sub: SubgoalInfo) -> Decision:
        assert self._policy is not None
        step = self._policy.decide(obs, sub.extra["view"], list(sub.feedback))
        if self._policy.stats["consecutive_call_failed"] >= INFRA_CONSECUTIVE_FAILS:
            self._summary["termination"] = "infra_error"
            self.close()
            raise InfraError(f"连续 {INFRA_CONSECUTIVE_FAILS} 步大模型调用失败: {step.error}")
        a = step.action
        return Decision(a.vx, a.vy, a.wz, a.hold_s, done=a.done, note=a.note, raw=step)

    # ---- Hook 4: accept or ignore VLM done/subgoal_reached ----
    def after_decide(self, obs: Observation, sub: SubgoalInfo, dec: Decision) -> Decision:
        step: PolicyStep = dec.raw
        self._advance_progress(step, self._decision)
        if step.action.done and self._done_by_pose and self._pose_feed is not None:
            self._summary["vlm_done_ignored"] += 1  # VLM reports done but pose-based arrival
            # disagrees: hold still and continue.
            self._feedback.append(FEEDBACK_NOT_AT_GOAL)  # Tell the model on the next step that
            # arrival was rejected.
            step = dataclasses.replace(step, action=dataclasses.replace(step.action, done=False))
        elif step.action.done:
            self._set_phase(DONE, self._decision)
            self._summary["termination"] = "done"
        self._latency.append(step.latency)
        view: Optional[SubgoalView] = sub.extra["view"]
        sub_pair = None if view is None else [view.index, view.total]
        debug = {
            "latency": step.latency,
            "phase": sub.phase,
            "subgoal": sub_pair,
            "subgoal_kf": sub.kf_id,
            "status": step.status,
        }
        a = step.action
        return Decision(a.vx, a.vy, a.wz, a.hold_s, done=a.done, note=a.note, debug=debug,
                        log=self._log_row(self._decision, sub.phase, sub_pair, step, sub.kf_id),
                        raw=step)  # fmt: skip

    def _log_row(
        self,
        decision: int,
        phase: str,
        sub_pair: Optional[List[int]],
        step: PolicyStep,
        subgoal_kf: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """One vlm/steps.jsonl row, preserving the existing fields."""
        if self._dir is None:
            return None
        d = step.decision
        return {
            "decision": decision,
            "phase": phase,
            "subgoal": sub_pair,
            "subgoal_kf": subgoal_kf,
            "status": step.status,
            "error": step.error,
            "calls": step.calls,
            "reply": None
            if d is None
            else {
                "observation": d.observation,
                "target_visible": d.target_visible,
                "subgoal_reached": d.subgoal_reached,
                "done": d.done,
                "memory": d.memory,
                "vx": d.vx,
                "vy": d.vy,
                "wz": d.wz,
            },
            "cmd": [step.action.vx, step.action.vy, step.action.wz],
            "hold_s": step.action.hold_s,
            "done": step.action.done,
            "clamped": step.clamp.any,
            "latency": step.latency,
        }

    # ---- Episode completion ----
    def summary(self) -> Dict[str, Any]:
        out = dict(self._summary)
        if self._policy is not None:
            out["policy_stats"] = dict(self._policy.stats)
            out["memory_truncations"] = {
                "observation": self._policy.memory.obs_truncations,
                "memory": self._policy.memory.note_truncations,
            }
            out["usage"] = dict(self._sink.total)
            inp = self._sink.total["input_tokens"]
            out["cache_hit_ratio"] = self._sink.total["cached_tokens"] / inp if inp else 0.0
        out["latency"] = {
            key: _percentiles([lat[key] for lat in self._latency if key in lat])
            for key in ("vlm_api_s", "server_ms")
        }
        out["final_phase"] = self._phase
        out.update(self.transit_summary())
        return out

    # ---- Save / restore ----
    def state_dict(self) -> Dict[str, Any]:
        assert self._policy is not None
        state = {
            "phase": self._phase,
            "feedback": list(self._feedback),
            "feed": None if self._feed is None else self._feed.state_dict(),
            "policy": self._policy.to_state(),
            "summary": self._summary,
        }
        if self._transit_driver is not None:
            state["elevator_transit"] = self._transit_driver.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        assert self._policy is not None, "先 reset 再恢复"
        if (state["feed"] is None) != (self._feed is None):
            raise ValueError("存档和当前回合的子目标序列有无不一致")
        if self._feed is not None:
            self._feed.load_state_dict(state["feed"])
        self._phase = state["phase"]
        self._feedback = list(state["feedback"])
        self._policy.load_state(state["policy"])
        self._summary = state["summary"]
        if "elevator_transit" in state:
            if self._transit_driver is None:
                raise ValueError("存档有乘梯候选，但当前地图无法重新识别")
            self._transit_driver.load_state_dict(state["elevator_transit"])


def _percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    """Return step-latency median, p95 and maximum, or None when no samples exist."""
    if not values:
        return {"median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def make(method_id: str) -> ImgGoalVlmMethod:
    return ImgGoalVlmMethod(method_id)
