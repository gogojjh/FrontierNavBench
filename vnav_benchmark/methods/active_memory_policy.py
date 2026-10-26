"""One-call navigation and memory selection for the experimental AMM method."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any, Optional, Sequence

from vnav_benchmark.fairness import ClampReport, clamp_action
from vnav_benchmark.interfaces import Action, Observation
from vnav_benchmark.methods.active_memory import ActiveMemory
from vnav_benchmark.methods.imggoal_vlm import ImgGoalVlmMethod
from vnav_benchmark.methods.nav_template import Decision, SubgoalInfo
from vnav_benchmark.vlm.client import TextPart, VlmError, VlmFatalError
from vnav_benchmark.vlm.memory import LogEntry
from vnav_benchmark.vlm.policy import (
    MAX_REASKS,
    ImageGoalVlmPolicy,
    PolicyStep,
    ProviderFatalError,
    RgbView,
)
from vnav_benchmark.vlm.prompt import (
    FEEDBACK_PREV_CALL_FAILED,
    FEEDBACK_PREV_INVALID,
    HOLD_S,
    SYSTEM_PROMPT,
    StepInputs,
    SubgoalView,
    build_request,
)
from vnav_benchmark.vlm.schema import ACTION_SCHEMA, OBS_MAX_WORDS, parse_decision, truncate_words

MEMORY_INSTRUCTIONS = """WHAT YOU RECEIVE EACH DECISION
You receive the goal, subgoal if any, current RGB, and only your retained history and note.
You manage episodic memory yourself, simultaneously with the next action. No extra call is made.
Select absolute decision IDs using retain_frames and retain_logs; include the current decision
if you want to remember this observation. Deleted items cannot be retrieved later. Empty lists
and an empty memory note are allowed. There is no fixed window, sample spacing, or note length.
Next-step memory has a combined budget of 1024 estimated context credits: each retained 320x240
thumbnail costs 96; UTF-8 bytes of note plus formatted retained action logs cost ceil(bytes/3).
Log labels count too. These are surrogate units, not actual API token counts. Stay under budget.
Choose content and size to balance navigation success, decision latency and total API usage.
No coordinates, depth, remaining motion budget, or unselected map images are available.
"""

AMM_SYSTEM = (
    SYSTEM_PROMPT[:SYSTEM_PROMPT.index("WHAT YOU RECEIVE EACH DECISION")]
    + MEMORY_INSTRUCTIONS
    + SYSTEM_PROMPT[SYSTEM_PROMPT.index("\nEXPLORING"):]
).replace("at most 40 words:", "within the combined memory budget:")
AMM_SYSTEM = AMM_SYSTEM.replace(
    '"vx": number, "vy": number, "wz": number}',
    '"vx": number, "vy": number, "wz": number, '
    '"retain_frames": array of integer decision IDs, "retain_logs": array of integer decision IDs}',
)
AMM_SCHEMA = deepcopy(ACTION_SCHEMA)
for _name in ("retain_frames", "retain_logs"):
    AMM_SCHEMA["properties"][_name] = {"type": "array", "items": {"type": "integer"}}
    AMM_SCHEMA["required"].append(_name)
AMM_FINGERPRINT = "sha256:" + hashlib.sha256(
    json.dumps(["memory-budget-v2", AMM_SYSTEM, AMM_SCHEMA], sort_keys=True).encode()
).hexdigest()


class ActiveMemoryPolicy(ImageGoalVlmPolicy):
    """Reuse baseline accounting and action limits with an explicit active-memory protocol."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._memory = ActiveMemory()
        self.fingerprint = AMM_FINGERPRINT

    def decide(
        self, obs: Observation, subgoal: Optional[SubgoalView] = None,
        feedback: Sequence[str] = (),
    ) -> PolicyStep:
        """Request an action and a validated atomic memory update in one normal API call."""
        view, d = RgbView(obs), self._decision
        self._memory.check_budget(d)
        fb = list(feedback)
        if self._last_status == "invalid":
            fb.append(FEEDBACK_PREV_INVALID)
        elif self._last_status == "call_failed":
            fb.append(FEEDBACK_PREV_CALL_FAILED)
        inputs = StepInputs(d, self._goal_jpeg, view.rgb_jpeg, subgoal,
                            self._memory.history(d), self._memory.log(), self._memory.note, fb)
        available_frames, available_logs = self._memory.available(d)
        menu = TextPart(
            f"Available absolute frame IDs: {available_frames}; log IDs: {available_logs}. "
            f"Current decision ID: {d}. Current retained memory cost: {self._memory.credits}."
        )
        latency = {"vlm_api_s": 0.0, "server_ms": 0.0, "retries": 0.0}
        decision, entry, payload = None, None, None
        calls, error, status = 0, "", "invalid"
        action, clamp = Action(0, 0, 0, HOLD_S), ClampReport()
        for attempt in range(1 + MAX_REASKS):
            base = build_request(inputs, reask_error=error)
            req = replace(base, system=AMM_SYSTEM, schema_name="vnav_action_amm_v1",
                          schema=AMM_SCHEMA, parts=base.parts + (menu,))
            calls += 1
            try:
                result = self._client.complete(req)
            except VlmFatalError as exc:
                self._log_call(d, attempt, req, None, "fatal", str(exc))
                raise ProviderFatalError(str(exc)) from exc
            except VlmError as exc:
                status, error = "call_failed", str(exc)
                self._log_call(d, attempt, req, None, status, error)
                break
            self._usage.add(result.usage)
            latency["vlm_api_s"] += result.wall_s
            latency["server_ms"] += result.server_ms or 0
            latency["retries"] += result.retries
            candidate, error = parse_decision(result.text)
            if candidate is not None:
                try:
                    payload = json.loads(result.text)
                    action, clamp = clamp_action(Action(
                        candidate.vx, candidate.vy, candidate.wz, HOLD_S, done=candidate.done
                    ))
                    observation, cut = truncate_words(candidate.observation, OBS_MAX_WORDS)
                    entry = LogEntry(d, action.vx, action.vy, action.wz, observation)
                    self._memory.validate(d, entry, candidate.memory,
                                          payload["retain_frames"], payload["retain_logs"])
                    self._memory.obs_truncations += int(cut)
                    decision, status = candidate, "ok"
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
            self._log_call(d, attempt, req, result, status, error)
            if decision is not None:
                break
            if attempt < MAX_REASKS:
                self.stats["reasks"] += 1
        if decision is not None:
            self._memory.update(d, view.rgb, entry, decision.memory,
                                payload["retain_frames"], payload["retain_logs"])
            action = replace(action, note=decision.observation)
        else:
            action, clamp = Action(0, 0, 0, HOLD_S), ClampReport()
        self._update_stats(status, clamp, action, calls)
        self._decision += 1
        self._last_status = status
        return PolicyStep(action, status, decision, clamp, calls, error, latency)

    def load_state(self, state: dict) -> None:
        """Restore only active-memory checkpoints with the same protocol fingerprint."""
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("active-memory checkpoint fingerprint differs")
        self._decision, self._last_status = state["decision"], state["last_status"]
        self._memory = ActiveMemory.from_state(state["memory"])
        self.stats = dict(state["stats"])


class ActiveMemoryMethod(ImgGoalVlmMethod):
    """Keep M5's navigation permissions and replace only its episodic memory policy."""

    def reset(self, spec: Any, obs: Observation, ctx: Any) -> None:
        """Initialize baseline localization/planning, then attach a fresh AMM policy."""
        super().reset(spec, obs, ctx)
        client = self._policy._client
        self._policy = ActiveMemoryPolicy(client, self._sink, spec.goal_rgb_jpeg, ctx.artifact_dir)
        self._summary.update(
            fingerprint=self._policy.fingerprint, experiment="Active Map and Memory"
        )

    def after_decide(self, obs: Observation, sub: SubgoalInfo, dec: Decision) -> Decision:
        """Expose retained IDs and budget use for the evaluator's audit logs."""
        result = super().after_decide(obs, sub, dec)
        state = self._policy.memory.to_state()
        audit = {
            "frame_ids": [int(i) for i in state["frames"]],
            "log_ids": [e["decision"] for e in state["log"]],
            "note_words": len(state["note"].split()), "credits": state["credits"],
        }
        return replace(result, debug=dict(result.debug, active_memory=audit))
