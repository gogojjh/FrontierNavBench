"""Shared image-goal VLM policy.

Each step builds a request, calls the model, validates output, reasks at most
once if invalid, clamps the action and falls back to zero velocity on failure.
RgbView permits only RGB-related observation access, raising on odometry, depth
or remaining budget. Log each HTTP call in calls.jsonl with a reconstructable
request manifest; store images once by SHA and send usage to UsageSink.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from frontier_nav_bench.fairness import ClampReport, clamp_action
from frontier_nav_bench.interfaces import Action, FatalInfraError, Observation, UsageSink
from frontier_nav_bench.vlm.client import (
    ImagePart,
    TextPart,
    VlmClientLike,
    VlmError,
    VlmFatalError,
    VlmRequest,
    VlmResult,
)
from frontier_nav_bench.vlm.memory import VlmMemory
from frontier_nav_bench.vlm.prompt import (
    FEEDBACK_PREV_CALL_FAILED,
    FEEDBACK_PREV_INVALID,
    HOLD_S,
    StepInputs,
    SubgoalView,
    build_request,
    prompt_fingerprint,
)
from frontier_nav_bench.vlm.schema import VlmDecision, parse_decision

MAX_REASKS = 1  # Maximum same-step reasks after invalid output


class ProviderFatalError(FatalInfraError):
    """Provider authentication/quota/billing failure; FatalInfraError stops the campaign unscored.
    """


class RgbView:
    """Read-only observation view permitting step_index / rgb / rgb_jpeg / K and rejecting other
    fields.
    """

    _ALLOWED = ("step_index", "rgb", "rgb_jpeg", "K")

    def __init__(self, obs: Observation) -> None:
        object.__setattr__(self, "_obs", obs)

    def __getattr__(self, name: str) -> Any:
        if name in RgbView._ALLOWED:
            return getattr(object.__getattribute__(self, "_obs"), name)
        raise AttributeError(f"看图策略不许读观测字段 {name!r}(只给 RGB)")


@dataclass(frozen=True)
class PolicyStep:
    """One policy decision, with clamped velocity and zero motion on failure/done."""

    action: Action
    status: str  # "ok" | "invalid" | "call_failed"
    decision: Optional[VlmDecision]
    clamp: ClampReport
    calls: int  # HTTP calls in this step, including reasks
    error: str = ""
    latency: Dict[str, float] = field(default_factory=dict)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ImageGoalVlmPolicy:
    """One instance per episode; artifact_dir=None disables output for tests."""

    def __init__(
        self,
        client: VlmClientLike,
        usage: UsageSink,
        goal_jpeg: bytes,
        artifact_dir: Optional[Path] = None,
    ) -> None:
        self._client = client
        self._usage = usage
        self._goal_jpeg = goal_jpeg
        self._dir = artifact_dir / "vlm" if artifact_dir is not None else None
        self._memory = VlmMemory()
        self._decision = 0
        self._last_status = "ok"
        self._saved_images: set = set()
        self.fingerprint = prompt_fingerprint()
        self.stats: Dict[str, int] = {
            "decisions": 0,
            "calls": 0,
            "reasks": 0,
            "invalid": 0,
            "call_failed": 0,
            "clamped": 0,
            "done": 0,
            "consecutive_call_failed": 0,
        }
        if self._dir is not None:
            (self._dir / "images").mkdir(parents=True, exist_ok=True)

    # ---- Main decision flow ----
    def decide(
        self,
        obs: Observation,
        subgoal: Optional[SubgoalView] = None,
        feedback: Sequence[str] = (),
    ) -> PolicyStep:
        view = RgbView(obs)
        d = self._decision
        fb: List[str] = []
        if self._last_status == "invalid":
            fb.append(FEEDBACK_PREV_INVALID)
        elif self._last_status == "call_failed":
            fb.append(FEEDBACK_PREV_CALL_FAILED)
        fb.extend(feedback)
        inputs = StepInputs(
            decision=d,
            goal_jpeg=self._goal_jpeg,
            current_jpeg=view.rgb_jpeg,
            subgoal=subgoal,
            history=self._memory.history(d),
            log=self._memory.log(),
            memory_note=self._memory.note,
            feedback=fb,
        )
        latency = {"vlm_api_s": 0.0, "server_ms": 0.0, "retries": 0.0}
        decision: Optional[VlmDecision] = None
        status, error, calls, reask_error = "ok", "", 0, ""
        for attempt in range(1 + MAX_REASKS):
            req = build_request(inputs, reask_error=reask_error)
            calls += 1
            try:
                result = self._client.complete(req)
            except VlmFatalError as exc:
                self._log_call(d, attempt, req, None, "fatal", str(exc))
                raise ProviderFatalError(str(exc)) from exc
            except VlmError as exc:
                self._log_call(d, attempt, req, None, "call_failed", str(exc))
                status, error = "call_failed", str(exc)
                break
            self._usage.add(result.usage)
            latency["vlm_api_s"] += result.wall_s
            latency["server_ms"] += result.server_ms or 0.0
            latency["retries"] += result.retries
            decision, reason = parse_decision(result.text)
            self._log_call(d, attempt, req, result, "ok" if decision else "invalid", reason)
            if decision is not None:
                status, error = "ok", ""
                break
            status, error, reask_error = "invalid", reason, reason
            if attempt < MAX_REASKS:
                self.stats["reasks"] += 1

        if decision is not None:
            raw = Action(decision.vx, decision.vy, decision.wz, HOLD_S, done=decision.done)
            action, clamp = clamp_action(raw)
            action = Action(
                action.vx,
                action.vy,
                action.wz,
                action.hold_s,
                done=action.done,
                note=decision.observation,
            )
            self._memory.set_note(decision.memory)
        else:
            action, clamp = Action(0.0, 0.0, 0.0, HOLD_S), ClampReport()
        self._memory.record(
            d, action.vx, action.vy, action.wz, decision.observation if decision else "", status
        )
        self._memory.add_frame(d, view.rgb)
        self._update_stats(status, clamp, action, calls)
        self._last_status = status
        self._decision += 1
        return PolicyStep(action, status, decision, clamp, calls, error, latency)

    def _update_stats(self, status: str, clamp: ClampReport, action: Action, calls: int) -> None:
        s = self.stats
        s["decisions"] += 1
        s["calls"] += calls
        s["clamped"] += int(clamp.any)
        s["done"] += int(action.done)
        if status == "call_failed":
            s["call_failed"] += 1
            s["consecutive_call_failed"] += 1
        else:
            s["consecutive_call_failed"] = 0
            s["invalid"] += int(status == "invalid")

    # ---- Artifact output ----
    def _manifest(self, req: VlmRequest) -> List[Dict[str, Any]]:
        """Request manifest with exact text and image SHA/dimensions; images live in
        images/<sha>.jpg.
        """
        items: List[Dict[str, Any]] = []
        for part in req.parts:
            if isinstance(part, TextPart):
                items.append({"t": part.text, "bp": part.cache_breakpoint})
            else:
                assert isinstance(part, ImagePart)
                sha = _sha(part.jpeg)
                self._save_image(sha, part.jpeg)
                items.append({"img": sha, "w": part.width, "h": part.height, "detail": part.detail})
        return items

    def _save_image(self, sha: str, data: bytes) -> None:
        if self._dir is None or sha in self._saved_images:
            return
        path = self._dir / "images" / f"{sha}.jpg"
        if not path.exists():
            path.write_bytes(data)
        self._saved_images.add(sha)

    def _log_call(
        self,
        decision: int,
        attempt: int,
        req: VlmRequest,
        result: Optional[VlmResult],
        status: str,
        error: str,
    ) -> None:
        if self._dir is None:
            return
        row: Dict[str, Any] = {
            "decision": decision,
            "attempt": attempt,
            "status": status,
            "error": error,
            "endpoint": self._client.endpoint.name,
            "fingerprint": self.fingerprint,
            "parts": self._manifest(req),
        }
        if result is not None:
            row.update(
                text=result.text,
                model_reported=result.model_reported,
                response_id=result.response_id,
                service_tier=result.service_tier,
                finish_reason=result.finish_reason,
                server_ms=result.server_ms,
                wall_s=result.wall_s,
                retries=result.retries,
                counts=asdict(result.counts),
                usage=asdict(result.usage),
                reasoning_chars=len(result.reasoning_text),
            )
        with open(self._dir / "calls.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- Save / restore ----
    def to_state(self) -> Dict[str, Any]:
        return {
            "decision": self._decision,
            "last_status": self._last_status,
            "memory": self._memory.to_state(),
            "stats": dict(self.stats),
            "fingerprint": self.fingerprint,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("存档的提示词指纹和当前代码不一致,不能续跑")
        self._decision = int(state["decision"])
        self._last_status = state["last_status"]
        self._memory = VlmMemory.from_state(state["memory"])
        self.stats = {k: int(v) for k, v in state["stats"].items()}

    @property
    def memory(self) -> VlmMemory:
        return self._memory
