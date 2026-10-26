"""Shared navigation-method template for point-goal SRU and image-goal VLM methods.

Every step follows Observation input, system-level subgoal progression producing
:class:`SubgoalInfo`, a :class:`Decision`, and Decision.to_action(). The action is
cmd_vel-equivalent: body-frame vx/vy/wz, hold_s and an arrival declaration.

Subclasses implement four hooks; this template owns ordering, logging and atomic
summary writes:
1. setup(spec, obs, ctx): localize, plan and start the SRU sidecar or VLM client.
2. subgoal(obs): advance from poses or the previous VLM decision. A pose-based
   system arrival sets system_done, skips the controller and uses
   system_done_decision for this step.
3. decide(obs, sub): infer one action. SRU receives depth and subgoal coordinates;
   the VLM receives RGB and subgoal images.
4. after_decide(obs, sub, dec): accept/ignore VLM completion/progression, update
   counters and prepare feedback. Point-goal methods return the decision unchanged.

close_backend() shuts down the controller; summary() supplies episode-summary content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from frontier_nav_bench.interfaces import Action, EpisodeSpec, MethodContext, Observation, Pose2D


@dataclass(frozen=True)
class SubgoalInfo:
    """System subgoal passed to the controller; coordinates remain internal and never reach the VLM.
    """

    phase: str  # Method phase: subgoal/final/align/done for point goals, NAVIGATE etc. for image
    # goals
    system_done: bool = False  # System pose-based arrival or completed episode; skip the controller
    # this step
    kf_id: Optional[int] = None  # Current subgoal keyframe ID
    index: Optional[int] = None  # One-based subgoal index used in VLM labels
    total: Optional[int] = None
    pose_map: Optional[Pose2D] = None  # Subgoal coordinates for the point-goal controller
    image_jpeg: Optional[bytes] = None  # Subgoal image for the VLM controller
    feedback: Tuple[str, ...] = ()  # System feedback sentence for the VLM; always empty for
    # point-goal methods
    extra: Dict[str, Any] = field(default_factory=dict)  # Method-private auxiliary data


@dataclass(frozen=True)
class Decision:
    """One decision: cmd_vel-equivalent output plus step logging/debug information."""

    vx: float
    vy: float
    wz: float
    hold_s: float
    done: bool = False
    note: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)  # Copied unchanged into Action.debug for
    # runner logging
    log: Optional[Dict[str, Any]] = None  # Method steps.jsonl row; None suppresses the row
    raw: Any = None  # Raw controller result, such as PolicyStep, for after_decide

    def to_action(self) -> Action:
        return Action(
            self.vx, self.vy, self.wz, self.hold_s, done=self.done, note=self.note,
            debug=self.debug,
        )  # fmt: skip


class NavMethod:
    """Fixed reset/act/close flow with subclass hooks.

    setup configures steps_log_path and summary_path; None disables each output.
    """

    uses_truth = False
    method_id: str = ""
    steps_log_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    LOG_ENSURE_ASCII = True  # Preserve each method family existing steps.jsonl format
    SUMMARY_JSON_KW: Dict[str, Any] = {"indent": 1}

    # ---- Subclass hooks ----
    def setup(self, spec: EpisodeSpec, obs: Observation, ctx: MethodContext) -> None:
        raise NotImplementedError

    def subgoal(self, obs: Observation) -> SubgoalInfo:
        raise NotImplementedError

    def decide(self, obs: Observation, sub: SubgoalInfo) -> Decision:
        raise NotImplementedError

    def after_decide(self, obs: Observation, sub: SubgoalInfo, dec: Decision) -> Decision:
        return dec

    def system_done_decision(self, obs: Observation, sub: SubgoalInfo) -> Decision:
        raise NotImplementedError

    def close_backend(self) -> None:
        return None

    def summary(self) -> Dict[str, Any]:
        return {}

    # ---- Fixed lifecycle ----
    def reset(self, spec: EpisodeSpec, obs: Observation, ctx: MethodContext) -> None:
        previous = getattr(self, "_transit_driver", None)
        if previous is not None:
            previous.close()
        self._transit_driver = None
        self.setup(spec, obs, ctx)

    def act(self, obs: Observation) -> Action:
        driver = getattr(self, "_transit_driver", None)
        dec = driver.act(obs) if driver is not None else None
        if dec is not None:
            self._log_step(dec)
            return dec.to_action()
        sub = self.subgoal(obs)
        if sub.system_done:
            dec = self.system_done_decision(obs, sub)
        else:
            dec = self.after_decide(obs, sub, self.decide(obs, sub))
        self._log_step(dec)
        return dec.to_action()

    def configure_transit(
        self, ctx: MethodContext, route_ids: Sequence[int], start: Pose2D,
        obs: Observation, controller: Any = None,
    ) -> None:
        """Configure the same depth-only transfer executor for mapped methods."""
        from frontier_nav_bench.methods.transit_driver import TransitDriver

        self._transit_driver = TransitDriver.create(
            ctx, route_ids, start, obs, self.transit_completed, controller,
        )

    def transit_completed(self, consumed: Sequence[int]) -> None:
        """Consume the covered prefix in the subclass's normal route representation."""
        raise NotImplementedError

    def transit_summary(self) -> Dict[str, Any]:
        """Return transfer provenance only when a candidate was identified."""
        driver = getattr(self, "_transit_driver", None)
        return driver.summary() if driver is not None else {}

    def _log_step(self, dec: Decision) -> None:
        if dec.log is None or self.steps_log_path is None:
            return
        self.steps_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.steps_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dec.log, ensure_ascii=self.LOG_ENSURE_ASCII) + "\n")

    def write_summary(self) -> None:
        """Atomically write a summary through .tmp and rename; skip empty summaries."""
        out = self.summary()
        if self.summary_path is None or not out:
            return
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.summary_path.with_name(self.summary_path.name + ".tmp")
        tmp.write_text(json.dumps(out, **self.SUMMARY_JSON_KW), encoding="utf-8")
        tmp.replace(self.summary_path)

    def close(self) -> None:
        try:
            self.close_backend()
        finally:
            driver = getattr(self, "_transit_driver", None)
            if driver is not None:
                driver.close()
        self.write_summary()
