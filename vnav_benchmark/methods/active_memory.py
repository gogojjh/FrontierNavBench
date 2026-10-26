"""Agent-owned episodic storage with an explicit, bounded context budget."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from vnav_benchmark.vlm.memory import LogEntry, shrink_rgb
from vnav_benchmark.vlm.prompt import StepInputs, step_text

MEMORY_CREDITS = 1024
IMAGE_CREDITS = 96  # Surrogate cost for one 320x240 historical thumbnail.
TEXT_BYTES_PER_CREDIT = 3  # Conservative surrogate, not the API's actual token count.


class ActiveMemory:
    """Retain exactly the agent-selected observations, without an archival retrieval path."""

    def __init__(self) -> None:
        self._frames: Dict[int, bytes] = {}
        self._logs: Dict[int, LogEntry] = {}
        self.note = ""
        self.obs_truncations = 0
        self.note_truncations = 0
        self.credits = 0

    def history(self, decision: int) -> List[Tuple[int, bytes]]:
        """Return retained thumbnails labeled by age, oldest first."""
        return [(decision - i, jpeg) for i, jpeg in sorted(self._frames.items())]

    def log(self) -> Tuple[LogEntry, ...]:
        """Return only retained action/observation records."""
        return tuple(entry for _, entry in sorted(self._logs.items()))

    def available(self, decision: int) -> Tuple[List[int], List[int]]:
        """Include the current observation as an optional next-step memory item."""
        return sorted(set(self._frames) | {decision}), sorted(set(self._logs) | {decision})

    @staticmethod
    def _credits_for(decision: int, logs: Sequence[LogEntry], note: str, images: int) -> int:
        rendered = step_text(StepInputs(decision, b"", b"", None, (), logs, note, ()))
        memory_text = rendered.split("\n", 1)[1].rsplit("\nFEEDBACK: ", 1)[0]
        return images * IMAGE_CREDITS + math.ceil(
            len(memory_text.encode("utf-8")) / TEXT_BYTES_PER_CREDIT
        )

    def check_budget(self, decision: int) -> None:
        """Recheck the actual rendered state as retained log age labels grow.

        Raises:
            ValueError: Sending the retained memory would exceed the configured budget.
        """
        used = self._credits_for(decision, self.log(), self.note, len(self._frames))
        if used > MEMORY_CREDITS:
            raise ValueError(f"memory budget exceeded before request: {used} credits")
        self.credits = used

    def validate(
        self, decision: int, entry: LogEntry, note: str,
        frame_ids: Sequence[int], log_ids: Sequence[int],
    ) -> int:
        """Validate a proposed update without changing stored content.

        Raises:
            ValueError: A reference is unavailable, duplicated, malformed, or over budget.
        """
        for ids, available in zip((frame_ids, log_ids), self.available(decision)):
            if not isinstance(ids, list) or any(type(i) is not int for i in ids):
                raise ValueError("memory references must be arrays of integer IDs")
            if len(set(ids)) != len(ids) or not set(ids).issubset(available):
                raise ValueError("memory references must be unique and available")
        logs = dict(self._logs)
        logs[decision] = entry
        used = self._credits_for(decision + 1, [logs[i] for i in sorted(log_ids)],
                                 note, len(frame_ids))
        if used > MEMORY_CREDITS:
            raise ValueError(f"memory budget exceeded: {used} > {MEMORY_CREDITS} credits")
        return used

    def update(
        self, decision: int, rgb: np.ndarray, entry: LogEntry, note: str,
        frame_ids: Sequence[int], log_ids: Sequence[int],
    ) -> None:
        """Atomically commit a validated selection; deleted items cannot be recovered."""
        used = self.validate(decision, entry, note, frame_ids, log_ids)
        frames, logs = dict(self._frames), dict(self._logs)
        if decision in frame_ids:
            frames[decision] = shrink_rgb(rgb)
        logs[decision] = entry
        self._frames = {i: frames[i] for i in sorted(frame_ids)}
        self._logs = {i: logs[i] for i in sorted(log_ids)}
        self.note, self.credits = note, used

    def to_state(self) -> Dict[str, Any]:
        """Serialize retained memory only; there is no deleted-item archive."""
        return {
            "frames": {str(i): base64.b64encode(j).decode("ascii")
                       for i, j in self._frames.items()},
            "log": [asdict(entry) for entry in self.log()],
            "note": self.note, "credits": self.credits,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> ActiveMemory:
        """Restore a trusted checkpoint produced by this memory implementation."""
        memory = cls()
        memory._frames = {int(i): base64.b64decode(j) for i, j in state["frames"].items()}
        memory._logs = {e["decision"]: LogEntry(**e) for e in state["log"]}
        memory.note, memory.credits = state["note"], state["credits"]
        return memory
