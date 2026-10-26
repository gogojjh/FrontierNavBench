"""VLM episode memory: sparse historical images, action logs and a model-written summary.

At step t, provide current images from t-8, t-4, t-2 and t-1 as re-encoded
320x240 thumbnails. Keep the latest 15 executed commands and model observations
(up to 15 words), plus the preceding model memory unchanged (up to 40 words).
All state supports save/restore through to_state/from_state using plain JSON.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Tuple

import cv2
import numpy as np

from frontier_nav_bench.vlm.schema import MEMORY_MAX_WORDS, OBS_MAX_WORDS, truncate_words

HISTORY_OFFSETS = (8, 4, 2, 1)  # Frame offsets t-k, oldest first
HISTORY_SIZE = (320, 240)  # (Width, height) pixels; 96 tokens per image for the gpt-6 family
HISTORY_JPEG_QUALITY = 85  # Match the original simulation JPEG quality
LOG_STEPS = 15  # Number of action-log steps to retain
_KEEP_FRAMES = max(HISTORY_OFFSETS)  # Only retain thumbnails for the latest eight steps


@dataclass(frozen=True)
class LogEntry:
    decision: int
    vx: float  # Actually executed velocity after clamping
    vy: float
    wz: float
    observation: str  # Truncated to OBS_MAX_WORDS; empty for invalid replies
    status: str = "ok"  # "ok" | "invalid" | "call_failed"


def shrink_rgb(rgb: np.ndarray) -> bytes:
    """Deterministically resize RGB uint8 to a 320x240 JPEG using INTER_AREA and quality 85."""
    small = cv2.resize(rgb, HISTORY_SIZE, interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, HISTORY_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


class VlmMemory:
    """Memory for a single episode; create a fresh instance each time."""

    def __init__(self) -> None:
        self._frames: Dict[int, bytes] = {}
        self._log: Deque[LogEntry] = deque(maxlen=LOG_STEPS)
        self.note = ""  # Truncated memory written by the model on the previous step
        self.obs_truncations = 0
        self.note_truncations = 0

    def add_frame(self, decision: int, rgb: np.ndarray) -> None:
        """Store the current decision frame as a thumbnail for subsequent historical-image inputs.
        """
        self._frames[decision] = shrink_rgb(rgb)
        for old in [d for d in self._frames if d <= decision - _KEEP_FRAMES]:
            del self._frames[old]

    def history(self, decision: int) -> List[Tuple[int, bytes]]:
        """Return available history as (steps ago, JPEG) pairs, oldest first, skipping missing
        frames.
        """
        return [
            (k, self._frames[decision - k])
            for k in HISTORY_OFFSETS
            if decision - k >= 0 and decision - k in self._frames
        ]

    def record(
        self, decision: int, vx: float, vy: float, wz: float, observation: str, status: str = "ok"
    ) -> None:
        obs, cut = truncate_words(observation, OBS_MAX_WORDS)
        self.obs_truncations += int(cut)
        self._log.append(LogEntry(decision, vx, vy, wz, obs, status))

    def set_note(self, text: str) -> None:
        self.note, cut = truncate_words(text, MEMORY_MAX_WORDS)
        self.note_truncations += int(cut)

    def log(self) -> Tuple[LogEntry, ...]:
        return tuple(self._log)

    def to_state(self) -> Dict[str, Any]:
        frames = {str(d): base64.b64encode(j).decode("ascii") for d, j in self._frames.items()}
        return {
            "frames": frames,
            "log": [asdict(e) for e in self._log],
            "note": self.note,
            "obs_truncations": self.obs_truncations,
            "note_truncations": self.note_truncations,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> VlmMemory:
        mem = cls()
        mem._frames = {int(d): base64.b64decode(j) for d, j in state["frames"].items()}
        for e in state["log"]:
            mem._log.append(LogEntry(**e))
        mem.note = state["note"]
        mem.obs_truncations = int(state["obs_truncations"])
        mem.note_truncations = int(state["note_truncations"])
        return mem
