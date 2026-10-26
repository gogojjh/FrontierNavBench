"""Shared per-step VLM JSON output schema and parser.

Both providers/methods use byte-identical field order: observation,
target_visible, subgoal_reached, done and memory before velocities. Numeric bounds
match interface limits. Out-of-range values are accepted, then clamped/counted
by the policy.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from frontier_nav_bench.fairness import VX_MAX, VX_MIN, VY_MAX, WZ_MAX

SCHEMA_NAME = "vnav_action_v1"
OBS_MAX_WORDS = 15  # Observation word limit; accept longer input and truncate when logging
MEMORY_MAX_WORDS = 40  # Memory word limit, handled like observation

FIELD_ORDER = (
    "observation",
    "target_visible",
    "subgoal_reached",
    "done",
    "memory",
    "vx",
    "vy",
    "wz",
)

ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "target_visible": {"type": "boolean"},
        "subgoal_reached": {"type": "boolean"},
        "done": {"type": "boolean"},
        "memory": {"type": "string"},
        "vx": {"type": "number", "minimum": VX_MIN, "maximum": VX_MAX},
        "vy": {"type": "number", "minimum": -VY_MAX, "maximum": VY_MAX},
        "wz": {"type": "number", "minimum": -WZ_MAX, "maximum": WZ_MAX},
    },
    "required": list(FIELD_ORDER),
    "additionalProperties": False,
}

_BOOL_FIELDS = ("target_visible", "subgoal_reached", "done")
_STR_FIELDS = ("observation", "memory")
_NUM_FIELDS = ("vx", "vy", "wz")
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


@dataclass(frozen=True)
class VlmDecision:
    observation: str
    target_visible: bool
    subgoal_reached: bool
    done: bool
    memory: str
    vx: float
    vy: float
    wz: float


def truncate_words(text: str, max_words: int) -> Tuple[str, bool]:
    """Split on whitespace and keep max_words; return (text, whether truncated)."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words), False
    return " ".join(words[:max_words]), True


def parse_decision(text: str) -> Tuple[Optional[VlmDecision], str]:
    """Parse model text into a decision, or return (None, English feedback reason).

    Allow surrounding whitespace, ```json fences and extra ignored fields.
    Reject missing fields, incorrect types and nonfinite numbers. Failure reasons
    are sent back to the model in FEEDBACK.
    """
    body = text.strip()
    fence = _FENCE_RE.match(body)
    if fence:
        body = fence.group(1)
    try:
        data = json.loads(body)
    except ValueError:
        return None, "the reply was not valid JSON"
    if not isinstance(data, dict):
        return None, "the reply was not a JSON object"
    missing = [k for k in FIELD_ORDER if k not in data]
    if missing:
        return None, "missing fields: " + ", ".join(missing)
    for key in _BOOL_FIELDS:
        if not isinstance(data[key], bool):
            return None, f'"{key}" must be true or false'
    for key in _STR_FIELDS:
        if not isinstance(data[key], str):
            return None, f'"{key}" must be a string'
    for key in _NUM_FIELDS:
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f'"{key}" must be a number'
        if not math.isfinite(float(value)):
            return None, f'"{key}" must be a finite number'
    return (
        VlmDecision(
            observation=data["observation"],
            target_visible=data["target_visible"],
            subgoal_reached=data["subgoal_reached"],
            done=data["done"],
            memory=data["memory"],
            vx=float(data["vx"]),
            vy=float(data["vy"]),
            wz=float(data["wz"]),
        ),
        "",
    )
