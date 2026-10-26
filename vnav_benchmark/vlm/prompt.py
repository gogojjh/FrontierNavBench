"""Shared prompt and request assembly, with method-specific subgoal-image inclusion only.

Preserve byte-identical prompt content for both providers and image-goal methods.
Stable prefixes precede variable input for provider caching:
system prompt [breakpoint], goal image + END [breakpoint], optional subgoal image
+ END [breakpoint], historical frames, current frame and per-step text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from vnav_benchmark.fairness import VX_MAX, VX_MIN, VY_MAX, WZ_MAX
from vnav_benchmark.vlm.client import ImagePart, Part, TextPart, VlmRequest
from vnav_benchmark.vlm.memory import HISTORY_OFFSETS, HISTORY_SIZE, LOG_STEPS, LogEntry
from vnav_benchmark.vlm.schema import (
    ACTION_SCHEMA,
    MEMORY_MAX_WORDS,
    OBS_MAX_WORDS,
    SCHEMA_NAME,
)

PROMPT_VERSION = "v1"
FRAME_SIZE = (640, 480)  # (Width, height) for current, goal and keyframe images
HOLD_S = 1.0  # Simulation seconds per decision, included in the prompt

# V6 Go2/env0 calibration: one trial per command, starting still and holding for 1.0 s;
# Source: acceptance/V6/calib_20261021-185235.json; wz 0.3/0.5/0.7 -> 14.4/28.5/40.6 degrees.
# vx 0.3 / 0.5 / 0.8 produced 0.22 / 0.43 / 0.68 m. Camera height is 0.63 m standing,
# about 0.68 m walking above the spawn-floor map origin; the prompt uses about 0.7 m.
CALIBRATED = True
CALIBRATION = {
    "TURN_DEG_W03": "14",  # Degrees turned by holding wz=0.3 rad/s for one second
    "TURN_DEG_W05": "28",  # wz 0.5
    "TURN_DEG_W07": "41",  # wz 0.7
    "FWD_M_V05": "0.4",  # Meters traveled by holding vx=0.5 m/s for one second
    "FWD_M_V08": "0.7",  # vx 0.8
    "DECISIONS_PER_TURN": "9",  # Decisions per full turn using wz=0.7: 360 / 41
}

SYSTEM_PROMPT_TEMPLATE = """\
You are the pilot of a four-legged robot (Unitree Go2) indoors. You see only through one \
forward-facing RGB camera (640x480, about 95 degrees horizontal field of view, mounted about \
0.7 m above the floor). You control the robot by choosing a velocity command at every decision.

TASK
You are given a GOAL IMAGE: a photo taken by this same camera at the target place. Drive to the \
place where the goal image was taken and face the same direction. The episode succeeds only if, \
when you declare done, the robot is within 0.5 m of that place and within 45 degrees of that \
heading. Declaring done anywhere else is a failure, and so is running out of time. Time is \
limited, so move efficiently, but do not rush the final alignment.

SUBGOALS (only when a CURRENT SUBGOAL IMAGE is given)
A subgoal image is a photo taken earlier, during a mapping run, along a route that leads toward \
the goal. Drive to where the subgoal image was taken. When you are within about 1 m of that spot \
and your view roughly matches it, set "subgoal_reached": true; you will then be given the next \
subgoal image, or no subgoal image when the goal is close. Subgoal images may be several meters \
apart and the next one may not be visible yet: head in the direction the route is going. The \
goal image is always the final target. When no subgoal image is given, always set \
"subgoal_reached": false.

ACTIONS
Each decision applies your command (vx, vy, wz) for exactly {hold} second, then you decide \
again. The world is paused while you think.
- vx: forward speed in m/s, from {vx_min} to {vx_max}. The robot cannot move backward.
- vy: sideways speed in m/s, from -{vy_max} to {vy_max}. Positive = to the robot's left.
- wz: turning speed in rad/s, from -{wz_max} to {wz_max}. Positive = turn left \
(counter-clockwise).
Measured effect of one decision: wz 0.3 turns about {TURN_DEG_W03} degrees, wz 0.5 about \
{TURN_DEG_W05} degrees, wz 0.7 about {TURN_DEG_W07} degrees. vx 0.5 moves about \
{FWD_M_V05} m, vx 0.8 about {FWD_M_V08} m. A full turn in place takes about \
{DECISIONS_PER_TURN} decisions.
Avoid collisions: if a wall or furniture fills the lower part of the view, turn or side-step \
instead of going forward. If the robot falls over, the episode fails. Near the goal, \
use small speeds to line up.

WHAT YOU RECEIVE EACH DECISION
- GOAL IMAGE, and CURRENT SUBGOAL IMAGE if any.
- PAST FRAMES: up to {n_hist} smaller ({hist_w}x{hist_h}) frames from {hist_list} decisions \
ago, oldest first, labeled d-{hist_max} ... d-1. Use them to judge your own motion and to \
recognize places you have already seen.
- CURRENT FRAME: what the camera sees now.
- RECENT LOG: your last {log_steps} commands (as executed) with the observation you wrote at \
each decision.
- MEMORY: the note you wrote at the previous decision.
- FEEDBACK: messages from the system, e.g. that your previous reply was invalid or that the \
subgoal changed.
You get no map, no position and no depth. Rely on what you see.

EXPLORING
If the target is not in view, look around by turning in place, then go through doorways and \
along corridors toward areas you have not visited. Use MEMORY to avoid searching the same room \
twice.

OUTPUT
Reply with one JSON object only, with exactly these fields in this order:
{{"observation": string, at most {obs_words} words: what you see now that matters;
 "target_visible": boolean, true if the place in the current target image (the subgoal image \
if given, otherwise the goal image) is visible in the current frame;
 "subgoal_reached": boolean, see SUBGOALS;
 "done": boolean, true only when you are at the goal place facing the goal direction; this \
ends the episode;
 "memory": string, at most {mem_words} words: where you have been, where the goal probably is, \
your plan; it is shown back to you next decision;
 "vx": number, "vy": number, "wz": number}}
When "done" is true the robot stops; your velocities are ignored."""


def _render_system_prompt() -> str:
    hist = sorted(HISTORY_OFFSETS, reverse=True)
    return SYSTEM_PROMPT_TEMPLATE.format(
        hold=f"{HOLD_S:.1f}",
        vx_min=f"{VX_MIN:.1f}",
        vx_max=f"{VX_MAX:.1f}",
        vy_max=f"{VY_MAX:.1f}",
        wz_max=f"{WZ_MAX:.1f}",
        n_hist=len(hist),
        hist_w=HISTORY_SIZE[0],
        hist_h=HISTORY_SIZE[1],
        hist_list=", ".join(str(k) for k in hist[:-1]) + f" and {hist[-1]}",
        hist_max=hist[0],
        log_steps=LOG_STEPS,
        obs_words=OBS_MAX_WORDS,
        mem_words=MEMORY_MAX_WORDS,
        **CALIBRATION,
    )


SYSTEM_PROMPT = _render_system_prompt()

# ---- Content labels; stable labels form the cached prefix ----
GOAL_LABEL = "GOAL IMAGE (final target: the place where this photo was taken, facing the same way):"
GOAL_END = "END OF GOAL IMAGE"
SUBGOAL_LABEL = "CURRENT SUBGOAL IMAGE (subgoal {k} of {n}, from the earlier mapping run):"
SUBGOAL_END = "END OF SUBGOAL IMAGE"
PAST_LABEL = "PAST FRAMES (oldest first):"
PAST_NONE = "PAST FRAMES: none yet."
PAST_ITEM = "d-{k}:"
CURRENT_LABEL = "CURRENT FRAME (decision {d}):"
REPLY_LINE = "Reply with the JSON object only."

# ---- System feedback selected by policy/method wrappers for the FEEDBACK line ----
FEEDBACK_NONE = "none"
FEEDBACK_PREV_INVALID = "Your previous reply could not be used; the robot stood still."
FEEDBACK_PREV_CALL_FAILED = (
    "The previous decision failed due to a system error; the robot stood still."
)
FEEDBACK_SUBGOAL_NEW = "You reached the previous subgoal; this is a new subgoal image."
FEEDBACK_SUBGOAL_FINAL = "No more subgoal images: the goal is close. Use the goal image."
# For pose-judged M2/M3, send this next step when VLM done disagrees with system arrival.
FEEDBACK_NOT_AT_GOAL = "You are not at the goal yet: move closer and line up with the goal image."
REASK_LINE = "Your reply above could not be used ({error}). Reply again with only the JSON object."


@dataclass(frozen=True)
class SubgoalView:
    jpeg: bytes
    index: int  # One-based subgoal index
    total: int
    width: int = FRAME_SIZE[0]
    height: int = FRAME_SIZE[1]


@dataclass(frozen=True)
class StepInputs:
    decision: int
    goal_jpeg: bytes
    current_jpeg: bytes
    subgoal: Optional[SubgoalView]
    history: Sequence[Tuple[int, bytes]]  # (Steps ago, 320x240 JPEG) pairs, oldest first
    log: Sequence[LogEntry]
    memory_note: str
    feedback: Sequence[str]


def _fmt_speed(v: float) -> str:
    text = f"{v:.2f}"
    return "0.00" if text == "-0.00" else text


def format_log_line(entry: LogEntry, decision: int) -> str:
    head = (
        f"d-{decision - entry.decision}: vx={_fmt_speed(entry.vx)} vy={_fmt_speed(entry.vy)} "
        f"wz={_fmt_speed(entry.wz)}"
    )
    if entry.status == "invalid":
        return head + " | (reply could not be used, robot stood still)"
    if entry.status == "call_failed":
        return head + " | (system error, robot stood still)"
    return head + " | obs: " + entry.observation


def step_text(inp: StepInputs) -> str:
    lines = [f"DECISION {inp.decision}"]
    if inp.log:
        lines.append("RECENT LOG (oldest first):")
        lines.extend(format_log_line(e, inp.decision) for e in inp.log)
    else:
        lines.append("RECENT LOG: none yet.")
    lines.append("MEMORY: " + (inp.memory_note if inp.memory_note else "none yet."))
    lines.append("FEEDBACK: " + (" ".join(inp.feedback) if inp.feedback else FEEDBACK_NONE))
    lines.append(REPLY_LINE)
    return "\n".join(lines)


def build_request(inp: StepInputs, reask_error: str = "") -> VlmRequest:
    """Build one request; nonempty reask_error adds feedback for a same-step reask."""
    w, h = FRAME_SIZE
    parts: List[Part] = [
        TextPart(GOAL_LABEL),
        ImagePart(inp.goal_jpeg, w, h),
        TextPart(GOAL_END, cache_breakpoint=True),
    ]
    if inp.subgoal is not None:
        sg = inp.subgoal
        parts += [
            TextPart(SUBGOAL_LABEL.format(k=sg.index, n=sg.total)),
            ImagePart(sg.jpeg, sg.width, sg.height),
            TextPart(SUBGOAL_END, cache_breakpoint=True),
        ]
    if inp.history:
        parts.append(TextPart(PAST_LABEL))
        for k, jpeg in inp.history:
            parts += [TextPart(PAST_ITEM.format(k=k)), ImagePart(jpeg, *HISTORY_SIZE)]
    else:
        parts.append(TextPart(PAST_NONE))
    parts += [TextPart(CURRENT_LABEL.format(d=inp.decision)), ImagePart(inp.current_jpeg, w, h)]
    parts.append(TextPart(step_text(inp)))
    if reask_error:
        parts.append(TextPart(REASK_LINE.format(error=reask_error)))
    return VlmRequest(
        system=SYSTEM_PROMPT, parts=tuple(parts), schema_name=SCHEMA_NAME, schema=ACTION_SCHEMA
    )


def prompt_fingerprint() -> str:
    """Fingerprint system text, labels, feedback, schema and memory constants; reject mixed reports.
    """
    material = {
        "version": PROMPT_VERSION,
        "system": SYSTEM_PROMPT,
        "labels": [
            GOAL_LABEL,
            GOAL_END,
            SUBGOAL_LABEL,
            SUBGOAL_END,
            PAST_LABEL,
            PAST_NONE,
            PAST_ITEM,
            CURRENT_LABEL,
            REPLY_LINE,
            REASK_LINE,
        ],
        "feedback": [
            FEEDBACK_NONE,
            FEEDBACK_PREV_INVALID,
            FEEDBACK_PREV_CALL_FAILED,
            FEEDBACK_SUBGOAL_NEW,
            FEEDBACK_NOT_AT_GOAL,
            FEEDBACK_SUBGOAL_FINAL,
        ],
        "schema": [SCHEMA_NAME, ACTION_SCHEMA],
        "memory": [
            list(HISTORY_OFFSETS),
            list(HISTORY_SIZE),
            LOG_STEPS,
            OBS_MAX_WORDS,
            MEMORY_MAX_WORDS,
        ],
        "frame": [list(FRAME_SIZE), HOLD_S],
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Placeholder fingerprint before V6 calibration; recomputed and pinned afterward by tests
PROMPT_FINGERPRINT_V1 = "sha256:00106881f1c6a91b476b15ec847cf643b79df076d272046c9f8c5b54204c4591"
