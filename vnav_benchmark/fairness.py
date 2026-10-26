"""Fairness rules: method resources, velocity limits, hold quantization and localization quotas.

Method names define the contract:
``{dense|sparse|nomap}-{contloc|initloc|noloc}-{ptgoal|imggoal}[-astra|-qwen]``.
They determine the map, localization quota and arrival judge (plan sections 2 and 6.3).
The interface layer is frozen after T-I0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from vnav_benchmark.interfaces import Action, Intrinsics, Localizer, LocResult

# ---- Shared velocity envelope, matching SRU simulation (plan section 6.8 / D5) ----
VX_MIN, VX_MAX = 0.0, 0.8  # m/s; reverse motion is disabled by the approved benchmark protocol
VY_MAX = 0.3  # m/s; symmetric lateral limit
WZ_MAX = 0.7  # rad/s; symmetric yaw-rate limit

# ---- Hold duration for one command ----
HOLD_QUANTUM_S = 0.1  # Quantization step: 0.1 s equals 20 physics steps at 200 Hz
HOLD_MIN_S = 0.1
HOLD_MAX_S = 2.0

MAP_VARIANTS = ("dense", "sparse", "nomap")
LOC_MODES = ("contloc", "initloc", "noloc")
PLANNERS = ("ptgoal", "imggoal")
VLMS = ("astra", "qwen")

DEBUG_PREFIX = "debug-"
DEBUG_METHOD_IDS = ("debug-stop-now",)

V1_METHOD_IDS = (
    "dense-contloc-ptgoal",  # B0
    "sparse-contloc-ptgoal",  # M1
    "dense-contloc-imggoal-astra",  # M2
    "dense-contloc-imggoal-qwen",
    "sparse-contloc-imggoal-astra",  # M3
    "sparse-contloc-imggoal-qwen",
    "dense-initloc-imggoal-astra",  # M4
    "dense-initloc-imggoal-qwen",
    "sparse-initloc-imggoal-astra",  # M5
    "sparse-initloc-imggoal-qwen",
    "nomap-noloc-imggoal-astra",  # M6
    "nomap-noloc-imggoal-qwen",
)
"""Seven official v1 methods with 12 configurations (plan section 2)."""

METHOD_LABELS = {
    "dense-contloc-ptgoal": "B0",
    "sparse-contloc-ptgoal": "M1",
    "dense-contloc-imggoal": "M2",
    "sparse-contloc-imggoal": "M3",
    "dense-initloc-imggoal": "M4",
    "sparse-initloc-imggoal": "M5",
    "nomap-noloc-imggoal": "M6",
}
"""Three-part method names, without VLM suffixes, mapped to report labels B0 / M1-M6."""


def method_label(method_id: str) -> str:
    """Return a short label such as B0 or M2-astra; leave debugging method IDs unchanged."""
    parts = method_id.split("-")
    base = METHOD_LABELS.get("-".join(parts[:3]))
    if base is None:
        return method_id
    return base if len(parts) == 3 else f"{base}-{parts[3]}"


UNLIMITED = -1
"""Quota sentinel allowing self-localization once per frame in contloc mode."""


class MethodIdError(ValueError):
    pass


@dataclass(frozen=True)
class MethodProfile:
    """Resources and decision rules derived from a method ID."""

    method_id: str
    map_variant: Optional[str]  # "dense" | "sparse" | None (no map)
    loc_mode: str  # "contloc" | "initloc" | "noloc" | "debug"
    planner: str  # "ptgoal" | "imggoal" | "debug"
    vlm: Optional[str]  # "astra" | "qwen" | None
    progress_judge: str  # Subgoal judge: "pose" (position), "vlm" (images), or "debug"
    done_judge: str  # Arrival judge: "pose" | "vlm" | "debug"
    goal_loc_quota: int  # Goal-image localization calls per episode
    self_loc_quota: int  # Self-localization calls per episode; UNLIMITED permits one per frame
    uses_truth: bool = False

    @property
    def is_debug(self) -> bool:
        return self.planner == "debug"


def parse_method_id(method_id: str) -> MethodProfile:
    """Parse three method-name fields plus an optional VLM field and validate resources.

    nomap requires noloc and vice versa. ptgoal requires a map and contloc because
    SRU needs coordinates. Only imggoal takes a fourth field, which is mandatory.
    contloc uses pose-based subgoal progression and arrival checks for both
    planners; initloc/noloc with imggoal delegates both checks to the VLM.
    """
    if method_id in DEBUG_METHOD_IDS:
        return MethodProfile(
            method_id, None, "debug", "debug", None, "debug", "debug", 0, 0, uses_truth=True
        )
    parts = method_id.split("-")
    if len(parts) not in (3, 4):
        raise MethodIdError(f"方法名应为 3 或 4 段: {method_id!r}")
    variant, loc, planner = parts[:3]
    vlm = parts[3] if len(parts) == 4 else None
    if variant not in MAP_VARIANTS or loc not in LOC_MODES or planner not in PLANNERS:
        raise MethodIdError(f"未知的段: {method_id!r}")
    if (variant == "nomap") != (loc == "noloc"):
        raise MethodIdError(f"nomap 必须配 noloc(反之亦然): {method_id!r}")
    if planner == "ptgoal" and (vlm is not None or loc != "contloc"):
        raise MethodIdError(f"ptgoal 只能是 *-contloc-ptgoal、不带模型段: {method_id!r}")
    if planner == "imggoal" and vlm not in VLMS:
        raise MethodIdError(f"imggoal 要带模型段 -astra / -qwen: {method_id!r}")
    has_map = variant != "nomap"
    self_quota = {"contloc": UNLIMITED, "initloc": 1, "noloc": 0}[loc]
    return MethodProfile(
        method_id=method_id,
        map_variant=variant if has_map else None,
        loc_mode=loc,
        planner=planner,
        vlm=vlm,
        progress_judge="pose" if loc == "contloc" else "vlm",
        done_judge="pose" if loc == "contloc" else "vlm",
        goal_loc_quota=1 if has_map else 0,
        self_loc_quota=self_quota,
    )


# ---- Clamping and quantization ----


@dataclass(frozen=True)
class ClampReport:
    clamped_vx: bool = False
    clamped_vy: bool = False
    clamped_wz: bool = False
    nonfinite: bool = False  # NaN / infinity replaced with zero
    hold_changed: bool = False  # Hold duration was quantized or clamped

    @property
    def any(self) -> bool:
        return (
            self.clamped_vx or self.clamped_vy or self.clamped_wz or self.nonfinite
        ) or self.hold_changed


def _clip(value: float, lo: float, hi: float) -> Tuple[float, bool, bool]:
    """Return (clipped value, whether clipped, whether nonfinite)."""
    if not math.isfinite(value):
        return 0.0, False, True
    clipped = min(max(value, lo), hi)
    return clipped, clipped != value, False


def quantize_hold(hold_s: float) -> float:
    """Round holds to 0.1 s and clamp to [0.1, 2.0]; use the minimum for nonfinite values."""
    if not math.isfinite(hold_s):
        return HOLD_MIN_S
    steps = int(math.floor(hold_s / HOLD_QUANTUM_S + 0.5 + 1e-9))
    steps = min(max(steps, round(HOLD_MIN_S / HOLD_QUANTUM_S)), round(HOLD_MAX_S / HOLD_QUANTUM_S))
    return round(steps * HOLD_QUANTUM_S, 6)


def clamp_action(action: Action) -> Tuple[Action, ClampReport]:
    """Apply shared interface limits; the server checks again. Zero velocities when done=True."""
    vx, cvx, nvx = _clip(float(action.vx), VX_MIN, VX_MAX)
    vy, cvy, nvy = _clip(float(action.vy), -VY_MAX, VY_MAX)
    wz, cwz, nwz = _clip(float(action.wz), -WZ_MAX, WZ_MAX)
    hold = quantize_hold(float(action.hold_s))
    if action.done:
        vx = vy = wz = 0.0
    report = ClampReport(
        clamped_vx=cvx,
        clamped_vy=cvy,
        clamped_wz=cwz,
        nonfinite=nvx or nvy or nwz,
        hold_changed=not math.isclose(hold, float(action.hold_s), abs_tol=1e-9),
    )
    return replace(action, vx=vx, vy=vy, wz=wz, hold_s=hold), report


# ---- Localization quotas ----


class QuotaLocalizer:
    """Wrap a localizer with method-specific quotas and counters.

    The framework calls :meth:`on_frame` for each observation.
    Goal calls are limited to ``goal_loc_quota`` per episode (one for mapped methods).
    Self calls allow one per frame for contloc, one per episode for initloc and
    zero for noloc, which receives no localizer. Excess calls increment the denial
    counter and return ``LocResult(ok=False, reason="quota")`` without raising.
    """

    def __init__(self, inner: Localizer, profile: MethodProfile) -> None:
        self._inner = inner
        self._profile = profile
        self._frame = -1
        self._self_frame_used = -2
        self.used: Dict[str, int] = {"goal": 0, "self": 0}
        self.ok: Dict[str, int] = {"goal": 0, "self": 0}
        self.denied: Dict[str, int] = {"goal": 0, "self": 0}

    def on_frame(self, step_index: int) -> None:
        self._frame = step_index

    def _allowed(self, purpose: str) -> bool:
        if purpose == "goal":
            return self.used["goal"] < self._profile.goal_loc_quota
        quota = self._profile.self_loc_quota
        if quota == UNLIMITED:
            return self._self_frame_used != self._frame
        return self.used["self"] < quota

    def localize(
        self, rgb_jpeg: bytes, depth_m: Optional[np.ndarray], K: Intrinsics, *, purpose: str
    ) -> LocResult:
        if purpose not in ("goal", "self"):
            raise ValueError(f"purpose 只能是 goal / self: {purpose!r}")
        if not self._allowed(purpose):
            self.denied[purpose] += 1
            return LocResult(False, None, None, "quota")
        self.used[purpose] += 1
        if purpose == "self":
            self._self_frame_used = self._frame
        result = self._inner.localize(rgb_jpeg, depth_m, K, purpose=purpose)
        if result.ok:
            self.ok[purpose] += 1
        return result

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {"used": dict(self.used), "ok": dict(self.ok), "denied": dict(self.denied)}


# ---- Static method-code checks against ground-truth leakage (plan section 6.1) ----

FORBIDDEN_MODULES = (
    "oracle",
    "metrics",
    "sim_env",
    "episodes",
    "localizer",
    "mapview",
    "runner",
    "rundir",
    "sim",
)
"""Repository modules forbidden in methods, including their submodules.

Maps must be accessed exclusively through ctx.map.
"""

FORBIDDEN_STRINGS = ("poses_abs_gt", "sim_truth", "odom_world", "configs/episodes")
"""Forbidden source strings identifying truth files, world odometry or episode directories."""

LEAK_EXEMPT_DIRS = ("debug",)
"""Unscored methods/debug/ implementations with uses_truth=True are exempt."""


def leakage_violations(source: str, filename: str = "<method>") -> List[str]:
    """Return all violations in one method source file; an empty list means none."""
    import ast

    out: List[str] = []
    for needle in FORBIDDEN_STRINGS:
        if needle in source:
            out.append(f"{filename}: 出现禁用字串 {needle!r}")
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        names: List[str] = []  # Normalize to package-relative paths, removing the vnav_benchmark.
        # prefix
        if isinstance(node, ast.Import):
            names = [_inner(alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            base = base if node.level else _inner(base)
            subs = [f"{base}.{a.name}" if base else a.name for a in node.names]
            names = ([base] if base else []) + subs
        elif isinstance(node, ast.Call) and (
            getattr(node.func, "attr", None) == "import_module"
            or getattr(node.func, "id", None) == "__import__"
        ):
            names = [_inner(a.value) for a in node.args if isinstance(a, ast.Constant)]
        for name in names:
            head = name.split(".")[0] if isinstance(name, str) else ""
            if head in FORBIDDEN_MODULES:
                out.append(f"{filename}:{getattr(node, 'lineno', 0)}: 方法层不许 import {name}")
    return out


def _inner(name: object) -> str:
    """Convert an absolute module name to its repository path, or return empty for other packages.
    """
    if not isinstance(name, str):
        return ""
    if name == "vnav_benchmark" or name.startswith("vnav_benchmark."):
        return name[len("vnav_benchmark.") :] if "." in name else ""
    return ""
