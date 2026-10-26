"""Method registry mapping method IDs to lazily imported factories.

Implementations live in ptgoal_sru, imggoal_vlm and debug/. Entries contain only
``module:factory`` paths so each method loads its own dependencies. Factories have
signature ``make(method_id: str) -> Method`` and create one object per episode
(plan section 6.8).
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict

from frontier_nav_bench.fairness import DEBUG_METHOD_IDS, V1_METHOD_IDS, parse_method_id
from frontier_nav_bench.interfaces import Method

_PTGOAL = "frontier_nav_bench.methods.ptgoal_sru:make"
_IMGGOAL = "frontier_nav_bench.methods.imggoal_vlm:make"

FACTORIES: Dict[str, str] = {
    **{mid: _PTGOAL for mid in V1_METHOD_IDS if mid.endswith("-ptgoal")},
    **{mid: _IMGGOAL for mid in V1_METHOD_IDS if "-imggoal-" in mid},
    "debug-stop-now": "frontier_nav_bench.methods.debug.stop_now:make",
}


def registered_ids() -> tuple:
    return tuple(FACTORIES)


def make_method(method_id: str) -> Method:
    """Validate the ID with parse_method_id, import its registered factory and create a method."""
    if method_id not in FACTORIES:
        raise KeyError(f"未注册的方法: {method_id!r}")
    profile = parse_method_id(method_id)
    module_name, _, func_name = FACTORIES[method_id].partition(":")
    factory: Callable[[str], Method] = getattr(importlib.import_module(module_name), func_name)
    method = factory(method_id)
    if method.method_id != method_id or bool(method.uses_truth) != profile.uses_truth:
        raise ValueError(f"{method_id}: 工厂返回的 method_id / uses_truth 和名字对不上")
    return method


assert set(DEBUG_METHOD_IDS) <= set(FACTORIES)
