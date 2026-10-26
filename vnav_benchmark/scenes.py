"""Read the frozen evaluator scene data without importing scene-building tools.

These files contain evaluator ground truth. Navigation methods must not import
this module or read data/scenes; only the evaluator and simulator identity use it.
"""

import json
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import numpy as np

from vnav_benchmark.paths import repository_root

SCENES = (
    "env0_office_single_floor",
    "env1_garden_outdoor",
    "env3_office_two_floors",
)


def _scene_directory(scene: str) -> Path:
    if scene not in SCENES:
        raise ValueError(f"Unknown frozen scene: {scene!r}")
    return repository_root() / "data" / "scenes" / scene


def _readonly(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_readonly(item) for item in value)
    return value


@lru_cache(maxsize=len(SCENES))
def scene_data(scene: str) -> Mapping[str, Any]:
    """Return immutable geometry, floor topology and original layout identity."""
    payload = json.loads((_scene_directory(scene) / "scene.json").read_text(encoding="utf-8"))
    if payload["schema"] != "vnav-scene-v1" or payload["scene"] != scene:
        raise ValueError(f"Invalid frozen scene metadata: {scene!r}")
    return _readonly(payload)


def truth_arrays(scene: str, floor: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the exported default grid, returning independent writable arrays.

    NumPy pickle loading is disabled; the archive contains only boolean arrays.
    """
    with np.load(_scene_directory(scene) / "truth.npz", allow_pickle=False) as archive:
        return archive[floor + "_free"].copy(), archive[floor + "_raw_blocked"].copy()
