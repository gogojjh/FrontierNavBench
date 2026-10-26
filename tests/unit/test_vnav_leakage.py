"""Prevent truth leakage: non-debug methods cannot import evaluation/truth modules or truth strings.
"""

from pathlib import Path

import pytest

from vnav_benchmark.fairness import LEAK_EXEMPT_DIRS, leakage_violations

METHODS_DIR = Path(__file__).resolve().parents[2] / "vnav_benchmark" / "methods"


def _method_files() -> list:
    out = []
    for path in sorted(METHODS_DIR.rglob("*.py")):
        rel = path.relative_to(METHODS_DIR)
        if rel.parts and rel.parts[0] in LEAK_EXEMPT_DIRS:
            continue
        out.append(path)
    return out


def test_methods_tree_clean() -> None:
    bad = []
    for path in _method_files():
        bad += leakage_violations(path.read_text(encoding="utf-8"), str(path.name))
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize(
    "src",
    [
        "import vnav_benchmark.oracle\n",
        "from vnav_benchmark import metrics\n",
        "from vnav_benchmark.sim_env import SimEnv\n",
        "from ..episodes import load\n",
        "from .. import localizer\n",
        "from vnav_benchmark.sim.ctrl_client import CtrlClient\n",
        "import importlib\nimportlib.import_module('vnav_benchmark.runner')\n",
        "x = __import__('vnav_benchmark.mapview')\n",
        "p = 'maps/x/poses_abs_gt.txt'\n",
        "k = obs['odom_world']\n",
        "d = 'configs/episodes/env0_test_v1'\n",
        "s = 'sim_truth.json'\n",
    ],
)
def test_detector_catches(src: str) -> None:
    assert leakage_violations(src)


@pytest.mark.parametrize(
    "src",
    [
        "import numpy as np\nfrom vnav_benchmark.interfaces import Action\n",
        "from vnav_benchmark.fairness import clamp_action\n",
        "from .base_topo import TopoNavCore\n",
        "from . import sru\n",
        "import metrics_lib\n",  # Not a module from this repository
    ],
)
def test_detector_allows(src: str) -> None:
    assert leakage_violations(src) == []
