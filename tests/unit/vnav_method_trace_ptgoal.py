"""Replay point-goal B0/M1 methods using fake simulation and a kinematic SRU substitute.

Record per-step actions/debug fields plus full episode summaries, steps.jsonl, route.json and
summary.json. Pre-refactor golden traces establish exact behavior across nav_template adoption.

Generate a golden from a chosen checkout with:
python tests/unit/vnav_method_trace_ptgoal.py <output-json>
"""

from __future__ import annotations

import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import test_vnav_ptgoal_sru as T  # noqa: E402
from vnav_t_fixtures import CAMERA_FORWARD_M, L_CORRIDOR, make_map  # noqa: E402

from frontier_nav_bench import fairness  # noqa: E402
from frontier_nav_bench import mapview as mv  # noqa: E402
from frontier_nav_bench.interfaces import EpisodeSpec, MethodContext, Pose2D  # noqa: E402
from frontier_nav_bench.localizer import OracleLocalizer  # noqa: E402
from frontier_nav_bench.methods import ptgoal_sru as pg  # noqa: E402
from frontier_nav_bench.sim.backend import KinematicBackend  # noqa: E402
from frontier_nav_bench.sim.ctrl_client import InProcessClient  # noqa: E402
from frontier_nav_bench.sim.ctrl_server import CtrlServer  # noqa: E402
from frontier_nav_bench.sim_env import SimEnv  # noqa: E402

# Call act() again after declaring arrival to test repeated calls after termination.
AFTER_DONE_CALLS = 2


def _act_row(a: Any) -> Dict[str, Any]:
    return {"cmd": [a.vx, a.vy, a.wz, a.hold_s], "done": a.done, "note": a.note,
            "debug": json.loads(json.dumps(a.debug, default=str))}  # fmt: skip


def _run(
    tmp: Path,
    method_id: str,
    trav_edges: Sequence[Tuple[int, int]] = (),
    budget_s: float = 120.0,
    poses: Sequence[Tuple[float, float, float]] = L_CORRIDOR,
) -> Dict[str, Any]:
    """Run as in test_vnav_ptgoal_sru.run_episode, additionally recording actions step by step."""
    src = make_map(tmp / "src", poses=poses, trav_edges=trav_edges)
    mv.build_sanitized_view(src, tmp / "view")
    profile = fairness.parse_method_id(method_id)
    handle = mv.SanitizedMap(tmp / "view", "m", profile.map_variant, "trav", CAMERA_FORWARD_M)
    backend = KinematicBackend(
        occupancy=T.l_corridor_occupancy(), origin_xy=T.ORIGIN, cell_m=T.CELL_M
    )
    client = InProcessClient(CtrlServer(backend, "inproc://trace"))
    client.hello()
    env = SimEnv(client, budget_s)
    obs = env.start(Pose2D(*T.START))
    oracle = OracleLocalizer(env.truth_pose, Pose2D(*T.GOAL), T.IDENTITY16)
    loc = fairness.QuotaLocalizer(oracle, profile)
    config = yaml.safe_load((T.REPO / "configs/methods" / f"{method_id}.yaml").read_text())
    out_dir = tmp / "method"
    ctx = MethodContext(
        profile=profile, config=config, map=handle, localizer=loc,
        rng=np.random.default_rng(0), log=logging.getLogger("trace"), usage=T._Sink(),
        artifact_dir=out_dir,
    )  # fmt: skip
    controller = T.UnicycleController()
    method = pg.PtGoalSruMethod(method_id, controller_factory=lambda _log: controller)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    depth = np.zeros((48, 64), np.float32)
    spec = EpisodeSpec("tok", 0, "env0", img, b"g", depth, (1.0,), budget_s)
    loc.on_frame(obs.step_index)
    method.reset(spec, obs, ctx)
    actions = []
    extra = 0
    while not env.out_of_budget:
        loc.on_frame(obs.step_index)
        raw = method.act(obs)
        actions.append(_act_row(raw))
        action, _ = fairness.clamp_action(raw)
        if action.done:
            extra += 1
            if extra > AFTER_DONE_CALLS:
                break
            continue
        obs, _ = env.step(action)
    method.close()
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(out_dir.glob("*.json*"))}
    return {
        "actions": actions,
        "summary": json.loads(json.dumps(method.summary(), default=str)),
        "files": files,
        "controller": {"resets": controller.resets, "closed": controller.closed,
                       "targets": [list(t) for t in controller.targets]},  # fmt: skip
    }


def scenario_b0_reach_align(tmp: Path) -> Dict[str, Any]:
    return _run(tmp, "dense-contloc-ptgoal")


def scenario_m1_reach_align(tmp: Path) -> Dict[str, Any]:
    return _run(tmp, "sparse-contloc-ptgoal")


def scenario_b0_stall_replan(tmp: Path) -> Dict[str, Any]:
    """Keyframe 9 lies in a wall; shortcut 1->9->8 causes a stall, edge blocking and replanning."""
    poses = tuple(L_CORRIDOR) + ((1.0, 2.0, math.pi / 2),)
    return _run(tmp, "dense-contloc-ptgoal", trav_edges=[(1, 9), (9, 8)], poses=poses)


def scenario_b0_budget_out(tmp: Path) -> Dict[str, Any]:
    return _run(tmp, "dense-contloc-ptgoal", budget_s=3.0)


SCENARIOS = {
    "b0_reach_align": scenario_b0_reach_align,
    "m1_reach_align": scenario_m1_reach_align,
    "b0_stall_replan": scenario_b0_stall_replan,
    "b0_budget_out": scenario_b0_budget_out,
}


def run_all() -> Dict[str, Any]:
    out = {}
    for name, fn in SCENARIOS.items():
        with tempfile.TemporaryDirectory() as d:
            out[name] = fn(Path(d))
    return out


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(run_all(), indent=1, sort_keys=True))
    print("wrote", sys.argv[1])
