"""Replay scripted fake-model/fake-simulation scenarios to check navigation-wrapper equivalence.

Record actions, debug fields, request summaries and episode summaries step by
step. Golden output comes from the code before refactoring and verifies exact
behavior preservation. To regenerate for a checkout, run from that repository:
``python tests/unit/vnav_method_trace.py <output.json>``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import test_vnav_imggoal_vlm as T  # noqa: E402

from frontier_nav_bench.vlm.client import ImagePart  # noqa: E402


def _req_digest(req: Any) -> str:
    parts = [
        {"img": hashlib.sha256(p.jpeg).hexdigest(), "w": p.width, "h": p.height}
        if isinstance(p, ImagePart)
        else {"t": p.text, "bp": p.cache_breakpoint}
        for p in req.parts
    ]
    blob = json.dumps([req.system, parts, req.schema_name, req.schema], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _act_row(a: Any) -> Dict[str, Any]:
    return {"cmd": [a.vx, a.vy, a.wz, a.hold_s], "done": a.done, "note": a.note,
            "debug": json.loads(json.dumps(a.debug, default=str))}  # fmt: skip


def _trace(ep: Any, actions: List[Any]) -> Dict[str, Any]:
    summary = json.loads(json.dumps(ep.method.summary(), default=str))
    return {
        "actions": [_act_row(a) for a in actions],
        "requests": [_req_digest(r) for r in ep.fake.requests],
        "summary": summary,
    }


def scenario_m2_pose(tmp: Path) -> Dict[str, Any]:
    ep = T._pose_episode(tmp, [T._reply(subgoal_reached=True, done=True)] * 40)
    path = [(0.5, 0.0, 0.0), (1.2, 0.0, 0.0), (2.2, 0.0, 0.0), (3.2, 0.0, 0.0), (4.0, 0.3, 1.57),
            (4.0, 1.5, 1.57), (4.0, 2.5, 1.57), (4.0, 3.4, 1.57), (4.0, 3.9, 1.57),
            (4.0, 4.0, math.pi / 2), (4.0, 4.0, math.pi / 2), (4.0, 4.0, math.pi / 2)]  # fmt: skip
    ep.method.reset(ep.spec, ep.world.observation(), ep.ctx)
    actions = []
    for step, pose in enumerate(path):  # Call once more after arrival to verify behavior after
    # episode completion
        ep.world.pose = T.Pose2D(*pose)
        ep.world.step_index = step
        ep.loc.on_frame(step)
        actions.append(ep.method.act(ep.obs()))
    ep.method.close()
    return _trace(ep, actions)


def _run_world(ep: Any, n: int) -> List[Any]:
    ep.method.reset(ep.spec, ep.world.observation(), ep.ctx)
    actions = []
    for _ in range(n):
        a = ep.method.act(ep.obs())
        actions.append(a)
        ep.world.step(a.vx, a.vy, a.wz, a.hold_s)
        if a.done:
            break
    ep.method.close()
    return actions


def scenario_m4_vlm(tmp: Path) -> Dict[str, Any]:
    script = [T._reply(subgoal_reached=i % 3 == 2, vx=0.3 + 0.01 * i) for i in range(12)]
    script += ["not json", T._reply(), T._reply(done=True)]
    ep = T._Episode(tmp, "dense-initloc-imggoal-astra", script)
    return _trace(ep, _run_world(ep, 20))


def scenario_m6_nomap(tmp: Path) -> Dict[str, Any]:
    script = [T._reply(wz=0.5)] * 6 + [T._reply(done=True)]
    ep = T._Episode(tmp, "nomap-noloc-imggoal-qwen", script)
    return _trace(ep, _run_world(ep, 10))


SCENARIOS = {"m2_pose": scenario_m2_pose, "m4_vlm": scenario_m4_vlm, "m6_nomap": scenario_m6_nomap}


def run_all() -> Dict[str, Any]:
    out = {}
    for name, fn in SCENARIOS.items():
        with tempfile.TemporaryDirectory() as d:
            out[name] = fn(Path(d))
    return out


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(run_all(), indent=1, sort_keys=True))
    print("wrote", sys.argv[1])
