"""Integrate the real runner and image-goal method with fake simulation and VLM backends.

Run complete episodes and validate run.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import pytest
from vnav_fake_client import FakeVlmClient
from vnav_route_fixtures import DEFAULT_ROUTES, make_route_episode_set
from vnav_t_fixtures import CAMERA_FORWARD_M, make_map

from vnav_benchmark import episodes as ep
from vnav_benchmark import mapview as mv
from vnav_benchmark import oracle
from vnav_benchmark.budget import BudgetGuard
from vnav_benchmark.methods.imggoal_vlm import ImgGoalVlmMethod
from vnav_benchmark.report.results_io import row_from_run
from vnav_benchmark.rundir import read_steps, validate_run
from vnav_benchmark.runner import BenchConfig, Job, KinematicProvider, RunContext, run_episode
from vnav_benchmark.vlm.client import VlmFatalError

FAST = BenchConfig(budget_s=20.0)


@pytest.fixture(scope="module")
def grid() -> oracle.TruthGrid:
    return oracle.build_truth_grid()


@pytest.fixture(scope="module")
def dev_set(grid: oracle.TruthGrid, tmp_path_factory: pytest.TempPathFactory) -> ep.EpisodeSet:
    """Build one synthetic route whose first/last keyframes define start/goal and the goal photo."""
    root = tmp_path_factory.mktemp("eps")
    return make_route_episode_set(root, grid, "env0_dev_vV", DEFAULT_ROUTES[:1])


class _TinyMaps:
    """Use the small L-shaped map as dense/sparse data to test wiring, not route quality."""

    def __init__(self, root: Path) -> None:
        src = make_map(root / "src")
        mv.build_sanitized_view(src, root / "view")
        self.handle = mv.SanitizedMap(root / "view", "tiny", "dense", "trav", CAMERA_FORWARD_M)

    def get(self, route: str, variant: str) -> Tuple[Any, List[float], dict]:
        eye = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        info = {"map_id": "tiny", "variant": variant, "role": "test", "sha256": "x",
                "edge_type": "trav"}  # fmt: skip
        return self.handle, eye, info


def _reply(done: bool = False, reached: bool = False) -> str:
    return json.dumps({"observation": "corridor", "target_visible": True,
                       "subgoal_reached": reached, "done": done, "memory": "m",
                       "vx": 0.4, "vy": 0.0, "wz": 0.2})  # fmt: skip


def _ctx(root: Path, es: ep.EpisodeSet, grid: oracle.TruthGrid, script: List[Any],
         maps: Any = None) -> Tuple[RunContext, List[FakeVlmClient]]:  # fmt: skip
    clients: List[FakeVlmClient] = []

    def factory(method_id: str) -> ImgGoalVlmMethod:
        def client_factory(endpoint: str) -> FakeVlmClient:
            clients.append(FakeVlmClient(script, endpoint))
            return clients[-1]

        return ImgGoalVlmMethod(method_id, client_factory=client_factory)

    ctx = RunContext(root, es, KinematicProvider(grid), maps, BudgetGuard(root / "ledger.jsonl"),
                     bench=FAST, grid=grid, method_factory=factory)  # fmt: skip
    return ctx, clients


def test_nomap_m6_episode_through_runner(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    token = dev_set.tokens()[0]
    ctx, clients = _ctx(tmp_path, dev_set, grid, [_reply()] * 5 + [_reply(done=True)])
    doc = run_episode(Job("nomap-noloc-imggoal-astra", token, 0), ctx)
    validate_run(doc)
    assert doc["status"] == "scored" and doc["outcome"]["termination"] == "done"
    assert doc["timing"]["decisions"] == 6 and len(clients[0].requests) == 6
    assert doc["usage"]["calls"] == 6 and doc["usage"]["billed_usd"] > 0
    summary = doc["method_stats"]["summary"]
    assert summary["endpoint"]["endpoint"] == "astra_flex" and summary["termination"] == "done"
    run = tmp_path / "runs" / "nomap-noloc-imggoal-astra" / token / "r0"
    steps = read_steps(run / "steps.jsonl")
    assert steps[0]["applied"]["hold_s"] == 1.0 and steps[-1]["action"]["done"]
    assert steps[0]["debug"]["phase"] == "FINAL_APPROACH"
    assert (run / "method" / "vlm" / "calls.jsonl").is_file()
    assert (run / "method" / "vlm" / "summary.json").is_file()
    row = row_from_run(doc)  # Reports can read the runner's run.json directly.
    assert row.calls == 6 and row.endpoint == "astra_flex" and row.code == "M6"


def test_initloc_m4_episode_gets_map_subgoals(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    token = dev_set.tokens()[0]
    script = [_reply(reached=True)] * 12 + [_reply(done=True)] * 3
    ctx, clients = _ctx(tmp_path, dev_set, grid, script, _TinyMaps(tmp_path / "maps"))
    doc = run_episode(Job("dense-initloc-imggoal-qwen", token, 0), ctx)
    validate_run(doc)
    summary = doc["method_stats"]["summary"]
    assert summary["route"]["kf_ids"] and summary["subgoal_advances"] >= 1
    assert doc["method_stats"]["loc"]["used"] == {"goal": 1, "self": 1}
    assert doc["settings"]["map"]["map_id"] == "tiny"
    labels = [p.text for p in clients[0].requests[0].parts if hasattr(p, "text")]
    assert any(t.startswith("CURRENT SUBGOAL IMAGE") for t in labels)


def test_fatal_provider_error_is_unscored(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    from vnav_benchmark.interfaces import FatalInfraError

    token = dev_set.tokens()[0]
    ctx, _ = _ctx(tmp_path, dev_set, grid, [VlmFatalError("credit_balance_exhausted")])
    with pytest.raises(FatalInfraError):
        run_episode(Job("nomap-noloc-imggoal-astra", token, 0), ctx)
    assert not list((tmp_path / "runs").rglob("run.json"))
