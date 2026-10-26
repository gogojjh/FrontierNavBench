"""Test campaigns with real debug methods on fake simulation backends.

Cover retries, resume, interruption, budgets, circuit breaking and coverage reports.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import pytest
import yaml
from vnav_route_fixtures import DEFAULT_ROUTES, make_route_episode_set

from vnav_benchmark import episodes as ep
from vnav_benchmark import oracle, runner
from vnav_benchmark.budget import BudgetGuard
from vnav_benchmark.interfaces import Action, EpisodeSpec, FatalInfraError, UsageRecord
from vnav_benchmark.rundir import read_steps, read_trace, validate_run
from vnav_benchmark.runner import (
    BenchConfig,
    Campaign,
    CampaignStopped,
    Job,
    KinematicProvider,
    RunContext,
    job_order,
    run_episode,
)

FAST = BenchConfig(budget_s=60.0)


@pytest.mark.parametrize("scene, expected", [
    ("env1_garden_outdoor", 90.0),
    ("env3_office_two_floors", 120.0),
    ("env0_office_single_floor", 90.0),
    ("unlisted_scene", 90.0),
])
def test_scene_budget_from_config(scene: str, expected: float) -> None:
    bench = BenchConfig.load()
    assert bench.budget_s_for(scene) == expected
    assert bench.settings(scene=scene)["budget_s"] == expected
    assert bench.settings()["budget_s"] == 90.0


def test_legacy_config_without_scene_budgets(tmp_path: Path) -> None:
    data = yaml.safe_load(runner.BENCH_CONFIG.read_text(encoding="utf-8"))
    data["run"].pop("budget_s_by_scene", None)
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    bench = BenchConfig.load(path)
    assert bench.budget_s_by_scene == {}
    assert bench.budget_s_for("env1_garden_outdoor") == 90.0
    assert BenchConfig(budget_s=5.0).budget_s_for("env1_garden_outdoor") == 5.0


@pytest.fixture(scope="module")
def grid() -> oracle.TruthGrid:
    return oracle.build_truth_grid()


@pytest.fixture(scope="module")
def dev_set(grid: oracle.TruthGrid, tmp_path_factory: pytest.TempPathFactory) -> ep.EpisodeSet:
    """Freeze two synthetic routes sharing a start but differing in goals, only for campaign tests.
    """
    root = tmp_path_factory.mktemp("eps")
    return make_route_episode_set(root, grid, "env0_dev_vT", DEFAULT_ROUTES[:2])


def _ctx(root: Path, es: ep.EpisodeSet, grid: oracle.TruthGrid, **kw) -> RunContext:  # type: ignore[no-untyped-def]
    provider = kw.pop("sim", None) or KinematicProvider(grid)
    return RunContext(root, es, provider, None, BudgetGuard(root / "ledger.jsonl"),
                      bench=kw.pop("bench", FAST), grid=grid, **kw)  # fmt: skip


def test_runner_records_artifacts_and_reuses_completed_run(
    tmp_path: Path, dev_set: ep.EpisodeSet, grid: oracle.TruthGrid,
) -> None:
    token = dev_set.tokens()[0]
    doc = run_episode(Job("debug-stop-now", token, 0), _ctx(tmp_path, dev_set, grid))
    validate_run(doc)
    assert doc["status"] == "excluded_debug" and doc["outcome"]["termination"] == "done"
    assert not doc["metrics"]["success"] and doc["metrics"]["plr"] is None
    run = tmp_path / "runs" / "debug-stop-now" / token / "r0"
    steps = read_steps(run / "steps.jsonl")
    trace = read_trace(run / "trace.csv")
    assert len(steps) == doc["timing"]["decisions"] and steps[-1]["action"]["done"]
    times = [r["t_sim"] for r in trace]
    assert times == sorted(set(times)) and times[0] == 0.0
    ego = sorted((run / "frames" / "ego").iterdir())
    # Frames match the 10 Hz truth trace one-to-one without repeated frames.
    assert len(ego) == len(trace)
    assert (run / "goal" / "goal.jpg").is_file() and doc["infra"]["attempts"] == 1
    # Re-running reads the existing run.json instead of executing again.
    again = run_episode(Job("debug-stop-now", token, 0), _ctx(tmp_path, dev_set, grid))
    assert again == json.loads((run / "run.json").read_text())


def test_runner_records_done_and_timeout(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    token = dev_set.tokens()[1]
    stop = run_episode(Job("debug-stop-now", token, 0), _ctx(tmp_path, dev_set, grid))
    assert stop["outcome"]["termination"] == "done" and not stop["metrics"]["success"]
    assert stop["outcome"]["premature_done"] and stop["timing"]["decisions"] == 1
    walk = run_episode(
        Job("debug-stop-now", dev_set.tokens()[0], 0),
        _ctx(tmp_path, dev_set, grid, bench=BenchConfig(budget_s=5.0),
             method_factory=lambda _mid: _Spender(0.0)),
    )
    assert walk["outcome"]["termination"] == "timeout"
    assert walk["timing"]["sim_s"] == pytest.approx(5.0)


@pytest.mark.parametrize("scene, expected", [
    ("env1_garden_outdoor", 90.0),
    ("env3_office_two_floors", 120.0),
])
def test_attempt_uses_scene_budget_and_records_it(
    tmp_path: Path, dev_set: ep.EpisodeSet, grid: oracle.TruthGrid,
    monkeypatch: pytest.MonkeyPatch, scene: str, expected: float,
) -> None:
    es = replace(dev_set, payload={**dev_set.payload, "scene": scene})
    if scene == oracle.SCENE_ENV3:
        es = replace(es, records={
            token: replace(rec, start_floor="f1", goal_floor="f1")
            for token, rec in es.records.items()
        })
    env_budgets: List[float] = []
    spec_budgets: List[float] = []
    original_spec = ep.EpisodeSet.spec

    class RecordingEnv(runner.SimEnv):
        def __init__(
            self, link: runner.SimLink, budget_s: float, want_third: bool = False,
        ) -> None:
            env_budgets.append(budget_s)
            super().__init__(link, budget_s, want_third=want_third)

    def recording_spec(
        self: ep.EpisodeSet, token: str, repeat_index: int, budget_s: float,
    ) -> EpisodeSpec:
        spec = original_spec(self, token, repeat_index, budget_s)
        spec_budgets.append(spec.budget_s)
        return spec

    monkeypatch.setattr(runner, "SimEnv", RecordingEnv)
    monkeypatch.setattr(ep.EpisodeSet, "spec", recording_spec)
    ctx = _ctx(tmp_path, es, grid, bench=BenchConfig.load())
    campaign = Campaign(ctx, ("debug-stop-now",), (es.tokens()[0],), 1)
    campaign.run()

    assert env_budgets == spec_budgets == [expected]
    run_path = next((tmp_path / "runs").rglob("run.json"))
    doc = json.loads(run_path.read_text(encoding="utf-8"))
    assert doc["settings"]["budget_s"] == expected
    assert doc["outcome"]["termination"] == "done"
    header = json.loads((tmp_path / "campaign.json").read_text(encoding="utf-8"))
    assert header["bench"]["budget_s"] == expected

    sim = KinematicProvider(grid)
    sim.fail_next = ["unavailable"] * ctx.bench.max_attempts
    failed = run_episode(
        Job("debug-stop-now", es.tokens()[0], 1),
        _ctx(tmp_path, es, grid, bench=ctx.bench, sim=sim),
    )
    assert failed["status"] == "unscored_infra"
    assert failed["settings"]["budget_s"] == expected


def test_infra_retry_then_scored(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    sim = KinematicProvider(grid)
    sim.fail_next = ["desktop gone", "still gone"]
    doc = run_episode(
        Job("debug-stop-now", dev_set.tokens()[0], 0), _ctx(tmp_path, dev_set, grid, sim=sim)
    )
    assert doc["infra"]["attempts"] == 3 and len(doc["infra"]["errors"]) == 2
    attempts = tmp_path / "runs" / "debug-stop-now" / dev_set.tokens()[0] / "r0" / "attempts"
    assert sorted(p.name for p in attempts.iterdir()) == ["1", "2"]


def test_infra_gives_up_after_three(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    sim = KinematicProvider(grid)
    sim.fail_next = ["x"] * 5
    doc = run_episode(
        Job("debug-stop-now", dev_set.tokens()[0], 0), _ctx(tmp_path, dev_set, grid, sim=sim)
    )
    assert doc["status"] == "unscored_infra" and doc["outcome"] is None and sim.opened == 3


class _Interrupting:
    """Simulate process termination on the third act() call.

    BaseException propagates through the runner and leaves a partial run directory.
    """

    method_id = "debug-stop-now"
    uses_truth = True

    def __init__(self) -> None:
        self.n = 0

    def attach_truth(self, truth) -> None:  # type: ignore[no-untyped-def]
        pass

    def reset(self, spec, obs, ctx) -> None:  # type: ignore[no-untyped-def]
        pass

    def act(self, obs):  # type: ignore[no-untyped-def]
        self.n += 1
        if self.n == 3:
            raise KeyboardInterrupt
        return Action(0.3, 0.0, 0.0, 1.0)

    def close(self) -> None:
        pass


def test_kill_midway_then_resume_no_duplicates(
    tmp_path: Path, dev_set: ep.EpisodeSet, grid
) -> None:  # type: ignore[no-untyped-def]
    methods = ("debug-stop-now",)
    tokens = tuple(dev_set.tokens())
    ctx = _ctx(tmp_path, dev_set, grid, method_factory=lambda mid: _Interrupting())
    with pytest.raises(KeyboardInterrupt):
        Campaign(ctx, methods, tokens, 2).run()
    # Resume settings must match the first run, otherwise raise CampaignStopped.
    ctx2 = _ctx(tmp_path, dev_set, grid)
    cov = Campaign(ctx2, methods, tokens, 2).run()
    assert cov["done"] == cov["total"] == 4
    runs = list((tmp_path / "runs").rglob("run.json"))
    assert len(runs) == 4  # Exactly one result per campaign cell.
    killed = [p for p in (tmp_path / "runs").rglob("attempts") if p.is_dir()]
    assert len(killed) == 1 and (killed[0] / "1" / "steps.jsonl").is_file()
    doc = json.loads((killed[0].parent / "run.json").read_text())
    assert (
        doc["attempt"] == 2
        and len(read_steps(killed[0].parent / "steps.jsonl")) == doc["timing"]["decisions"]
    )


def test_campaign_params_locked(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    tokens = tuple(dev_set.tokens())
    Campaign(_ctx(tmp_path, dev_set, grid), ("debug-stop-now",), tokens, 1).run()
    with pytest.raises(CampaignStopped):
        Campaign(_ctx(tmp_path, dev_set, grid), ("debug-stop-now",), tokens, 2).run()


def test_consecutive_infra_breaker(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    sim = KinematicProvider(grid)
    sim.fail_next = ["down"] * 100
    alerts: List[str] = []
    camp = Campaign(_ctx(tmp_path, dev_set, grid, sim=sim), ("debug-stop-now",),
                    tuple(dev_set.tokens()), 2, alert=alerts.append)  # fmt: skip
    with pytest.raises(CampaignStopped):
        camp.run()
    cov = json.loads((tmp_path / "coverage.json").read_text())
    assert cov["by_method"]["debug-stop-now"].get("unscored_infra", 0) >= 1 and alerts
    assert sum(c["status"] == "unscored_infra" for c in cov["cells"]) == 3


class _Spender:
    method_id = "debug-stop-now"
    uses_truth = True

    def __init__(self, per_call: float, fatal_at: int = 0) -> None:
        self.per_call, self.fatal_at, self.n = per_call, fatal_at, 0

    def attach_truth(self, truth) -> None:  # type: ignore[no-untyped-def]
        pass

    def reset(self, spec, obs, ctx) -> None:  # type: ignore[no-untyped-def]
        self.usage = ctx.usage

    def act(self, obs):  # type: ignore[no-untyped-def]
        self.n += 1
        if self.fatal_at and self.n == self.fatal_at:
            raise FatalInfraError("credit_balance_exhausted")
        self.usage.add(
            UsageRecord(
                "openai", "gpt-6-astra", billed_usd=self.per_call, list_usd=2 * self.per_call
            )
        )
        return Action(0.2, 0.0, 0.0, 1.0)

    def close(self) -> None:
        pass


def test_cost_guard_and_ledger(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    ctx = _ctx(tmp_path, dev_set, grid, method_factory=lambda mid: _Spender(2.0))
    doc = run_episode(Job("debug-stop-now", dev_set.tokens()[0], 0), ctx)
    assert doc["outcome"]["termination"] == "cost_guard" and doc["status"] == "excluded_debug"
    assert doc["usage"]["billed_usd"] == pytest.approx(6.0) and doc["usage"][
        "list_usd"
    ] == pytest.approx(12.0)
    guard = BudgetGuard(tmp_path / "ledger.jsonl")  # The ledger persists across reopening.
    assert guard.spent("openai") == pytest.approx(6.0)


def test_budget_guard_blocks_new_episode(tmp_path: Path) -> None:
    guard = BudgetGuard(tmp_path / "ledger.jsonl", caps={"openai": 10.0, "dashscope": 30.0})
    assert guard.can_start(None)[0]
    # 0 + 180 times the $0.042 prior is $7.56, within the $10 limit.
    assert guard.can_start("openai")[0]
    guard.record({"provider": "openai", "billed_usd": 3.0})
    ok, why = guard.can_start("openai")
    # 3 + 180 times 3.0 exceeds 10; use the larger estimate when samples are insufficient.
    assert not ok and "上限" in why
    assert guard.can_start("dashscope")[0]  # Maintain separate provider ledgers.


def test_campaign_skips_blocked_and_reports(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    ctx = _ctx(tmp_path, dev_set, grid)
    ctx.guard = BudgetGuard(tmp_path / "ledger.jsonl", caps={"openai": 1.0, "dashscope": 30.0})
    camp = Campaign(ctx, ("debug-stop-now",), tuple(dev_set.tokens()), 1)
    camp.methods = ("debug-stop-now",)
    cov = camp.run()
    # Debug methods are unscored and have no paired cell.
    assert cov["done"] == 2 and cov["paired_cells"] == []


def test_fatal_stops_campaign(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    alerts: List[str] = []
    ctx = _ctx(tmp_path, dev_set, grid, method_factory=lambda mid: _Spender(0.01, fatal_at=2))
    with pytest.raises(CampaignStopped):
        Campaign(ctx, ("debug-stop-now",), tuple(dev_set.tokens()), 1, alert=alerts.append).run()
    assert alerts and "credit_balance_exhausted" in alerts[0]
    # No result is written; attempts retain diagnostic evidence.
    assert not list((tmp_path / "runs").rglob("run.json"))
    assert list((tmp_path / "runs").rglob("attempts"))


def test_job_order_pairs_first_and_rotates() -> None:
    jobs = job_order(["A", "B", "C"], ["p1", "p2"], 2)
    assert [j.method_id for j in jobs[:6]] == ["A", "B", "C", "B", "C", "A"]
    assert [j.token for j in jobs[:6]] == ["p1"] * 3 + ["p2"] * 3
    assert {j.repeat for j in jobs[6:]} == {1} and len(jobs) == 12


def test_test_set_refuses_debug(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    as_test = replace(dev_set, name="env0_test_vT")
    with pytest.raises(ep.EpisodeSetError):
        run_episode(Job("debug-stop-now", dev_set.tokens()[0], 0), _ctx(tmp_path, as_test, grid))


def test_one_process_per_campaign(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    import fcntl

    tmp_path.mkdir(exist_ok=True)
    # Simulate another process owning the run lock.
    with open(tmp_path / ".campaign.lock", "a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(CampaignStopped):
            Campaign(
                _ctx(tmp_path, dev_set, grid), ("debug-stop-now",), tuple(dev_set.tokens()), 1
            ).run()
    cov = Campaign(
        _ctx(tmp_path, dev_set, grid), ("debug-stop-now",), tuple(dev_set.tokens()), 1
    ).run()
    assert cov["done"] == 2  # Run normally once the lock is released.


def test_v1_plan_and_order(dev_set: ep.EpisodeSet) -> None:
    """The v1 schedule runs all 12 configurations once per route and B0/M1 three times."""
    from vnav_benchmark import fairness
    from vnav_benchmark.runner import uniform_plan, v1_plan

    plan = v1_plan(dev_set)
    jobs = job_order(fairness.V1_METHOD_IDS, dev_set.tokens(), 1, plan)
    by: Dict[str, list] = {}
    for j in jobs:
        by.setdefault(j.method_id, []).append(j)
    n = len(dev_set.tokens())
    assert len(by) == 12 and len(jobs) == 10 * n + 2 * 3 * n
    assert len(by["dense-contloc-ptgoal"]) == n * 3
    assert all(len(by[m]) == n for m in fairness.V1_METHOD_IDS if "-imggoal-" in m)
    assert all(j.method_id.endswith("ptgoal") for j in jobs if j.repeat > 0)
    assert job_order(["A", "B"], ["p"], 2) == job_order(
        ["A", "B"], ["p"], 2, uniform_plan(["A", "B"], ["p"], 2)
    )


def test_deprecated_set_refused(tmp_path: Path, dev_set: ep.EpisodeSet, grid) -> None:  # type: ignore[no-untyped-def]
    # Invalidated on 2026-10-20 when R3 was recollected.
    old = replace(dev_set, name="env0_test_v1")
    with pytest.raises(ep.EpisodeSetError):
        Campaign(
            _ctx(tmp_path, old, grid), ("dense-contloc-ptgoal",), tuple(dev_set.tokens()), 1
        ).run()
    assert ep.TEST_SET == "env0_test_v3"
