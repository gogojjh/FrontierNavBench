"""Stage identity checks without Isaac, SSH, or a Git checkout."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from vnav_route_fixtures import DEFAULT_ROUTES, make_route_episode_set

from vnav_benchmark import oracle
from vnav_benchmark.budget import BudgetGuard
from vnav_benchmark.interfaces import FatalInfraError, Pose2D
from vnav_benchmark.runner import Campaign, CampaignStopped, IsaacProvider, RunContext
from vnav_benchmark.sim import ctrl_client, isaac_server
from vnav_benchmark.sim.stage_check import sim_dir_sha256


def test_sim_digest_is_stable_and_independent_of_directory_order(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    files = {"z.py": b"Z = 1\n", "a.py": b"A = 1\n", "nested/b.py": b"B = 1\n"}
    for root, names in ((first, list(files)), (second, list(reversed(files)))):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[name])
    digest = sim_dir_sha256(first)
    assert len(digest) == 64
    assert digest == sim_dir_sha256(first) == sim_dir_sha256(second)
    assert sim_dir_sha256() == sim_dir_sha256(Path(isaac_server.__file__).resolve().parent)


@pytest.mark.parametrize("change", ["edit", "add", "remove", "rename", "nested_edit"])
def test_sim_digest_detects_python_changes(tmp_path: Path, change: str) -> None:
    source = tmp_path / "backend.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    nested = tmp_path / "nested" / "trigger.py"
    nested.parent.mkdir()
    nested.write_text("VALUE = 2\n", encoding="utf-8")
    before = sim_dir_sha256(tmp_path)
    if change == "edit":
        source.write_text("VALUE = 3\n", encoding="utf-8")
    elif change == "add":
        (tmp_path / "extra.py").write_bytes(b"")
    elif change == "remove":
        source.unlink()
    elif change == "rename":
        source.rename(tmp_path / "renamed.py")
    else:
        nested.write_text("VALUE = 3\n", encoding="utf-8")
    assert sim_dir_sha256(tmp_path) != before


def test_sim_digest_ignores_non_python_files(tmp_path: Path) -> None:
    (tmp_path / "backend.py").write_bytes(b"# source\n")
    before = sim_dir_sha256(tmp_path)
    (tmp_path / "notes.txt").write_bytes(b"notes")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "backend.cpython-38.pyc").write_bytes(b"cache")
    assert sim_dir_sha256(tmp_path) == before


def test_server_identity_hashes_its_own_archive_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sim" / "isaac_server.py"
    source.parent.mkdir()
    source.write_bytes(b"# archived server\n")
    monkeypatch.setattr(isaac_server, "__file__", str(source))
    monkeypatch.setattr(isaac_server, "git_sha", lambda path: {"head": "", "dirty": None})

    identity = isaac_server.server_identity("env0_office_single_floor", None)
    assert identity["sim_dir_sha256"] == sim_dir_sha256(source.parent)
    assert identity["git"]["vnav_benchmark"]["head"] == ""


class FakeClient:
    def __init__(self, reply: Dict[str, Any]) -> None:
        self.reply = reply
        self.closed = False

    def hello(self) -> Dict[str, Any]:
        return dict(self.reply)

    def close(self) -> None:
        self.closed = True


class FakeOrchestrator:
    def __init__(self, identity: Dict[str, Any]) -> None:
        self.info = identity
        self.starts = 0
        self.stops = 0

    def start(
        self, scene: str, xy: Tuple[float, float], yaw: float, log: Path, third: bool
    ) -> str:
        self.starts += 1
        return "inproc://fake-isaac"

    def identity(self) -> Dict[str, Any]:
        return dict(self.info)

    def stop(self) -> None:
        self.stops += 1


@pytest.mark.parametrize("remote_sha", ["0" * 64, None], ids=["mismatch", "old-server"])
def test_provider_refuses_mismatched_or_missing_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_sha: Optional[str]
) -> None:
    local_sha = sim_dir_sha256()
    reply = {"robot": "go2"}
    if remote_sha is not None:
        reply["sim_dir_sha256"] = remote_sha
    client = FakeClient(reply)
    # A stale orchestrator cache must not hide the actual client's hello mismatch.
    orch = FakeOrchestrator({"robot": "go2", "sim_dir_sha256": local_sha})
    monkeypatch.setattr(ctrl_client, "CtrlClient", lambda endpoint: client)

    with pytest.raises(FatalInfraError) as caught:
        IsaacProvider(orch, robot="go2").open("scene", Pose2D(0.0, 0.0, 0.0), False, tmp_path)
    message = str(caught.value)
    assert "sim_dir_sha256" in message
    assert local_sha in message and str(remote_sha) in message
    assert "bash scripts/desktop_stage.sh" in message
    assert client.closed and orch.stops == 1 and orch.starts == 1


def test_provider_accepts_matching_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = {"robot": "go2", "sim_dir_sha256": sim_dir_sha256()}
    client = FakeClient(reply)
    orch = FakeOrchestrator(reply)
    monkeypatch.setattr(ctrl_client, "CtrlClient", lambda endpoint: client)

    session = IsaacProvider(orch, robot="go2").open(
        "scene", Pose2D(0.0, 0.0, 0.0), False, tmp_path
    )
    assert session.link is client and session.identity == reply
    assert not client.closed and orch.stops == 0
    session.close()
    assert client.closed and orch.stops == 1


def test_digest_mismatch_stops_campaign_without_retry_or_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = oracle.build_truth_grid()
    episodes = make_route_episode_set(tmp_path / "episodes", grid, pairs=DEFAULT_ROUTES[:1])
    root = tmp_path / "campaign"
    reply = {"robot": "go2", "sim_dir_sha256": "0" * 64}
    client = FakeClient(reply)
    orch = FakeOrchestrator(reply)
    monkeypatch.setattr(ctrl_client, "CtrlClient", lambda endpoint: client)
    ctx = RunContext(
        root, episodes, IsaacProvider(orch, robot="go2"), None,
        BudgetGuard(root / "ledger.jsonl"), grid=grid,
    )
    alerts: List[str] = []

    with pytest.raises(CampaignStopped, match="sim_dir_sha256"):
        Campaign(ctx, ("debug-stop-now",), tuple(episodes.tokens()), 1, alert=alerts.append).run()
    assert orch.starts == 1 and orch.stops == 1 and client.closed
    assert not list((root / "runs").rglob("run.json"))
    assert "致命基础设施错误" in alerts[0]
    assert "bash scripts/desktop_stage.sh" in alerts[0]
