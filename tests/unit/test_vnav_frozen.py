"""Frozen-manifest consistency and tooling checks detect changes to frozen interface/evaluation
layers.
"""

import json
from pathlib import Path

import pytest

from vnav_benchmark import frozen

REPO = Path(__file__).resolve().parents[2]


def test_repo_matches_latest_manifest() -> None:
    """Frozen-layer changes/additions/deletions fail; updates require a new version and rerunning
    all methods.
    """
    assert frozen.layer_manifests(REPO / "configs"), "Release must include frozen manifests"
    assert frozen.check_layer(REPO, REPO / "configs") == []


def _fake_repo(tmp_path: Path) -> Path:
    for rel in ("vnav_benchmark/interfaces.py", "vnav_benchmark/sim/ctrl_server.py",
                "configs/bench.yaml", "configs/episodes/env0_test_v1/episodes.json",
                "vnav_benchmark/methods/imggoal_vlm.py"):  # fmt: skip
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n")
    return tmp_path


def test_scope_excludes_method_layer(tmp_path: Path) -> None:
    files = frozen.layer_files(_fake_repo(tmp_path))
    assert "vnav_benchmark/methods/imggoal_vlm.py" not in files
    assert "configs/episodes/env0_test_v1/episodes.json" in files and len(files) == 4


def test_freeze_check_and_chain(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    cfg = repo / "configs"
    assert frozen.check_layer(repo, cfg) == []  # No prior frozen manifest
    v1 = frozen.freeze_layer(repo, cfg, "T-I0")
    assert (
        v1.name == "frozen_manifest_v1.json" and json.loads(v1.read_text())["parent_sha256"] is None
    )
    assert frozen.check_layer(repo, cfg) == []
    (repo / "vnav_benchmark/methods/imggoal_vlm.py").write_text("changed\n")  # Method-layer changes
    # do not affect the frozen layer
    assert frozen.check_layer(repo, cfg) == []
    (repo / "vnav_benchmark/sim/ctrl_server.py").write_text("changed\n")
    (repo / "vnav_benchmark/sim/new_thing.py").write_text("x\n")
    problems = frozen.check_layer(repo, cfg)
    assert "改过 vnav_benchmark/sim/ctrl_server.py" in problems
    assert "新增未登记 vnav_benchmark/sim/new_thing.py" in problems
    (repo / "configs/bench.yaml").unlink()
    assert "缺文件 configs/bench.yaml" in frozen.check_layer(repo, cfg)
    v2 = frozen.freeze_layer(repo, cfg, "升版")
    assert json.loads(v2.read_text())["parent_sha256"] == frozen.sha256_bytes(v1.read_bytes())
    assert frozen.check_layer(repo, cfg) == []


def test_chain_tamper_detected(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    cfg = repo / "configs"
    v1 = frozen.freeze_layer(repo, cfg)
    frozen.freeze_layer(repo, cfg)
    v1.write_text(v1.read_text().replace('"note": ""', '"note": "edited"'))  # Changing an older
    # version breaks the manifest chain
    with pytest.raises(frozen.FrozenError):
        frozen.check_layer(repo, cfg)


def test_no_overwrite_and_gap(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    cfg = repo / "configs"
    v1 = frozen.freeze_layer(repo, cfg)
    with pytest.raises(FileExistsError):
        frozen.write_exclusive(v1, b"{}")
    (cfg / "frozen_manifest_v3.json").write_text("{}")  # Skipped version number
    with pytest.raises(frozen.FrozenError):
        frozen.freeze_layer(repo, cfg)
