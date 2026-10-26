#!/usr/bin/env python3
"""Verify bundled runtime, datasets and assets without starting Isaac or calling a VLM."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vnav_benchmark import frozen, mapview, oracle, pins  # noqa: E402
from vnav_benchmark.episodes import freeze_episode_set, load_episode_set  # noqa: E402
from vnav_benchmark.paths import asset_root, data_root  # noqa: E402
from vnav_benchmark.preflight import _route_set_scene_check  # noqa: E402
from vnav_benchmark.scene_fingerprint import scene_fingerprint  # noqa: E402
from vnav_benchmark.scenes import SCENES  # noqa: E402

EPISODE_SETS = ("env0_test_v3", "env1_test_v1", "env3_test_v1")


def published_path(value: str) -> Path:
    """Resolve a manifest path using the same asset/data overrides as the runtime."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid manifest path: " + value)
    roots = {"assets": asset_root(), "data": data_root()}
    if relative.parts[0] in roots:
        return roots[relative.parts[0]].joinpath(*relative.parts[1:])
    return REPO / relative


def verify_assets() -> int:
    """Check every published file and its recorded size; never follow asset symlinks."""
    manifest = json.loads((asset_root() / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = published_path(item["path"])
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Asset symlinks are not supported: " + item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError("Missing or resized asset: " + item["path"])
        if pins.file_sha256(path) != item["sha256"]:
            raise ValueError("Changed asset: " + item["path"])
    return len(manifest["files"])


def smoke() -> dict:
    """Run one excluded debugging cell on a temporary copy of the real ENV0 dataset."""
    from vnav_benchmark.budget import BudgetGuard
    from vnav_benchmark.runner import BenchConfig, Campaign, KinematicProvider, RunContext

    source = load_episode_set(REPO / "configs/episodes/env0_test_v3")
    payload = json.loads(json.dumps(source.payload))
    payload["name"] = "env0_dev_v1"
    payload["episodes"] = payload["episodes"][:1]
    record = payload["episodes"][0]
    images = {name: (source.root / name).read_bytes()
              for name in (record["goal_rgb"], record["goal_depth"])}
    with tempfile.TemporaryDirectory(prefix="vnav-smoke-") as temporary:
        root = Path(temporary)
        freeze_episode_set(root / "episodes", payload, images)
        episodes = load_episode_set(root / "episodes")
        context = RunContext(
            root=root / "campaign", episode_set=episodes,
            sim=KinematicProvider(oracle.build_truth_grid()), maps=None,
            guard=BudgetGuard(root / "ledger.jsonl"), bench=BenchConfig.load(),
        )
        result = Campaign(context, ("debug-stop-now",), tuple(episodes.tokens()), 1).run(max_jobs=1)
        runs = list((context.root / "runs").rglob("run.json"))
        if len(runs) != 1:
            raise ValueError("Kinematic smoke did not complete exactly one cell")
        doc = json.loads(runs[0].read_text(encoding="utf-8"))
        if doc["status"] != "excluded_debug":
            raise ValueError("Smoke cell must remain excluded from scoring")
        return {"cells": result["done"], "status": doc["status"], "model_calls": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Also run one CPU-only debug cell")
    parser.add_argument("--robot", action="store_true", help="Also verify licensed local Go2 files")
    args = parser.parse_args()
    pins.require_pins()
    problems = frozen.check_layer(REPO, REPO / "configs")
    if problems:
        raise ValueError("Frozen runtime differs: " + ", ".join(problems))
    assets = verify_assets()
    config = mapview.load_maps_config()
    for entry in config.maps.values():
        mapview.verify_view(entry.source, entry.view_sha256)
    selections = json.loads((data_root() / "active_maps/selections.json").read_text())
    for row in selections["routes"]:
        mapview.verify_view(published_path(row["map_path"]), row["view_sha256"])
    episodes = 0
    for name in EPISODE_SETS:
        episode_set = load_episode_set(REPO / "configs/episodes" / name)
        check = _route_set_scene_check(
            episode_set.root / "episodes.json", scene_fingerprint(episode_set.payload["scene"])
        )
        if not check.ok:
            raise ValueError(check.detail)
        for token in episode_set.tokens():
            spec = episode_set.spec(token, 0, 90.0)
            if spec.goal_rgb.shape != (480, 640, 3):
                raise ValueError("Unexpected goal-image dimensions")
        episodes += len(episode_set.records)
    result = {"runtime_files": len(pins.PINS), "assets": assets, "maps": len(config.maps),
              "active_maps": len(selections["routes"]), "episodes": episodes,
              "scenes": list(SCENES), "simulation_started": False, "model_calls": 0}
    if args.robot:
        subprocess.run([sys.executable, str(REPO / "setup/robot.py"), "--check"], check=True)
    if args.smoke:
        result["smoke"] = smoke()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
