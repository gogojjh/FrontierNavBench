"""Published input names, bindings and immutable map fingerprints."""

import json
from pathlib import Path

import pytest

from frontier_nav_bench.episodes import load_episode_set
from frontier_nav_bench.mapview import load_maps_config, materialize
from frontier_nav_bench.paths import repository_root

REPO = repository_root()
EPISODE_SETS = ("env0_test_v3", "env1_test_v1", "env3_test_v1")


def test_registry_matches_published_directories():
    config = load_maps_config()
    expected = {f"R{i:02d}_{variant}" for i in range(1, 16) for variant in ("dense", "sparse")}
    assert set(config.maps) == expected
    assert {path.name for path in (REPO / "data/maps").iterdir()} == expected
    for i in range(1, 16):
        for variant in ("dense", "sparse"):
            entry = config.for_route(f"R{i}", variant)
            assert entry.map_id == f"R{i:02d}_{variant}"
            assert entry.source == REPO / "data/maps" / entry.map_id


@pytest.mark.parametrize("name", EPISODE_SETS)
def test_frozen_episode_bindings_resolve(name):
    config = load_maps_config()
    episode_set = load_episode_set(REPO / "configs/episodes" / name)
    for source in episode_set.payload["routes"]:
        entry = config.for_route(source["route"], "dense")
        assert source["map_id"] == entry.map_id
        assert source["view_sha256"] == entry.view_sha256
        assert source["source"] == "${VNAV_DATA_DIR}/maps/" + entry.map_id


@pytest.mark.parametrize("variant", ("dense", "sparse"))
def test_every_map_materializes(tmp_path: Path, variant):
    config = load_maps_config()
    for i in range(1, 16):
        entry = config.for_route(f"R{i}", variant)
        view = materialize(entry, tmp_path, config.camera_forward_m)
        assert view.num_keyframes == entry.num_keyframes
        assert (tmp_path / entry.map_id / "view_meta.json").is_file()


def test_migration_inventory_preserves_fingerprints():
    config = load_maps_config()
    migration = json.loads((REPO / "data/maps_renamed.json").read_text())
    assert len(migration["maps"]) == 30
    for row in migration["maps"]:
        entry = config.maps[row["name"]]
        assert row["view_sha256"] == entry.view_sha256
        assert row["previous_name"] != row["name"]
