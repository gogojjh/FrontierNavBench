"""T0 tests for sanitized map views and configs/maps.yaml."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from vnav_t_fixtures import CAMERA_FORWARD_M, L_CORRIDOR, make_map

from frontier_nav_bench import mapview as mv
from frontier_nav_bench.topo import load_topo_graph


def _view(tmp_path: Path) -> Path:
    src = make_map(tmp_path / "src", trav_edges=[(0, 8)])
    dst = tmp_path / "view"
    mv.build_sanitized_view(src, dst)
    return dst


def test_view_keeps_only_whitelist(tmp_path: Path) -> None:
    dst = _view(tmp_path)
    top = sorted(path.name for path in dst.iterdir())
    assert top == sorted(
        list(mv.TOP_LEVEL_FILES) + list(mv.EDGE_FILES) + ["seq", mv.VIEW_META_NAME]
    )
    assert len(list((dst / "seq").iterdir())) == 2 * len(L_CORRIDOR)
    for name in mv.FORBIDDEN_NAMES + ("edges_room.txt", "objects.json"):
        assert not list(dst.rglob(name))
    meta_text = (dst / mv.VIEW_META_NAME).read_text()
    assert "/somewhere" not in meta_text and "boxer" not in meta_text
    meta = json.loads(meta_text)
    assert meta["num_keyframes"] == len(L_CORRIDOR)
    assert meta["keyframe_overrides"] == ["keyframe.trans_thresh_m=1.0"]


def test_view_is_read_only_and_not_overwritten(tmp_path: Path) -> None:
    dst = _view(tmp_path)
    assert not os.access(dst / "poses.txt", os.W_OK) or os.geteuid() == 0
    with pytest.raises(mv.MapViewError, match="已存在"):
        mv.build_sanitized_view(tmp_path / "src", dst)


def test_sha_is_reproducible_and_detects_tamper(tmp_path: Path) -> None:
    src = make_map(tmp_path / "src")
    sha_a = mv.build_sanitized_view(src, tmp_path / "a")
    sha_b = mv.build_sanitized_view(src, tmp_path / "b")
    assert sha_a == sha_b == mv.verify_view(tmp_path / "a", sha_a)
    edges = tmp_path / "b" / "edges_trav.txt"
    os.chmod(edges, 0o644)
    edges.write_text(edges.read_text() + "0 4 1.0\n")
    with pytest.raises(mv.MapViewError, match="指纹不符"):
        mv.verify_view(tmp_path / "b", sha_a)


def test_sanitized_map_matches_source(tmp_path: Path) -> None:
    dst = _view(tmp_path)
    handle = mv.SanitizedMap(dst, "m", "dense", "trav", CAMERA_FORWARD_M)
    src_graph = load_topo_graph(tmp_path / "src", CAMERA_FORWARD_M, "trav")
    assert handle.num_keyframes == len(src_graph) == len(L_CORRIDOR)
    for kf, (x, y, yaw) in enumerate(L_CORRIDOR):
        pose = handle.keyframe_pose(kf)
        assert pose == pytest.approx((x, y, yaw), abs=1e-6)
    assert handle.keyframe_rgb_jpeg(3) == (tmp_path / "src/seq/000003.color.jpg").read_bytes()
    depth = handle.keyframe_depth_m(2)
    assert depth.dtype == np.float32 and float(depth[0, 0]) == pytest.approx(1.002)
    assert dict(handle.topo_graph().neighbors(0))[8] > 0  # Shortcut edges were loaded
    with pytest.raises(KeyError):
        handle.keyframe_rgb_jpeg(99)
    chain = mv.SanitizedMap(dst, "m", "dense", "trav.chain", CAMERA_FORWARD_M)
    assert 8 not in dict(chain.topo_graph().neighbors(0))


def test_materialize_reuses_and_checks_sha(tmp_path: Path) -> None:
    src = make_map(tmp_path / "src")
    sha = mv.build_sanitized_view(src, tmp_path / "probe")
    entry = mv.MapEntry(
        "dev_x", "dense", "dev_placeholder", src, "trav", 1.0, 30.0, np.eye(4), sha, 9
    )
    first = mv.materialize(entry, tmp_path / "views", CAMERA_FORWARD_M)
    again = mv.materialize(entry, tmp_path / "views", CAMERA_FORWARD_M)
    assert first.sha256 == again.sha256 == sha
    wrong = entry.__class__(**{**entry.__dict__, "view_sha256": "0" * 64})
    with pytest.raises(mv.MapViewError):
        mv.materialize(wrong, tmp_path / "views", CAMERA_FORWARD_M)


def test_repo_maps_yaml_parses() -> None:
    config = mv.load_maps_config()
    assert config.camera_forward_m == pytest.approx(0.1582)
    assert {entry.variant for entry in config.maps.values()} == {"dense", "sparse"}
    for variant in ("dense", "sparse"):
        entry = config.for_variant(variant)
        assert entry.variant == variant and entry.edge_type == "trav"
        assert entry.view_sha256 and len(entry.view_sha256) == 64
        rotation = entry.map_to_world[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-4)


def test_maps_yaml_rejects_bad_default(tmp_path: Path) -> None:
    data = yaml.safe_load(mv.MAPS_CONFIG_PATH.read_text())
    data["defaults"]["dense"] = data["defaults"]["sparse"]
    text = yaml.safe_dump(data)
    bad = tmp_path / "maps.yaml"
    bad.write_text(text)
    with pytest.raises(mv.MapViewError, match="密度不符"):
        mv.load_maps_config(bad)


def test_maps_yaml_rejects_non_rigid(tmp_path: Path) -> None:
    config = yaml.safe_load(mv.MAPS_CONFIG_PATH.read_text())
    first = next(iter(config["maps"].values()))
    first["map_to_world"][0] = 2
    bad = tmp_path / "maps.yaml"
    bad.write_text(yaml.safe_dump(config))
    with pytest.raises(mv.MapViewError, match="刚体"):
        mv.load_maps_config(bad)


@pytest.mark.parametrize("variant", ["dense", "sparse"])
def test_bundled_default_maps(tmp_path: Path, variant: str) -> None:
    """Bundled maps match their registered node counts and fingerprints."""
    config = mv.load_maps_config()
    entry = config.for_variant(variant)
    assert entry.source.is_dir(), entry.source
    handle = mv.materialize(entry, tmp_path, config.camera_forward_m)
    source = load_topo_graph(entry.source, config.camera_forward_m, entry.edge_type)
    assert handle.num_keyframes == len(source) == entry.num_keyframes
    assert len(handle.topo_graph().components()) == 1


def test_view_copies_trav_verbatim(tmp_path: Path) -> None:
    """Copy edges_trav.txt unchanged; planning uses traversal edges without merging covisibility."""
    src = make_map(tmp_path / "src", trav_edges=[(0, 8)])
    (src / "edges_covis.txt").write_text("0 1 0.9\n2 0 0.5\n")
    dst = tmp_path / "view"
    mv.build_sanitized_view(src, dst)
    assert (dst / "edges_trav.txt").read_bytes() == (src / "edges_trav.txt").read_bytes()
    handle = mv.SanitizedMap(dst, "m", "dense", "trav", CAMERA_FORWARD_M)
    assert 2 not in dict(handle.topo_graph().neighbors(0))  # Covisibility edge 0-2 was excluded
    # from the planning graph


def test_for_route(tmp_path: Path) -> None:
    text = mv.MAPS_CONFIG_PATH.read_text()
    extra = ""
    for variant in ("dense", "sparse"):
        extra += f"""
  r1_{variant}:
    variant: {variant}
    role: formal_candidate
    route: 20261021-100000-aaaaaa
    source: /nonexistent
    keyframe_thresh: {{trans_m: 3.0, rot_deg: 45}}
    edge_type: trav
    map_to_world: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
"""
    path = tmp_path / "maps.yaml"
    path.write_text(text + extra)
    config = mv.load_maps_config(path)
    assert "20261021-100000-aaaaaa" in config.routes()
    assert config.for_route("20261021-100000-aaaaaa", "sparse").map_id == "r1_sparse"
    with pytest.raises(mv.MapViewError, match="0 张"):
        config.for_route("nope", "dense")


def test_repo_route_table_r1_to_r5() -> None:
    """maps.yaml routes registers all five approved routes for evaluation."""
    config = mv.load_maps_config()
    env0_labels = sorted(
        k for k, r in config.route_table.items() if r.scene == mv.DEFAULT_SCENE
    )
    assert env0_labels == ["R1", "R2", "R3", "R4", "R5"]
    assert [r.label for r in config.eval_routes()] == ["R1", "R2", "R3", "R4", "R5"]
    r4 = config.route_info("R4")
    assert r4.eval == "formal" and r4.end_room == "meeting_room"
    assert config.route_info(r4.session) is r4  # Lookup also accepts the recording-session
    # directory name
    shortest = [config.route_table[k].shortest_m for k in env0_labels]
    assert shortest == sorted(shortest)  # Labels follow increasing ground-truth shortest-path
    # length
    with pytest.raises(mv.MapViewError):
        config.route_info("not_registered")
