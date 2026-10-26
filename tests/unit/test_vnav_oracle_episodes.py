"""Test truth grids, shortest paths, endpoint sampling and freezing without a simulator.

Rendering uses a fake backend.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from vnav_route_fixtures import make_route_episode_set

from frontier_nav_bench import episodes as ep
from frontier_nav_bench import frozen
from frontier_nav_bench import oracle as orc


def _synthetic(free: np.ndarray, walls=()) -> orc.TruthGrid:
    return orc.TruthGrid("synthetic", 0.0, 0.0, 0.1, free, ~free, (), tuple(walls))


@pytest.fixture(scope="module")
def env0() -> orc.TruthGrid:
    return orc.build_truth_grid()


def test_env0_grid_regression(env0: orc.TruthGrid) -> None:
    # Inflation radius 0.30 m: Ranger half-width 0.25 m plus 5 cm; old Go2 radius yielded 22,713
    # cells.
    assert env0.free.shape == (220, 220) and env0.free_count() == 19712
    assert len(env0.rooms) == 9 and env0.room_of(12.5, 2.6) == "lobby"
    assert env0.is_free_xy(12.5, 2.6)  # Default spawn point.


def test_empty_room_geodesic_is_straight() -> None:
    grid = _synthetic(np.ones((100, 100), bool))
    a, b = (1.05, 1.05), (8.95, 6.35)
    straight = math.hypot(b[0] - a[0], b[1] - a[1])
    assert orc.shortest_path(grid, a, b).length_m == pytest.approx(straight, rel=1e-9)
    # Dijkstra plus string-pulling must remain within +/-1%, without the line-of-sight shortcut.
    field = orc.distance_field(grid, grid.cell_of(*b))
    cells = orc._descend(field, grid, grid.cell_of(*a))
    pts = [a] + [grid.center_of(c) for c in cells[1:-1]] + [b]
    assert orc.polyline_length(orc.pull_string(grid, pts)) == pytest.approx(straight, rel=0.01)
    # Omitting string-pulling overestimates path length.
    assert orc.polyline_length(pts) > straight * 1.02


def test_around_wall_known_case() -> None:
    free = np.ones((100, 100), bool)
    free[47:53, :73] = False  # Inflated wall spans x=4.7-5.3 and y=0-7.3.
    grid = _synthetic(free)
    path = orc.shortest_path(grid, (2.05, 2.05), (7.95, 2.05))
    expect = 2 * math.hypot(4.7 - 2.05, 7.3 - 2.05) + 0.6
    assert path.length_m == pytest.approx(expect, rel=0.02)
    assert max(p[1] for p in path.waypoints) >= 7.2


def test_unreachable_and_blocked() -> None:
    free = np.ones((50, 50), bool)
    free[24:26, :] = False  # A complete wall.
    grid = _synthetic(free)
    assert orc.shortest_path(grid, (1.0, 1.0), (4.0, 1.0)) is None
    assert orc.shortest_path(grid, (2.45, 1.0), (1.0, 1.0)) is None  # The start lies inside a wall.


def test_wall_los_consistent_with_free_los(env0: orc.TruthGrid) -> None:
    rng = np.random.default_rng(3)
    free = np.argwhere(env0.free)
    checked = 0
    for _ in range(1500):
        a = env0.center_of(tuple(free[rng.integers(len(free))]))
        b = env0.center_of(tuple(free[rng.integers(len(free))]))
        if orc.free_line_of_sight(env0, a, b):
            checked += 1
            assert orc.wall_line_of_sight(env0, a, b)
    assert checked > 50
    # The wall at x=11 blocks visibility between office_a and office_b.
    assert not orc.wall_line_of_sight(env0, (9.0, 18.0), (13.0, 18.0))


def test_nearest_free() -> None:
    free = np.ones((50, 50), bool)
    free[:20, :] = False  # Cells with x<2.0 m are blocked.
    grid = _synthetic(free)
    # Return an already-free point unchanged.
    assert orc.nearest_free(grid, (3.05, 1.05), 0.5) == (3.05, 1.05)
    snapped = orc.nearest_free(grid, (1.8, 1.05), 0.5)
    assert snapped is not None and snapped[0] >= 2.0 and abs(snapped[0] - 1.8) <= 0.5
    assert orc.nearest_free(grid, (0.5, 1.0), 0.5) is None


def test_freeze_load_spec(env0: orc.TruthGrid, tmp_path: Path) -> None:
    es = make_route_episode_set(tmp_path, env0, "env0_test_vX")
    assert len(es.tokens()) == 4 and es.is_test
    spread = es.payload["start_spread"]
    assert (
        spread["max_dist_m"] < 0.2 and len(spread["starts"]) == 4
    )  # Routes share a start; record any differences in the frozen set.
    tok = es.tokens()[0]
    spec = es.spec(tok, 0, 90.0)
    assert spec.goal_rgb.shape == (48, 64, 3) and spec.budget_s == 90.0
    assert float(np.median(spec.goal_depth_m)) > 1.0
    assert not [k for k in vars(spec) if "pose" in k or "start" in k or "route" in k]
    with pytest.raises(ep.EpisodeSetError):
        es.check_method_allowed(uses_truth=True)
    payload = es.payload
    files = {p.name: p.read_bytes() for p in es.root.iterdir() if p.name.startswith("goal_")}
    with pytest.raises(frozen.FrozenError):  # Overwriting is prohibited.
        ep.freeze_episode_set(es.root, payload, files)
    with pytest.raises(ep.EpisodeSetError):  # Freezing requires the goal photograph.
        ep.freeze_episode_set(tmp_path / "other", payload, {})
    (es.root / es.records[tok].goal_rgb).write_bytes(b"tampered")
    with pytest.raises(frozen.FrozenError):
        ep.load_episode_set(es.root)


def test_frozen_dir_detects_edits(tmp_path: Path) -> None:
    target = tmp_path / "set"
    frozen.freeze_dir(target, {"a.json": b"{}\n"}, {"k": 1})
    assert frozen.verify_dir(target)["meta"] == {"k": 1}
    (target / "extra.txt").write_text("x")
    with pytest.raises(frozen.FrozenError):
        frozen.verify_dir(target)
    (target / "extra.txt").unlink()
    manifest = json.loads((target / frozen.MANIFEST_NAME).read_text())
    # User edits differ from the fixed encoding.
    (target / frozen.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(frozen.FrozenError):
        frozen.verify_dir(target)


def test_real_route_registry() -> None:
    """Evaluation routes are registered only in maps.yaml routes; the selected R1-R5 all
    participate.
    """
    from frontier_nav_bench import mapview

    config = mapview.load_maps_config()
    assert [r.label for r in config.eval_routes()] == ["R1", "R2", "R3", "R4", "R5"]
