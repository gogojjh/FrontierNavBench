"""Elevator geometry and ordered entry/exit paths, without simulation."""

import math
from dataclasses import replace

import pytest

from frontier_nav_bench import oracle


@pytest.fixture(scope="module")
def env3() -> oracle.MultiFloorGrid:
    return oracle.build_multifloor_truth_grid(oracle.SCENE_ENV3)


def test_real_elevator_has_dense_entry_checkpoints(env3: oracle.MultiFloorGrid) -> None:
    transition = next(t for t in env3.transitions if t.kind == "elevator")
    points = transition.waypoints

    assert len(points) > 2
    assert points[0] == pytest.approx((8.6, 5.0))
    assert points[-1] == pytest.approx((8.6, 6.975))
    assert points[-1] == transition.a_xy == transition.b_xy
    assert max(math.dist(a, b) for a, b in zip(points, points[1:])) <= 0.8 + 1e-9
    assert all(env3.floors[f].is_free_xy(*p) for f in ("f1", "f2") for p in points)


def test_elevator_distance_includes_walking_to_cab_center(env3: oracle.MultiFloorGrid) -> None:
    path = oracle.multifloor_shortest_path(env3, "f1", (8.6, 3.4), "f2", (8.6, 4.6))
    transition = next(t for t in env3.transitions if t.kind == "elevator")

    assert path is not None
    assert 9.0 <= path.length_m <= 9.3
    assert transition.length_m == pytest.approx(3.2)
    assert path.elevator_rides == 1
    assert path.elevator_vertical_m == pytest.approx(3.2)
    first = oracle.shortest_path(env3.floors["f1"], (8.6, 3.4), transition.waypoints[0])
    last = oracle.shortest_path(env3.floors["f2"], transition.waypoints[0], (8.6, 4.6))
    assert first is not None and last is not None
    guided = first.length_m + 2 * oracle.polyline_length(transition.waypoints) + 3.2
    guided += last.length_m
    assert guided == pytest.approx(path.length_m, abs=0.1)


def test_starting_inside_cab_keeps_a_boarding_subgoal(env3: oracle.MultiFloorGrid) -> None:
    transition = next(t for t in env3.transitions if t.kind == "elevator")
    center = transition.a_xy
    path = oracle.multifloor_shortest_path(env3, "f1", center, "f2", center)
    assert path is not None
    assert path.length_m == transition.length_m
    # Probes consume waypoints[1:]; a robot already inside still needs to wait for its floor.
    assert path.waypoints == (center, center)


@pytest.mark.parametrize("reverse", [False, True])
def test_both_directions_route_via_door_then_cab_then_door(
    env3: oracle.MultiFloorGrid, reverse: bool
) -> None:
    transition = next(t for t in env3.transitions if t.kind == "elevator")
    assert transition.waypoints
    front = transition.waypoints[0]
    # Lateral approaches must reach the door front before aiming into the cab.
    start, goal = (front[0] - 0.6, front[1]), (front[0] + 0.6, front[1])
    start_floor, goal_floor = ("f2", "f1") if reverse else ("f1", "f2")
    path = oracle.multifloor_shortest_path(
        replace(env3, transitions=(transition,)), start_floor, start, goal_floor, goal
    )
    expected = transition.waypoints + tuple(reversed(transition.waypoints[:-1]))

    assert path is not None
    assert any(path.waypoints[i:i + len(expected)] == expected for i in range(len(path.waypoints)))
    first = oracle.shortest_path(env3.floors[start_floor], start, transition.a_xy)
    last = oracle.shortest_path(env3.floors[goal_floor], transition.b_xy, goal)
    assert first is not None and last is not None
    assert path.length_m == pytest.approx(first.length_m + transition.length_m + last.length_m)
