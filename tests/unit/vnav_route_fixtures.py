"""Freeze small synthetic episode sets using the public record and loading APIs."""

import math
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from vnav_benchmark import episodes as ep
from vnav_benchmark import oracle
from vnav_benchmark.interfaces import Pose2D

START = (12.5, 2.6)
DEFAULT_ROUTES: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
    (START, (18.0, 4.0)),
    (START, (4.0, 17.0)),
    (START, (9.0, 17.0)),
    (START, (5.0, 6.0)),
)


def make_route_episode_set(
    root: Path, grid: oracle.TruthGrid, name: str = "env0_dev_vT", pairs: Sequence = DEFAULT_ROUTES
) -> ep.EpisodeSet:
    """Construct known records and image assets, then freeze and verify them.

    Route collection and dataset generation are deliberately absent from the public
    evaluator. This fixture supplies synthetic records directly, without requiring
    the development tools that originally collected route maps.
    """
    records = []
    assets = {}
    for index, (start, goal) in enumerate(pairs):
        route = f"synthetic_route_{index}"
        token = ep.episode_token(name, route)
        start = oracle.nearest_free(grid, start, 1.0)
        goal = oracle.nearest_free(grid, goal, 1.0)
        assert start is not None and goal is not None
        path = oracle.shortest_path(grid, start, goal)
        assert path is not None, (start, goal)
        yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        rgb_name, depth_name = f"goal_{token}.jpg", f"goal_{token}.depth.png"
        ok_rgb, rgb = cv2.imencode(".jpg", np.full((48, 64, 3), 60 + index, np.uint8))
        ok_depth, depth = cv2.imencode(".png", np.full((48, 64), 1200 + index, np.uint16))
        assert ok_rgb and ok_depth
        assets[rgb_name], assets[depth_name] = rgb.tobytes(), depth.tobytes()
        start_room, goal_room = grid.room_of(*start), grid.room_of(*goal)
        records.append(ep.EpisodeRecord(
            token=token, route=route, label=f"R{index + 1}",
            start=Pose2D(*start, yaw), goal=Pose2D(*goal, yaw),
            route_kf_first=0, route_kf_last=1, shortest_m=path.length_m,
            straight_m=math.hypot(goal[0] - start[0], goal[1] - start[1]),
            tier=ep.tier_of(path.length_m), same_room=start_room == goal_room,
            start_room=start_room, goal_room=goal_room, waypoints=path.waypoints,
            goal_rgb=rgb_name, goal_depth=depth_name, goal_depth_median_m=1.2 + index / 1000,
        ))
    metadata = {"encoding": "u16mm+png", "width": 64, "height": 48,
                "K": [29.5, 0.0, 32.0, 0.0, 29.5, 24.0, 0.0, 0.0, 1.0]}
    ep.freeze_episode_set(root / name, ep.episode_set_json(name, records, grid, metadata), assets)
    return ep.load_episode_set(root / name)
