"""Ideal evaluator-supplied localization, frozen after T-I0 with no noise in v1.

All mapped v1 methods use OracleLocalizer. purpose=goal returns the truth body pose when the
goal photo was captured, stored after settling in the frozen episode set. purpose=self returns
the robot's current truth pose. Both transform into map coordinates with inv(map_to_world) times
T_world; configs/maps.yaml supplies map_to_world (identity for 29570b). fairness.QuotaLocalizer
wraps this implementation according to the method's localization allowance.

Optional v2 hooks add Gaussian xy/yaw noise. Keyframe-distance gating and real LiteVLoc can
implement the same Localizer protocol separately.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

from frontier_nav_bench import oracle
from frontier_nav_bench.interfaces import Intrinsics, LocResult, Pose2D

NOISE_SIGMA_XY_M = 0.0  # v1 adds no noise; the original D1 noise setting was withdrawn.
NOISE_SIGMA_YAW_RAD = 0.0


def map_from_world(map_to_world: Sequence[float]) -> np.ndarray:
    """Invert the row-major 16-value map-to-world transform from maps.yaml."""
    mat = np.asarray(map_to_world, dtype=float).reshape(4, 4)
    return np.linalg.inv(mat)


def transform_pose2d(t_map_world: np.ndarray, pose: Pose2D) -> Pose2D:
    """Transform a planar world pose into map xy/yaw; both frames are gravity-aligned."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    t = np.array([[c, -s, 0.0, pose.x], [s, c, 0.0, pose.y], [0, 0, 1, 0], [0, 0, 0, 1]])
    out = t_map_world @ t
    return Pose2D(float(out[0, 3]), float(out[1, 3]), math.atan2(out[1, 0], out[0, 0]))


class OracleLocalizer:
    """Construct per episode; ignore input images because localization uses evaluator truth."""

    def __init__(
        self,
        # Return the robot's current world-frame truth from SimEnv.truth_pose.
        truth_now: Callable[[], Pose2D],
        goal_pose_world: Pose2D,
        map_to_world: Sequence[float],
        rng: Optional[np.random.Generator] = None,
        sigma_xy_m: float = NOISE_SIGMA_XY_M,
        sigma_yaw_rad: float = NOISE_SIGMA_YAW_RAD,
        *,
        scene: Optional[str] = None,
        truth_z_now: Optional[Callable[[], float]] = None,
        goal_floor: Optional[str] = None,
    ) -> None:
        self._truth_now = truth_now
        self._goal = goal_pose_world
        self._t_map_world = map_from_world(map_to_world)
        self._rng = rng or np.random.default_rng(0)
        self._sigma_xy = sigma_xy_m
        self._sigma_yaw = sigma_yaw_rad
        self._scene = scene
        self._truth_z_now = truth_z_now
        if scene == oracle.SCENE_ENV3 and not goal_floor:
            raise ValueError("多层定位需要 goal_floor")
        self._goal_floor = goal_floor or (
            oracle.floor_of_z(scene, 0.0) if scene is not None else None
        )

    def _noisy(self, pose: Pose2D) -> Pose2D:
        if self._sigma_xy <= 0.0 and self._sigma_yaw <= 0.0:
            return pose
        dx, dy = self._rng.normal(0.0, self._sigma_xy, 2) if self._sigma_xy > 0 else (0.0, 0.0)
        dyaw = self._rng.normal(0.0, self._sigma_yaw) if self._sigma_yaw > 0 else 0.0
        return Pose2D(pose.x + float(dx), pose.y + float(dy), pose.yaw + float(dyaw))

    def localize(
        self, rgb_jpeg: bytes, depth_m: Optional[np.ndarray], K: Intrinsics, *, purpose: str
    ) -> LocResult:
        if purpose == "goal":
            world = self._goal
            floor = self._goal_floor
        elif purpose == "self":
            world = self._truth_now()
            floor = (
                oracle.floor_of_z(self._scene, self._truth_z_now())
                if self._scene is not None and self._truth_z_now is not None else None
            )
        else:
            raise ValueError(f"purpose 只能是 goal / self: {purpose!r}")
        pose = self._noisy(transform_pose2d(self._t_map_world, world))
        return LocResult(True, pose, None, "oracle", floor=floor)
