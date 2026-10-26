"""Calibrated Go2 camera transforms and gravity-aligned episode origins.

The camera remains 0.625 m above base_link; resets remove XY and yaw only."""

# Adapted from nav_sim/nav_sim/robot_frames.py.
# Source SHA256: 34880b83adeb491ff1286f649b401ec4e1e1de6d6b5c278b31071a8ed1714289.
# Adapted from nav_sim/nav_sim/go2_frames.py.
# Source SHA256: 9bf8042421e02b9b28910e66cc010ac42508b680e29aad93cd56023b31143003.
# Adapted from nav_sim/nav_sim/odom_reset.py.
# Source SHA256: f24b85fd9687a6c28893ae326ab4f73eab8f4afec1a56c2108a68df3115e50ec.
# Copyright (c) 2026 Jianhao Jiao. MIT license; see LICENSE.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

T_CAM_BODY = np.array(
    [
        [-0.0102, -0.99995, -0.00016, 0.05045],
        [0.00778, 8e-05, -0.99997, -0.00315],
        [0.99992, -0.0102, 0.00778, 0.00682],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def make_transform(xyz: Sequence[float], rotation: np.ndarray = None) -> np.ndarray:
    """Compose a homogeneous transform from position and optional rotation."""
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    transform[:3, 3] = [float(v) for v in xyz]
    return transform


def invert(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a generic matrix inverse."""
    rotation = transform[:3, :3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform[:3, 3]
    return result


def rot_y(angle_rad: float) -> np.ndarray:
    (c, s) = (math.cos(angle_rad), math.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_rad: float) -> np.ndarray:
    (c, s) = (math.cos(angle_rad), math.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Convert a normalized xyzw quaternion to a rotation matrix."""
    (x, y, z, w) = (float(v) for v in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    (x, y, z, w) = (x / norm, y / norm, z / norm, w / norm)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_xyzw(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a rotation matrix to a quaternion with a nonnegative w."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        (x, y, z) = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        x = 0.25 * s
        (w, y, z) = ((m[2, 1] - m[1, 2]) / s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        y = 0.25 * s
        (w, x, z) = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        z = 0.25 * s
        (w, x, y) = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s)
    if w < 0.0:
        (x, y, z, w) = (-x, -y, -z, -w)
    return (x, y, z, w)


def pose_to_transform(xyz: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    """Convert an xyzw pose to a homogeneous transform."""
    return make_transform(xyz, quat_xyzw_to_matrix(quat_xyzw))


def transform_to_pose(
    transform: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, ...]]:
    """Return position and xyzw orientation from a homogeneous transform."""
    xyz = tuple(float(v) for v in transform[:3, 3])
    return (xyz, matrix_to_quat_xyzw(transform[:3, :3]))


@dataclass(frozen=True)
class RobotFrames:
    """Fixed Go2 mount parameters and derived camera/IMU/base transforms."""

    name: str
    urdf_root: str
    mount_link: str
    t_base_link_camera: np.ndarray
    t_base_link_mount: np.ndarray
    t_base_link_urdf_root: np.ndarray

    @property
    def t_imu_camera(self) -> np.ndarray:
        return invert(T_CAM_BODY)

    @property
    def t_base_link_imu(self) -> np.ndarray:
        return self.t_base_link_camera @ T_CAM_BODY

    @property
    def t_imu_base_link(self) -> np.ndarray:
        return invert(self.t_base_link_imu)

    @property
    def t_world_odom(self) -> np.ndarray:
        return make_transform(self.t_base_link_imu[:3, 3])

    @property
    def t_mount_imu(self) -> np.ndarray:
        return invert(self.t_base_link_mount) @ self.t_base_link_imu

    @property
    def t_mount_camera(self) -> np.ndarray:
        return invert(self.t_base_link_mount) @ self.t_base_link_camera

    @property
    def stand_mount_height_m(self) -> float:
        return float(self.t_base_link_mount[2, 3])

    @property
    def camera_height_m(self) -> float:
        return float(self.t_base_link_camera[2, 3])

    def world_from_initial_imu(self, t_sim_imu: np.ndarray) -> np.ndarray:
        yaw = math.atan2(float(t_sim_imu[1, 0]), float(t_sim_imu[0, 0]))
        t_sim_odom = make_transform(t_sim_imu[:3, 3], rot_z(yaw))
        return t_sim_odom @ invert(self.t_world_odom)

    def odom_poses(self, t_world_mount: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project mount pose into yaw-only base odometry, full camera pose, and IMU pose."""
        t_world_imu = t_world_mount @ self.t_mount_imu
        t_world_camera = t_world_mount @ self.t_mount_camera
        t_world_base_link = t_world_imu @ self.t_imu_base_link
        yaw_only = odom_orientation_from_camera(t_world_camera[:3, :3])
        return (make_transform(t_world_base_link[:3, 3], yaw_only), t_world_camera, t_world_imu)

    def base_link_full_pose(self, t_world_imu: np.ndarray) -> np.ndarray:
        return t_world_imu @ self.t_imu_base_link


def optical_yaw(rotation_world_camera: np.ndarray) -> float:
    forward = rotation_world_camera[:, 2]
    return math.atan2(float(forward[1]), float(forward[0]))


def odom_orientation_from_camera(rotation_world_camera: np.ndarray) -> np.ndarray:
    return rot_z(optical_yaw(rotation_world_camera))


def rotate_twist(
    rotation: np.ndarray, linear: Sequence[float], angular: Sequence[float]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    world_linear = rotation @ np.asarray(linear, dtype=float)
    world_angular = rotation @ np.asarray(angular, dtype=float)
    return (
        (float(world_linear[0]), float(world_linear[1]), float(world_linear[2])),
        (float(world_angular[0]), float(world_angular[1]), float(world_angular[2])),
    )


def planar_twist(
    previous: np.ndarray, current: np.ndarray, dt: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Differentiate planar poses and express velocity in the current body frame."""
    if dt <= 0.0:
        raise ValueError(f"两次里程计的时间间隔必须 > 0,收到 {dt}")
    velocity_world = (current[:3, 3] - previous[:3, 3]) / dt
    linear = current[:3, :3].T @ velocity_world
    yaw_now = math.atan2(float(current[1, 0]), float(current[0, 0]))
    yaw_before = math.atan2(float(previous[1, 0]), float(previous[0, 0]))
    yaw_rate = math.remainder(yaw_now - yaw_before, 2.0 * math.pi) / dt
    return ((float(linear[0]), float(linear[1]), float(linear[2])), (0.0, 0.0, yaw_rate))


FRAME_URDF_ROOT = "base"
CAMERA_IN_BASE_LINK_XZ = (0.1582, 0.625)
REAL_IMU_TO_BASE_LINK_XYZ = (-0.151975, -0.04, -0.626543)
REAL_IMU_TO_BASE_LINK_PITCH_RAD = -0.009947
STAND_TRUNK_HEIGHT_M = 0.302
T_IMU_CAMERA = invert(T_CAM_BODY)
_REAL_BASE_LINK_CAMERA = (
    invert(make_transform(REAL_IMU_TO_BASE_LINK_XYZ, rot_y(REAL_IMU_TO_BASE_LINK_PITCH_RAD)))
    @ T_IMU_CAMERA
)
T_BASE_LINK_CAMERA = make_transform(
    (CAMERA_IN_BASE_LINK_XZ[0], _REAL_BASE_LINK_CAMERA[1, 3], CAMERA_IN_BASE_LINK_XZ[1]),
    _REAL_BASE_LINK_CAMERA[:3, :3],
)
T_BASE_LINK_TRUNK = make_transform((0.0, 0.0, STAND_TRUNK_HEIGHT_M))
GO2 = RobotFrames(
    name="go2",
    urdf_root=FRAME_URDF_ROOT,
    mount_link=FRAME_URDF_ROOT,
    t_base_link_camera=T_BASE_LINK_CAMERA,
    t_base_link_mount=T_BASE_LINK_TRUNK,
    t_base_link_urdf_root=T_BASE_LINK_TRUNK,
)
Position = Tuple[float, float, float]
QuaternionXYZW = Tuple[float, float, float, float]


def scene_to_sim_world(floor_z_scene_m: float) -> np.ndarray:
    """Preserve scene XY and heading while translating the spawn ground to zero."""
    if not math.isfinite(floor_z_scene_m):
        raise ValueError("仿真地板高度必须是有限数")
    return make_transform((0.0, 0.0, -float(floor_z_scene_m)))


def yaw_from_quat(quat_xyzw: Sequence[float]) -> float:
    (x, y, z, w) = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rpy_from_quat(quat_xyzw: Sequence[float]) -> Tuple[float, float, float]:
    (x, y, z, w) = (float(v) for v in quat_xyzw)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return (roll, pitch, yaw_from_quat(quat_xyzw))


def quat_from_yaw(yaw: float) -> QuaternionXYZW:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class OdomOrigin:
    """Gravity-aligned episode origin; resets change XY and yaw only."""

    def __init__(self, transform: np.ndarray) -> None:
        self._t = np.asarray(transform, dtype=float)
        self._t_inv = invert(self._t)
        self.is_identity = bool(np.allclose(self._t, np.eye(4), atol=1e-12))

    @classmethod
    def identity(cls) -> OdomOrigin:
        return cls(np.eye(4))

    @classmethod
    def from_pose(cls, position: Sequence[float], quat_xyzw: Sequence[float]) -> OdomOrigin:
        """Anchor the episode in XY and yaw while preserving ground-relative height."""
        return cls(
            pose_to_transform(
                (float(position[0]), float(position[1]), 0.0),
                quat_from_yaw(yaw_from_quat(quat_xyzw)),
            )
        )

    def transform(self) -> np.ndarray:
        return self._t.copy()

    def inverse(self) -> np.ndarray:
        return self._t_inv.copy()

    def pose(self) -> Tuple[Position, QuaternionXYZW]:
        (position, quat) = transform_to_pose(self._t)
        return (tuple(position), tuple(quat))

    def to_local_transform(self, t_world_x: np.ndarray) -> np.ndarray:
        if self.is_identity:
            return t_world_x
        return self._t_inv @ t_world_x

    def to_local_pose(
        self, position: Sequence[float], quat_xyzw: Sequence[float]
    ) -> Tuple[Position, QuaternionXYZW]:
        if self.is_identity:
            return (tuple(float(v) for v in position), tuple(float(v) for v in quat_xyzw))
        (pos, quat) = transform_to_pose(self._t_inv @ pose_to_transform(position, quat_xyzw))
        return (tuple(pos), tuple(quat))

    def to_local_xy(self, xy: Sequence[float]) -> Tuple[float, float]:
        p = self._t_inv @ np.array([float(xy[0]), float(xy[1]), 0.0, 1.0])
        return (float(p[0]), float(p[1]))

    def to_world_xy(self, xy: Sequence[float]) -> Tuple[float, float]:
        p = self._t @ np.array([float(xy[0]), float(xy[1]), 0.0, 1.0])
        return (float(p[0]), float(p[1]))

    def rotate_world_twist(
        self, linear: Sequence[float], angular: Sequence[float]
    ) -> Tuple[Position, Position]:
        if self.is_identity:
            return (tuple(float(v) for v in linear), tuple(float(v) for v in angular))
        return rotate_twist(self._t_inv[:3, :3], linear, angular)

    def describe(self) -> str:
        ((x, y, z), quat) = self.pose()
        (roll, pitch, yaw) = (math.degrees(v) for v in rpy_from_quat(quat))
        return f"xyz=({x:.3f}, {y:.3f}, {z:.3f}) m, rpy=({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)"
