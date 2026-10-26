"""Shared temporary topology-map fixtures using the offline OpenNavMap format.

Keyframes follow an L corridor along +x then +y. Body poses are specified as
(x, y, yaw); cameras sit CAMERA_FORWARD_M ahead and 0.3 m above ground, with
horizontal optical axes pointing along yaw.
"""

import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from vnav_benchmark.interfaces import LocResult, Observation, OdomState, Pose2D
from vnav_benchmark.methods.base_topo import wrap_angle

CAMERA_FORWARD_M = 0.1582
CAMERA_HEIGHT_M = 0.3
IMAGE_HW = (48, 64)  # Small images suffice for unit tests

# L corridor: one frame per meter from (0,0)->(4,0) facing +x, then (4,1)->(4,4) facing +y
L_CORRIDOR: Tuple[Tuple[float, float, float], ...] = tuple(
    [(float(i), 0.0, 0.0) for i in range(5)] + [(4.0, float(j), math.pi / 2) for j in range(1, 5)]
)


def _rotation_to_quat_wxyz(rot: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(np.trace(rot))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return (
            0.25 * s,
            (rot[2, 1] - rot[1, 2]) / s,
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[1, 0] - rot[0, 1]) / s,
        )
    i = int(np.argmax(np.diag(rot)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (rot[k, j] - rot[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (rot[j, i] + rot[i, j]) / s
    q[1 + k] = (rot[k, i] + rot[i, k]) / s
    return q[0], q[1], q[2], q[3]


def camera_pose_line(name: str, x: float, y: float, yaw: float) -> str:
    """Convert a planar body pose into a poses.txt row: task-to-camera wxyz quaternion and
    translation.
    """
    cam_in_map = np.eye(4)
    cam_in_map[:3, 0] = (math.sin(yaw), -math.cos(yaw), 0.0)  # Camera x points right
    cam_in_map[:3, 1] = (0.0, 0.0, -1.0)  # Camera y points down
    cam_in_map[:3, 2] = (math.cos(yaw), math.sin(yaw), 0.0)  # Optical axis
    cam_in_map[:3, 3] = (
        x + CAMERA_FORWARD_M * math.cos(yaw),
        y + CAMERA_FORWARD_M * math.sin(yaw),
        CAMERA_HEIGHT_M,
    )
    world_to_cam = np.linalg.inv(cam_in_map)
    qw, qx, qy, qz = _rotation_to_quat_wxyz(world_to_cam[:3, :3])
    tx, ty, tz = world_to_cam[:3, 3]
    return f"{name} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f}"


def make_map(
    root: Path,
    poses: Sequence[Tuple[float, float, float]] = L_CORRIDOR,
    trav_edges: Sequence[Tuple[int, int]] = (),
) -> Path:
    """Write an offline map under root and return its directory.

    edges_odom / edges_trav.chain contain adjacent-frame chains; edges_trav adds
    trav_edges. Deliberately include truth files and path-bearing map_meta.json
    to verify the sanitized view does not leak them.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "seq").mkdir(exist_ok=True)
    pose_lines: List[str] = []
    for index, (x, y, yaw) in enumerate(poses):
        name = f"seq/{index:06d}.color.jpg"
        color = np.full(IMAGE_HW + (3,), (index * 23) % 256, dtype=np.uint8)
        cv2.imwrite(str(root / name), color)
        depth = np.full(IMAGE_HW, 1000 + index, dtype=np.uint16)  # 1.000 m + index mm
        cv2.imwrite(str(root / f"seq/{index:06d}.depth.png"), depth)
        pose_lines.append(camera_pose_line(name, x, y, yaw))
    names = [line.split()[0] for line in pose_lines]
    (root / "poses.txt").write_text("\n".join(pose_lines) + "\n")
    (root / "intrinsics.txt").write_text(
        "".join(f"{name} 29.5 29.5 32.0 24.0 64 48\n" for name in names)
    )
    (root / "timestamps.txt").write_text(
        "".join(f"{name} {i * 100.0:.6f}\n" for i, name in enumerate(names))
    )
    (root / "database_descriptors.txt").write_text(
        "".join(f"{name} {i * 0.1:.3f} 0.5\n" for i, name in enumerate(names))
    )
    chain = [(i, i + 1) for i in range(len(poses) - 1)]
    edge_text = "".join(f"{i} {j} 1.0\n" for i, j in chain)
    (root / "edges_odom.txt").write_text(edge_text)
    (root / "edges_covis.txt").write_text(edge_text)
    (root / "edges_trav.chain.txt").write_text(edge_text)
    (root / "edges_trav.txt").write_text(
        edge_text + "".join(f"{i} {j} 1.0\n" for i, j in trav_edges)
    )
    (root / "edges_room.txt").write_text("room_0 0 1.0\n")
    (root / "poses_abs_gt.txt").write_text("truth\n")
    (root / "gps_data.txt").write_text("truth\n")
    (root / "objects.json").write_text("{}")
    meta = {
        "run_dir": "/somewhere/session",
        "sensors_dir": "/somewhere/session/sensors",
        "num_keyframes": len(poses),
        "intrinsics": {"fx": 29.5, "fy": 29.5, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48},
        "overrides": ["keyframe.trans_thresh_m=1.0", "boxer.url=http://127.0.0.1:9693"],
        "schema_version": "2.2",
        "opennavmap_git_sha": "abc",
    }
    (root / "map_meta.json").write_text(json.dumps(meta))
    return root


# ---------------------------------------- Fake simulation/localization/point controller

DT_S = 0.05  # Fake-simulation integration step
RGB_HW = (48, 64)


def _odom_state(pose: Pose2D, v_body: Tuple[float, float, float], wz: float) -> OdomState:
    from vnav_benchmark.interfaces import OdomState

    half = pose.yaw / 2.0
    return OdomState(
        pose=pose,
        position_xyz=(pose.x, pose.y, 0.3),
        quat_xyzw=(0.0, 0.0, math.sin(half), math.cos(half)),
        v_body=v_body,
        w_body=(0.0, 0.0, wz),
    )


class FakeWorld:
    """Planar kinematic simulator with world=map and spawn-relative odometry, matching SimEnv."""

    def __init__(self, start: Tuple[float, float, float], budget_s: float = 120.0) -> None:
        self.pose = Pose2D(*start)
        self.start = self.pose
        self.sim_time_s = 0.0
        self.budget_s = budget_s
        self.step_index = 0
        self.last_cmd = (0.0, 0.0, 0.0)
        self.path_m = 0.0

    def odom_pose(self) -> Pose2D:
        dx, dy = self.pose.x - self.start.x, self.pose.y - self.start.y
        c, s = math.cos(self.start.yaw), math.sin(self.start.yaw)
        yaw = wrap_angle(self.pose.yaw - self.start.yaw)
        return Pose2D(c * dx + s * dy, -s * dx + c * dy, yaw)

    def observation(self) -> Observation:
        vx, vy, wz = self.last_cmd
        depth = np.full(RGB_HW, 3.0, dtype=np.float32)
        return Observation(
            step_index=self.step_index,
            sim_time_s=self.sim_time_s,
            budget_left_s=self.budget_s - self.sim_time_s,
            rgb=np.zeros(RGB_HW + (3,), dtype=np.uint8),
            rgb_jpeg=b"jpeg",
            depth_m=depth,
            K=(29.5, 0.0, 32.0, 0.0, 29.5, 24.0, 0.0, 0.0, 1.0),
            odom=_odom_state(self.odom_pose(), (vx, vy, 0.0), wz),
        )

    def step(self, vx: float, vy: float, wz: float, hold_s: float) -> None:
        for _ in range(int(round(hold_s / DT_S))):
            x, y, yaw = self.pose
            c, s = math.cos(yaw), math.sin(yaw)
            nx, ny = x + (c * vx - s * vy) * DT_S, y + (s * vx + c * vy) * DT_S
            self.path_m += math.hypot(nx - x, ny - y)
            self.pose = Pose2D(nx, ny, wrap_angle(yaw + wz * DT_S))
        self.sim_time_s += hold_s
        self.step_index += 1
        self.last_cmd = (vx, vy, wz)


class TruthLocalizer:
    """Ideal localization uses current fake truth for self and the supplied goal pose; count calls.
    """

    def __init__(self, world: FakeWorld, goal: Tuple[float, float, float]) -> None:
        self.world = world
        self.goal = Pose2D(*goal)
        self.calls: List[str] = []

    def localize(self, rgb_jpeg: bytes, depth_m: object, K: object, *, purpose: str) -> LocResult:
        self.calls.append(purpose)
        pose = self.goal if purpose == "goal" else self.world.pose
        return LocResult(True, pose, None, "")


def unicycle_to(
    robot: Tuple[float, float, float], target: Tuple[float, float]
) -> Tuple[float, float, float]:
    """Fake point controller turns first, moving below 0.5 rad yaw error within SRU simulation
    limits.
    """
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    err = wrap_angle(math.atan2(dy, dx) - robot[2])
    wz = max(-0.7, min(0.7, 1.5 * err))
    vx = 0.0 if abs(err) > 0.5 else min(0.8, math.hypot(dx, dy))
    return vx, 0.0, wz
