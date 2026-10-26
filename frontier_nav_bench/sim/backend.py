"""Simulator backend protocol and a pure NumPy backend for tests and rehearsals.

ctrl_server.CtrlServer depends only on SimBackend. The Isaac implementation is in
isaac_backend.py. Backends execute already-validated requests with legal hold
durations and bounded velocities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from frontier_nav_bench.sim import codecs as nav_codecs
from frontier_nav_bench.sim import protocol as nav_protocol
from frontier_nav_bench.sim.ctrl_protocol import (
    E_POSE_INVALID,
    E_TELEPORT_UNSTABLE,
    TRACE_HZ,
    CtrlError,
    StepReply,
    TraceSample,
    WireObs,
    WirePose,
)

Cmd = Tuple[float, float, float]


class SimBackend(Protocol):
    def info(self) -> Dict[str, Any]: ...  # Scene, camera, backend and versions.

    def reset(self, x: float, y: float, yaw: float) -> WireObs: ...

    def step(self, cmd: Cmd, hold_s: float, want_frames: bool, want_third: bool) -> StepReply: ...

    def set_pose(self, x: float, y: float, yaw: float, settle_s: float) -> WireObs: ...

    def capture(self, warm_renders: int) -> WireObs: ...

    def truth(self) -> WirePose: ...

    def close(self) -> None: ...


def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def yaw_from_quat(q: Tuple[float, ...]) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---- Kinematic backend ----

KIN_WIDTH, KIN_HEIGHT = 640, 480  # Match the Isaac camera resolution.
KIN_K = (295.48, 0.0, 314.00, 0.0, 295.66, 242.41, 0.0, 0.0, 1.0)  # Camera K (plan 5.2).
KIN_PHYSICS_HZ = 200.0  # Match the Isaac physics integration frequency.
KIN_MAX_DEPTH_M = 8.0
KIN_FREE_DEPTH_M = 3.0  # Constant depth when no occupancy grid is supplied.


@dataclass
class KinematicFaults:
    """Inject poisoned-state, fall and teleport failures in tests."""

    raise_on_step: Optional[int] = None  # One-based step call that raises an exception.
    fall_at_s: Optional[float] = None  # Episode time at which up_z drops to 0.2.
    teleport_unstable: bool = False


class KinematicBackend:
    """Integrate body-frame velocities in a planar kinematic simulator.

    An optional occupancy grid blocks motion and supplies ray-cast depth.
    Images depend deterministically on pose for capture and deduplication tests.
    """

    def __init__(
        self,
        scene: str = "kinematic_test",
        occupancy: Optional[np.ndarray] = None,  # True = obstacle; rows y, columns x.
        origin_xy: Tuple[float, float] = (0.0, 0.0),  # World lower-left corner of cell [0, 0].
        cell_m: float = 0.1,
        faults: Optional[KinematicFaults] = None,
    ) -> None:
        self.scene = scene
        self.occupancy = occupancy
        self.origin_xy = origin_xy
        self.cell_m = cell_m
        self.faults = faults or KinematicFaults()
        self._x = self._y = self._yaw = 0.0
        self._v: Cmd = (0.0, 0.0, 0.0)
        self._t = 0.0
        self._steps = 0
        self.closed = False

    # -- Geometry --
    def _blocked(self, x: float, y: float) -> bool:
        if self.occupancy is None:
            return False
        col = int(math.floor((x - self.origin_xy[0]) / self.cell_m))
        row = int(math.floor((y - self.origin_xy[1]) / self.cell_m))
        h, w = self.occupancy.shape
        if not (0 <= row < h and 0 <= col < w):
            return True
        return bool(self.occupancy[row, col])

    def _ray(self, angle: float) -> float:
        if self.occupancy is None:
            return KIN_FREE_DEPTH_M
        step = self.cell_m / 2.0
        dist = step
        while dist < KIN_MAX_DEPTH_M:
            if self._blocked(self._x + dist * math.cos(angle), self._y + dist * math.sin(angle)):
                return dist
            dist += step
        return 0.0  # Out-of-range depth is invalid.

    # -- Rendering --
    def _render(self) -> Tuple[bytes, bytes, str]:
        fx, cx = KIN_K[0], KIN_K[2]
        cols = np.arange(KIN_WIDTH, dtype=np.float64)
        angles = self._yaw - np.arctan((cols - cx) / fx)
        ranges = np.array([self._ray(a) for a in angles[::8]], dtype=np.float32)
        ranges = np.repeat(ranges, 8)[:KIN_WIDTH]
        depth = np.tile(ranges, (KIN_HEIGHT, 1)).astype(np.float32)
        # Pose-dependent stripes: yaw controls hue, position controls brightness.
        hue = np.uint8(int((math.degrees(self._yaw) % 360.0) / 2.0))
        bgr = np.zeros((KIN_HEIGHT, KIN_WIDTH, 3), dtype=np.uint8)
        stripe = ((cols + 40.0 * self._x + 40.0 * self._y) // 32 % 2).astype(np.uint8)
        hsv = np.zeros_like(bgr)
        hsv[..., 0] = hue
        hsv[..., 1] = 200
        hsv[..., 2] = (90 + 120 * stripe)[None, :]
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        jpeg = nav_codecs.encode_rgb(bgr)
        depth_bytes, enc = nav_codecs.encode_depth(depth, nav_protocol.DEPTH_ENC_PNG)
        return jpeg, depth_bytes, enc

    def _pose(self) -> WirePose:
        up_z = 1.0
        if self.faults.fall_at_s is not None and self._t >= self.faults.fall_at_s - 1e-9:
            up_z = 0.2
        return WirePose(
            position_xyz=(self._x, self._y, 0.3),
            quat_xyzw=quat_from_yaw(self._yaw),
            v_body=(self._v[0], self._v[1], 0.0),
            w_body=(0.0, 0.0, self._v[2]),
            up_z=up_z,
        )

    def _obs(self) -> WireObs:
        jpeg, depth, enc = self._render()
        return WireObs(self._t, jpeg, depth, enc, KIN_WIDTH, KIN_HEIGHT, KIN_K, self._pose())

    # -- SimBackend --
    def info(self) -> Dict[str, Any]:
        return {
            "backend": "kinematic",
            "scene": self.scene,
            "width": KIN_WIDTH,
            "height": KIN_HEIGHT,
            "K": list(KIN_K),
            "trace_hz": TRACE_HZ,
        }

    def _place(self, x: float, y: float, yaw: float) -> None:
        if self._blocked(x, y):
            raise CtrlError(E_POSE_INVALID, f"({x:.2f}, {y:.2f}) 在障碍里或出界")
        self._x, self._y, self._yaw = float(x), float(y), float(yaw)
        self._v = (0.0, 0.0, 0.0)

    def reset(self, x: float, y: float, yaw: float) -> WireObs:
        self._place(x, y, yaw)
        self._t = 0.0
        return self._obs()

    def step(self, cmd: Cmd, hold_s: float, want_frames: bool, want_third: bool) -> StepReply:
        self._steps += 1
        if self.faults.raise_on_step is not None and self._steps == self.faults.raise_on_step:
            raise RuntimeError("注入的后端故障")
        vx, vy, wz = cmd
        self._v = (vx, vy, wz)
        per_trace = int(round(KIN_PHYSICS_HZ / TRACE_HZ))
        n_trace = int(round(hold_s * TRACE_HZ))
        dt = 1.0 / KIN_PHYSICS_HZ
        trace: List[TraceSample] = []
        frames: List[Tuple[float, bytes]] = []
        third: List[Tuple[float, bytes]] = []
        t0 = self._t
        for k in range(n_trace):
            for _ in range(per_trace):
                c, s = math.cos(self._yaw), math.sin(self._yaw)
                nx = self._x + (vx * c - vy * s) * dt
                ny = self._y + (vx * s + vy * c) * dt
                if not self._blocked(nx, ny):
                    self._x, self._y = nx, ny
                elif not self._blocked(nx, self._y):  # Slide using one component.
                    self._x = nx
                elif not self._blocked(self._x, ny):
                    self._y = ny
                self._yaw = math.atan2(math.sin(self._yaw + wz * dt), math.cos(self._yaw + wz * dt))
            self._t = round(t0 + (k + 1) / TRACE_HZ, 6)
            trace.append(TraceSample(self._t, self._pose()))
            if want_frames or want_third:
                jpeg = self._render()[0]
                if want_frames:
                    frames.append((self._t, jpeg))
                if want_third:
                    third.append((self._t, jpeg))
        obs = self._obs()
        return StepReply(
            obs=obs,
            trace=trace,
            frames=frames,
            third=third,
            applied_cmd=(vx, vy, wz),
            fallen_now=obs.pose.up_z < 0.5,
        )

    def set_pose(self, x: float, y: float, yaw: float, settle_s: float) -> WireObs:
        if self.faults.teleport_unstable:
            raise CtrlError(E_TELEPORT_UNSTABLE, "注入的瞬移失败")
        self._place(x, y, yaw)
        return self._obs()

    def capture(self, warm_renders: int) -> WireObs:
        return self._obs()

    def truth(self) -> WirePose:
        return self._pose()

    def close(self) -> None:
        self.closed = True
