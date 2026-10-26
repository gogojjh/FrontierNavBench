"""Calibrated RGB-D optics and optional collision-aware third-person camera."""

# Adapted from nav_sim_isaac/nav_sim_isaac/rendering/camera_lens.py.
# Source SHA256: a143f833eee08878daccf7b4461e41f01c9252d54d876a2cd80dffbc01b2cd05.
# Adapted from nav_sim_isaac/nav_sim_isaac/third_person_camera.py.
# Source SHA256: 25940c6b8bcc0eeef4d907f981ccc15a36fbf908aa1c510d2a5104b3e6c099e4.
# Copyright (c) 2026 Jianhao Jiao. MIT license; see LICENSE.
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

REFERENCE_FOCAL_LENGTH = 24.0
ODIN_NATIVE_WIDTH = 1600
ODIN_NATIVE_HEIGHT = 1296
ODIN_NATIVE_FX = 738.7118218386523
ODIN_NATIVE_FY = 739.1424877990535
ODIN_NATIVE_CX = 785.7409832015653
ODIN_NATIVE_CY = 654.7699723893847
REAL_RGBD_WIDTH = 640
REAL_RGBD_HEIGHT = 480


def habitat_intrinsics_from_native(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    out_width: int = REAL_RGBD_WIDTH,
    out_height: int = REAL_RGBD_HEIGHT,
) -> tuple[float, float, float, float]:
    """Center-crop and scale Odin calibration using pixel-center coordinates."""
    keep_height = int(round(width * out_height / float(out_width)))
    crop = height - keep_height
    if crop < 0:
        raise LensError(
            f"原图 {width}x{height} 裁不出 {out_width}x{out_height} 的宽高比(要 {keep_height} 行)"
        )
    top = crop // 2
    scale_x = out_width / float(width)
    scale_y = out_height / float(keep_height)
    return (
        fx * scale_x,
        fy * scale_y,
        (cx + 0.5) * scale_x - 0.5,
        (cy - top + 0.5) * scale_y - 0.5,
    )


(REAL_RGBD_FX, REAL_RGBD_FY, REAL_RGBD_CX, REAL_RGBD_CY) = habitat_intrinsics_from_native(
    ODIN_NATIVE_WIDTH,
    ODIN_NATIVE_HEIGHT,
    ODIN_NATIVE_FX,
    ODIN_NATIVE_FY,
    ODIN_NATIVE_CX,
    ODIN_NATIVE_CY,
)


class LensError(Exception):
    pass


def usd_lens_from_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    focal_length: float = REFERENCE_FOCAL_LENGTH,
) -> dict:
    """Convert focal pixels and principal point to USD lens attributes."""
    if min(fx, fy, focal_length) <= 0 or min(width, height) <= 0:
        raise LensError(
            f"Lens values must be positive: {fx}, {fy}, {width}, {height}, {focal_length}"
        )
    horizontal_aperture = focal_length * width / fx
    vertical_aperture = focal_length * height / fy
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    return {
        "focalLength": focal_length,
        "horizontalAperture": horizontal_aperture,
        "verticalAperture": vertical_aperture,
        "horizontalApertureOffset": (center_x - cx) * horizontal_aperture / width,
        "verticalApertureOffset": (cy - center_y) * vertical_aperture / height,
    }


def real_robot_lens(focal_length: float = REFERENCE_FOCAL_LENGTH) -> dict:
    """Return USD lens settings for the frozen 640×480 Odin RGB-D calibration."""
    return usd_lens_from_intrinsics(
        fx=REAL_RGBD_FX,
        fy=REAL_RGBD_FY,
        cx=REAL_RGBD_CX,
        cy=REAL_RGBD_CY,
        width=REAL_RGBD_WIDTH,
        height=REAL_RGBD_HEIGHT,
        focal_length=focal_length,
    )


(WIDTH, HEIGHT) = (960, 540)
HFOV_DEG = 80.0
BEHIND_M = 3.2
HEIGHT_M = 2.0
LOOK_AHEAD_M = 1.2
LOOK_HEIGHT_M = 0.3
FOCUS_SMOOTH_S = 0.6
HEADING_TIME_CONSTANT_S = 1.5
HEADING_MAX_RATE_DEG_S = 20.0
TRAVEL_WINDOW_S = 1.5
TRAVEL_MIN_DISPLACEMENT_M = 0.5
SAMPLE_HEIGHT_M = 0.4
SAMPLE_ALONG_M = (0.0, 0.3, -0.3)
HIDDEN_MIN_SAMPLES = 3
CLIP_MARGIN_M = 0.1
FLOOR_CLIP_MARGIN_M = 0.15
DOG_CLIP_GAP_M = 0.4
NEAR_HOLD_S = 1.0
APPROACH_ANGLE_DEG = 135.0
APPROACH_HOLD_S = 2.5
HEADING_SNAP_DEG = 90.0
OCCLUDED_SWITCH_S = 1.0
SWITCH_TURN_RATE_DEG_S = 30.0
SWITCH_MIN_DURATION_S = 2.0
SWITCH_COOLDOWN_S = 6.0
NEAR_POOR_M = 0.8
POOR_VIEW_SWITCH_S = 3.0
PREDICT_AHEAD_M = 1.0
OCCLUDED_SWITCH_ANGLE_DEG = 20.0
OCCLUDED_SEARCH_OFFSETS_DEG = (45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
_EPS_M = 1e-09
NEAR_CLIP_M = 0.05
FAR_CLIP_M = 200.0
RaycastFn = Callable[
    [Tuple[float, float, float], Tuple[float, float, float], float], Optional[float]
]


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    near_clip_m: float = NEAR_CLIP_M


@dataclass(frozen=True)
class _Visibility:
    hidden: int
    near_needed_m: float


def lens(
    width: int = WIDTH, height: int = HEIGHT, focal_length_mm: float = 18.0
) -> dict[str, float]:
    horizontal_aperture = 2.0 * focal_length_mm * math.tan(math.radians(HFOV_DEG) / 2.0)
    vertical_aperture = horizontal_aperture * height / width
    return {
        "focalLength": focal_length_mm,
        "horizontalAperture": horizontal_aperture,
        "verticalAperture": vertical_aperture,
        "near": NEAR_CLIP_M,
        "far": FAR_CLIP_M,
    }


_HALF_VFOV_RAD = math.atan(math.tan(math.radians(HFOV_DEG) / 2.0) * HEIGHT / WIDTH)


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _exp_smoothing_alpha(dt: float, time_constant_s: float) -> float:
    if dt <= 0.0:
        return 0.0
    return 1.0 - math.exp(-dt / time_constant_s)


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _smooth_damp(
    current: float, velocity: float, target: float, smooth_time_s: float, dt: float
) -> tuple[float, float]:
    if dt <= 0.0:
        return (current, velocity)
    omega = 2.0 / smooth_time_s
    x = omega * dt
    decay = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    temp = (velocity + omega * change) * dt
    velocity = (velocity - omega * temp) * decay
    return (target + (change + temp) * decay, velocity)


def _look_at_quaternion(
    position: tuple[float, float, float], look_at: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    dx = look_at[0] - position[0]
    dy = look_at[1] - position[1]
    dz = look_at[2] - position[2]
    horizontal = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, horizontal)
    (half_yaw, half_pitch) = (yaw / 2.0, pitch / 2.0)
    (cy, sy) = (math.cos(half_yaw), math.sin(half_yaw))
    (cp, sp) = (math.cos(half_pitch), math.sin(half_pitch))
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def _unit(vector: tuple[float, float, float]) -> tuple[tuple[float, float, float], float]:
    length = math.sqrt(sum(v * v for v in vector))
    if length <= _EPS_M:
        return ((0.0, 0.0, 0.0), 0.0)
    return ((vector[0] / length, vector[1] / length, vector[2] / length), length)


class FollowCamera:
    """Smooth the third-person viewpoint while avoiding scene occlusions."""

    def __init__(self, raycast: RaycastFn) -> None:
        self._raycast = raycast
        self._clock = 0.0
        self._heading = 0.0
        self._target_heading = 0.0
        self._base_yaw = 0.0
        self._focus = (0.0, 0.0)
        self._focus_velocity = (0.0, 0.0)
        self._position: tuple[float, float, float] = (0.0, 0.0, HEIGHT_M)
        self._history: deque = deque()
        self._approach_since: float | None = None
        self._occluded_since: float | None = None
        self._poor_since: float | None = None
        self._last_switch = -math.inf
        self._switch: tuple[float, float, float, float] | None = None
        self._near = NEAR_CLIP_M
        self._near_until = 0.0

    def reset(self, robot_xy: tuple[float, float], ground_z: float, yaw: float) -> CameraPose:
        self._clock = 0.0
        self._heading = yaw
        self._target_heading = yaw
        self._base_yaw = yaw
        self._focus = (robot_xy[0], robot_xy[1])
        self._focus_velocity = (0.0, 0.0)
        self._history = deque([(0.0, robot_xy)])
        candidates = [yaw]
        candidates += [_wrap_to_pi(yaw + math.radians(o)) for o in OCCLUDED_SEARCH_OFFSETS_DEG]
        start = min(
            candidates,
            key=lambda h: (
                self._view_score(h, [robot_xy], ground_z, yaw),
                abs(_wrap_to_pi(h - yaw)),
            ),
        )
        self._heading = self._target_heading = start
        self._position = self._desired_position(ground_z)
        self._approach_since = None
        self._occluded_since = None
        self._poor_since = None
        self._last_switch = -math.inf
        self._switch = None
        self._near = NEAR_CLIP_M
        self._near_until = 0.0
        return self._finalize(robot_xy, ground_z, yaw)

    def update(
        self, dt: float, robot_xy: tuple[float, float], ground_z: float, yaw: float
    ) -> CameraPose:
        dt = max(dt, 0.0)
        self._clock += dt
        self._history.append((self._clock, robot_xy))
        while len(self._history) > 2 and self._clock - self._history[1][0] >= TRAVEL_WINDOW_S:
            self._history.popleft()
        ((fx, vx), (fy, vy)) = (
            _smooth_damp(self._focus[i], self._focus_velocity[i], robot_xy[i], FOCUS_SMOOTH_S, dt)
            for i in range(2)
        )
        (self._focus, self._focus_velocity) = ((fx, fy), (vx, vy))
        self._maybe_start_switch(robot_xy, ground_z, yaw)
        if self._switch is not None:
            (started, start_heading, turn, duration) = self._switch
            progress = min((self._clock - started) / duration, 1.0)
            self._heading = _wrap_to_pi(start_heading + turn * _smoothstep(progress))
            if progress >= 1.0:
                self._switch = None
        else:
            self._update_target_heading()
            alpha = _exp_smoothing_alpha(dt, HEADING_TIME_CONSTANT_S)
            delta = _wrap_to_pi(self._target_heading - self._heading)
            max_step = math.radians(HEADING_MAX_RATE_DEG_S) * dt
            step = max(-max_step, min(max_step, alpha * delta))
            self._heading = _wrap_to_pi(self._heading + step)
        self._position = self._desired_position(ground_z)
        return self._finalize(robot_xy, ground_z, yaw)

    def _desired_position(
        self, ground_z: float, heading: float | None = None
    ) -> tuple[float, float, float]:
        h = self._heading if heading is None else heading
        return (
            self._focus[0] - math.cos(h) * BEHIND_M,
            self._focus[1] - math.sin(h) * BEHIND_M,
            ground_z + HEIGHT_M,
        )

    def _look_at(self, ground_z: float, heading: float | None = None) -> tuple[float, float, float]:
        h = self._heading if heading is None else heading
        return (
            self._focus[0] + math.cos(h) * LOOK_AHEAD_M,
            self._focus[1] + math.sin(h) * LOOK_AHEAD_M,
            ground_z + LOOK_HEIGHT_M,
        )

    def _travel_direction(self) -> float | None:
        (t0, (x0, y0)) = self._history[0]
        (t1, (x1, y1)) = self._history[-1]
        if t1 - t0 < TRAVEL_WINDOW_S * 0.8:
            return None
        (dx, dy) = (x1 - x0, y1 - y0)
        if math.hypot(dx, dy) < TRAVEL_MIN_DISPLACEMENT_M:
            return None
        return math.atan2(dy, dx)

    def _maybe_start_switch(
        self, robot_xy: tuple[float, float], ground_z: float, yaw: float
    ) -> None:
        if self._switch is not None or self._clock - self._last_switch < SWITCH_COOLDOWN_S:
            return
        occluded = (
            self._occluded_since is not None
            and self._clock - self._occluded_since >= OCCLUDED_SWITCH_S
        )
        poor = self._poor_since is not None and self._clock - self._poor_since >= POOR_VIEW_SWITCH_S
        if not (occluded or poor):
            return
        if occluded:
            self._occluded_since = self._clock
        if poor:
            self._poor_since = self._clock
        better = self._find_clear_heading(robot_xy, ground_z, yaw, self._travel_direction())
        if better is None:
            return
        turn = _wrap_to_pi(better - self._heading)
        duration = max(SWITCH_MIN_DURATION_S, abs(math.degrees(turn)) / SWITCH_TURN_RATE_DEG_S)
        self._switch = (self._clock, self._heading, turn, duration)
        self._target_heading = better
        self._approach_since = None
        self._last_switch = self._clock

    def _update_target_heading(self) -> None:
        travel = self._travel_direction()
        approaching = travel is not None and abs(
            _wrap_to_pi(travel - self._target_heading)
        ) > math.radians(APPROACH_ANGLE_DEG)
        if not approaching:
            self._approach_since = None
        elif self._approach_since is None:
            self._approach_since = self._clock
        elif self._clock - self._approach_since >= APPROACH_HOLD_S:
            step = math.radians(HEADING_SNAP_DEG)
            k = round(_wrap_to_pi(travel - self._base_yaw) / step)
            self._target_heading = _wrap_to_pi(self._base_yaw + k * step)
            self._approach_since = None

    def _view_score(
        self, heading: float, spots: list, ground_z: float, yaw: float
    ) -> tuple[int, int]:
        position = self._desired_position(ground_z, heading)
        look_at = self._look_at(ground_z, heading)
        hidden = poor = 0
        for spot in spots:
            visibility = self._visibility(position, look_at, spot, ground_z, yaw)
            hidden += visibility.hidden
            poor += visibility.near_needed_m > NEAR_POOR_M
        return (hidden, poor)

    def _find_clear_heading(
        self, robot_xy: tuple[float, float], ground_z: float, yaw: float, travel: float | None
    ) -> float | None:
        spots = [robot_xy]
        if travel is not None:
            spots.append(
                (
                    robot_xy[0] + math.cos(travel) * PREDICT_AHEAD_M,
                    robot_xy[1] + math.sin(travel) * PREDICT_AHEAD_M,
                )
            )
        current = self._view_score(self._target_heading, spots, ground_z, yaw)
        candidates = [] if travel is None else [travel]
        candidates += [
            _wrap_to_pi(self._target_heading + math.radians(o)) for o in OCCLUDED_SEARCH_OFFSETS_DEG
        ]
        (best, best_key) = (None, (current, 0.0))
        for heading in candidates:
            turn = abs(_wrap_to_pi(heading - self._target_heading))
            if turn < math.radians(OCCLUDED_SWITCH_ANGLE_DEG):
                continue
            key = (self._view_score(heading, spots, ground_z, yaw), turn)
            if key[0] < current and key < best_key:
                (best, best_key) = (heading, key)
        return best

    def _samples(self, robot_xy: tuple[float, float], ground_z: float, yaw: float) -> list:
        return [
            (
                robot_xy[0] + math.cos(yaw) * along,
                robot_xy[1] + math.sin(yaw) * along,
                ground_z + SAMPLE_HEIGHT_M,
            )
            for along in SAMPLE_ALONG_M
        ]

    def _visibility(
        self,
        position: tuple[float, float, float],
        look_at: tuple[float, float, float],
        robot_xy: tuple[float, float],
        ground_z: float,
        yaw: float,
    ) -> _Visibility:
        (forward, _) = _unit(tuple(look_at[i] - position[i] for i in range(3)))
        samples = self._samples(robot_xy, ground_z, yaw)
        dog_depth = min(sum((s[i] - position[i]) * forward[i] for i in range(3)) for s in samples)
        bottom = math.asin(max(-1.0, min(1.0, -forward[2]))) + _HALF_VFOV_RAD
        floor_depth = (
            (position[2] - ground_z) / math.sin(bottom) * math.cos(_HALF_VFOV_RAD)
            if bottom > _EPS_M
            else math.inf
        )
        capacity = min(floor_depth - FLOOR_CLIP_MARGIN_M, dog_depth - DOG_CLIP_GAP_M)
        hidden = 0
        near = NEAR_CLIP_M
        for sample in samples:
            (direction, length) = _unit(tuple(position[i] - sample[i] for i in range(3)))
            if length <= _EPS_M:
                continue
            hit = self._raycast(sample, direction, length)
            if hit is None:
                continue
            occluder = tuple(sample[i] + direction[i] * hit for i in range(3))
            needed = sum((occluder[i] - position[i]) * forward[i] for i in range(3)) + CLIP_MARGIN_M
            if needed <= capacity:
                near = max(near, needed)
            else:
                hidden += 1
        return _Visibility(hidden=hidden, near_needed_m=near)

    def _finalize(self, robot_xy: tuple[float, float], ground_z: float, yaw: float) -> CameraPose:
        look_at = self._look_at(ground_z)
        visibility = self._visibility(self._position, look_at, robot_xy, ground_z, yaw)
        if visibility.hidden >= HIDDEN_MIN_SAMPLES:
            if self._occluded_since is None:
                self._occluded_since = self._clock
        else:
            self._occluded_since = None
        if visibility.near_needed_m > NEAR_POOR_M:
            if self._poor_since is None:
                self._poor_since = self._clock
        else:
            self._poor_since = None
        if visibility.near_needed_m >= self._near or self._clock >= self._near_until:
            if visibility.near_needed_m > NEAR_CLIP_M:
                self._near_until = self._clock + NEAR_HOLD_S
            self._near = visibility.near_needed_m
        return CameraPose(
            position=self._position,
            orientation_wxyz=_look_at_quaternion(self._position, look_at),
            near_clip_m=self._near,
        )
