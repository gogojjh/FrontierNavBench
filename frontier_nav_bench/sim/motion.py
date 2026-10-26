"""Frozen-scene moving parts and the env3 elevator state machine.

Env3 uses immediate hall calls and the recorded boarding hold. Physics poses use
kinematic targets; visual poses synchronize through Fabric at the original rate.
The robot pose callback must include height and yaw so floor and boarding checks work."""

# Adapted from nav_sim_isaac/nav_sim_isaac/part_motion.py.
# Source SHA256: 60b521546172178931e6b6fd3478584359163a1d8049fdc2884496924c6160cc.
# Copyright (c) 2026 Jianhao Jiao. MIT license; see LICENSE.
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence

from frontier_nav_bench.paths import data_root
from frontier_nav_bench.sim.motion_records import (
    ELEV_BUTTON_PRESS_M,
    ELEV_DOOR_MOVE_S,
    ROBOT_RADIUS_M,
    USD_ELEV_ATTR_PREFIX,
    USD_OBJ_ATTR_PREFIX,
    USD_PART_ATTR_PREFIX,
    ElevatorRecord,
    ObjRecord,
    PartRecord,
    parse_elevator_attrs,
    parse_obj_attrs,
    parse_part_attrs,
)

DECISION_PERIOD_S = 0.05
CLOSE_CLEAR_S = 1.0
OBJECT_MOVED_M = 0.02
_EASE_TOL = 1e-09
PART_EVENT_DIR = data_root() / "sim_runs/part_events"
OBJECT_CHECK_PERIOD_S = 1.0
POSE_WRITE_MODE = "kinematic_target"
VISUAL_SYNC_USD = False
AUTO_MOTION = False


def start_positions(records: Sequence[PartRecord]) -> dict:
    return {r.id: r.open_position if r.parent is None else 0.0 for r in records}


VISUAL_SYNC_PERIOD_S = 0.05


class PartPhase(str, Enum):
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"


@dataclass(frozen=True)
class PartEvent:
    t_s: float
    part_id: str
    kind: str
    reason: str | None
    joint_position: float
    robot_xy: tuple[float, float] | None

    def to_dict(self) -> dict:
        return {
            "t_s": self.t_s,
            "part_id": self.part_id,
            "kind": self.kind,
            "reason": self.reason,
            "joint_position": self.joint_position,
            "robot_xy": list(self.robot_xy) if self.robot_xy is not None else None,
        }


def _ease(u: float) -> float:
    return 3.0 * u * u - 2.0 * u * u * u


def _dist_point_to_rect(rect: tuple[float, float, float, float], xy: tuple[float, float]) -> float:
    (x0, y0, x1, y1) = rect
    (x, y) = xy
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def _circle_hits_rect(
    center: tuple[float, float], radius: float, rect: tuple[float, float, float, float]
) -> bool:
    return _dist_point_to_rect(rect, center) < radius


class PartMotion:
    def __init__(self, record: PartRecord) -> None:
        self.record = record
        self.phase = PartPhase.CLOSED
        self.paused = False
        self._u = 0.0
        self._last_t_s: float | None = None
        self._next_decision_t_s: float | None = None
        self._open_since_s: float | None = None
        self._clear_since_s: float | None = None
        self._open_blocked_reported = False

    @property
    def joint_position(self) -> float:
        return self.record.open_position * _ease(self._u)

    def _hits_sweep(self, robot_xy: tuple[float, float] | None) -> bool:
        if robot_xy is None:
            return True
        return _circle_hits_rect(robot_xy, ROBOT_RADIUS_M, self.record.sweep_xy)

    def step(self, now_s: float, robot_xy: tuple[float, float] | None) -> list[PartEvent]:
        events: list[PartEvent] = []
        dt = 0.0 if self._last_t_s is None else max(0.0, now_s - self._last_t_s)
        self._last_t_s = now_s
        if self.phase in (PartPhase.OPENING, PartPhase.CLOSING):
            events += self._step_motion(now_s, robot_xy, dt)
        if self._next_decision_t_s is None or now_s >= self._next_decision_t_s:
            self._next_decision_t_s = now_s + DECISION_PERIOD_S
            events += self._decide(now_s, robot_xy)
        return events

    def _step_motion(
        self, now_s: float, robot_xy: tuple[float, float] | None, dt: float
    ) -> list[PartEvent]:
        events: list[PartEvent] = []
        colliding = self._hits_sweep(robot_xy)
        if colliding and (not self.paused):
            self.paused = True
            events.append(self._event(now_s, "paused", "blocked_by_robot", robot_xy))
        elif not colliding and self.paused:
            self.paused = False
            events.append(self._event(now_s, "resumed", None, robot_xy))
        if self.paused or self.record.move_s <= 0.0:
            return events
        step = dt / self.record.move_s
        if self.phase is PartPhase.OPENING:
            self._u = min(1.0, self._u + step)
            if self._u >= 1.0 - _EASE_TOL:
                self._u = 1.0
                self.phase = PartPhase.OPEN
                self._open_since_s = now_s
                self._clear_since_s = None
                events.append(self._event(now_s, "open_done", None, robot_xy))
        else:
            self._u = max(0.0, self._u - step)
            if self._u <= _EASE_TOL:
                self._u = 0.0
                self.phase = PartPhase.CLOSED
                events.append(self._event(now_s, "close_done", None, robot_xy))
        return events

    def _decide(self, now_s: float, robot_xy: tuple[float, float] | None) -> list[PartEvent]:
        if robot_xy is None:
            return []
        dist_trigger = math.hypot(
            robot_xy[0] - self.record.trigger_xy[0], robot_xy[1] - self.record.trigger_xy[1]
        )
        in_open_radius = dist_trigger <= self.record.open_radius_m
        hits_sweep = self._hits_sweep(robot_xy)
        if not in_open_radius:
            self._open_blocked_reported = False
        if self.phase is PartPhase.CLOSED:
            if not in_open_radius:
                return []
            if hits_sweep:
                if self._open_blocked_reported:
                    return []
                self._open_blocked_reported = True
                return [self._event(now_s, "open_blocked", "blocked_by_robot", robot_xy)]
            self.phase = PartPhase.OPENING
            self.paused = False
            return [self._event(now_s, "open_start", None, robot_xy)]
        if self.phase is PartPhase.OPEN:
            if dist_trigger > self.record.close_radius_m:
                if self._clear_since_s is None:
                    self._clear_since_s = now_s
            else:
                self._clear_since_s = None
            held_long_enough = (
                self._open_since_s is not None and now_s - self._open_since_s >= self.record.hold_s
            )
            clear_long_enough = (
                self._clear_since_s is not None and now_s - self._clear_since_s >= CLOSE_CLEAR_S
            )
            if held_long_enough and clear_long_enough and (not hits_sweep):
                self.phase = PartPhase.CLOSING
                self.paused = False
                return [self._event(now_s, "close_start", None, robot_xy)]
            return []
        if self.phase is PartPhase.CLOSING:
            if in_open_radius and (not hits_sweep):
                self.phase = PartPhase.OPENING
                self.paused = False
                return [self._event(now_s, "open_start", "robot_returned", robot_xy)]
            return []
        return []

    def _event(
        self, now_s: float, kind: str, reason: str | None, robot_xy: tuple[float, float] | None
    ) -> PartEvent:
        return PartEvent(
            t_s=now_s,
            part_id=self.record.id,
            kind=kind,
            reason=reason,
            joint_position=self.joint_position,
            robot_xy=robot_xy,
        )


class PartMotionSet:
    def __init__(self, records: Sequence[PartRecord]) -> None:
        self._motions: dict[str, PartMotion] = {r.id: PartMotion(r) for r in records}

    def step(self, now_s: float, robot_xy: tuple[float, float] | None) -> list[PartEvent]:
        events: list[PartEvent] = []
        for part_id in sorted(self._motions):
            events.extend(self._motions[part_id].step(now_s, robot_xy))
        return events

    def moving(self) -> bool:
        return any(
            m.phase in (PartPhase.OPENING, PartPhase.CLOSING) for m in self._motions.values()
        )

    def positions(self) -> dict:
        return {part_id: m.joint_position for (part_id, m) in self._motions.items()}


class ObjectWatch:
    def __init__(self) -> None:
        self._last: dict[str, tuple[float, float, float]] = {}

    def update(
        self, now_s: float, positions: Mapping[str, tuple[float, float, float]]
    ) -> list[dict]:
        events: list[dict] = []
        for obj_id, pos in positions.items():
            prev = self._last.get(obj_id)
            if prev is None:
                self._last[obj_id] = pos
                continue
            dist = math.sqrt(sum((a - b) ** 2 for (a, b) in zip(pos, prev)))
            if dist >= OBJECT_MOVED_M:
                events.append(
                    {
                        "t_s": now_s,
                        "obj_id": obj_id,
                        "kind": "object_moved",
                        "from": list(prev),
                        "to": list(pos),
                        "dist_m": round(dist, 6),
                    }
                )
                self._last[obj_id] = pos
        return events


ROBOT_BODY_SIZE_M = (0.7, 0.31)
ROBOT_BODY_SIZES_M = {"go2": ROBOT_BODY_SIZE_M}


def robot_body_size(robot_name: str) -> tuple:
    """Return the frozen Go2 footprint used for doorway and full-boarding checks."""
    return ROBOT_BODY_SIZES_M.get(robot_name, ROBOT_BODY_SIZE_M)


ENV3_BOARDING_HOLD_S = 10.0
ELEV_MOVE_ACCEL_S = 1.0


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    z: float
    yaw_deg: float
    body_size_m: tuple = ROBOT_BODY_SIZE_M


def _elevator_trapezoid_time(
    distance_m: float, speed_mps: float, accel_s: float = ELEV_MOVE_ACCEL_S
) -> float:
    if distance_m <= 0.0 or speed_mps <= 0.0:
        return 0.0
    accel_dist = 0.5 * speed_mps * accel_s
    if distance_m >= 2.0 * accel_dist:
        cruise_dist = distance_m - 2.0 * accel_dist
        return 2.0 * accel_s + cruise_dist / speed_mps
    peak_speed = math.sqrt(distance_m * speed_mps / accel_s)
    return 2.0 * (peak_speed / speed_mps) * accel_s


def _elevator_trapezoid_distance(
    elapsed_s: float, distance_m: float, speed_mps: float, accel_s: float = ELEV_MOVE_ACCEL_S
) -> float:
    if distance_m <= 0.0 or speed_mps <= 0.0 or elapsed_s <= 0.0:
        return 0.0
    total_time = _elevator_trapezoid_time(distance_m, speed_mps, accel_s)
    if elapsed_s >= total_time:
        return distance_m
    accel_dist = 0.5 * speed_mps * accel_s
    if distance_m >= 2.0 * accel_dist:
        cruise_dist = distance_m - 2.0 * accel_dist
        cruise_time = cruise_dist / speed_mps
        a = speed_mps / accel_s
        if elapsed_s <= accel_s:
            return 0.5 * a * elapsed_s * elapsed_s
        if elapsed_s <= accel_s + cruise_time:
            return accel_dist + speed_mps * (elapsed_s - accel_s)
        t_decel = elapsed_s - accel_s - cruise_time
        return accel_dist + cruise_dist + (speed_mps * t_decel - 0.5 * a * t_decel * t_decel)
    peak_speed = math.sqrt(distance_m * speed_mps / accel_s)
    half_time = peak_speed / speed_mps * accel_s
    a = peak_speed / half_time
    if elapsed_s <= half_time:
        return 0.5 * a * elapsed_s * elapsed_s
    t_decel = elapsed_s - half_time
    return distance_m / 2.0 + (peak_speed * t_decel - 0.5 * a * t_decel * t_decel)


def _robot_corners(pose: RobotPose) -> list[tuple[float, float]]:
    (half_l, half_w) = (s / 2.0 for s in pose.body_size_m)
    rad = math.radians(pose.yaw_deg)
    (cos_a, sin_a) = (math.cos(rad), math.sin(rad))
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            (lx, ly) = (sx * half_l, sy * half_w)
            corners.append((pose.x + lx * cos_a - ly * sin_a, pose.y + lx * sin_a + ly * cos_a))
    return corners


def _point_in_rect(xy: tuple[float, float], rect: tuple[float, float, float, float]) -> bool:
    (x0, x1, y0, y1) = rect
    return x0 - 1e-09 <= xy[0] <= x1 + 1e-09 and y0 - 1e-09 <= xy[1] <= y1 + 1e-09


def _all_corners_in(corners: Sequence[tuple[float, float]], rect) -> bool:
    return all(_point_in_rect(c, rect) for c in corners)


def _any_corner_in(corners: Sequence[tuple[float, float]], rect) -> bool:
    return any(_point_in_rect(c, rect) for c in corners)


def _robot_points(pose: RobotPose) -> list[tuple[float, float]]:
    return _robot_corners(pose) + [(pose.x, pose.y)]


class ElevatorPhase(str, Enum):
    IDLE = "idle"
    DOORS_OPENING = "doors_opening"
    DOORS_OPEN = "doors_open"
    DOORS_CLOSING = "doors_closing"
    MOVING = "moving"


DOOR_FRONT_DEPTH_M = 0.9
DOOR_FRONT_SIDE_MARGIN_M = 0.1


def _door_front_rect(record: ElevatorRecord) -> tuple[float, float, float, float]:
    """Derive a hall-side rectangle from the USD record, without scene-spec imports.

    Upstream centers the cab and door on the same along-wall coordinate. Their
    center difference therefore identifies the door normal and the hall direction.
    door_zone spans the cab front to the hall wall face; its outer edge is the
    exact start of our 0.9 m strip. Widen the door opening by 0.1 m on each side.
    Both geometries are shared by the served floors, as in the upstream record.
    """
    (x0, x1, y0, y1) = record.door_zone
    (cx0, cx1, cy0, cy1) = record.cab_rect
    dx = (x0 + x1 - cx0 - cx1) / 2.0
    dy = (y0 + y1 - cy0 - cy1) / 2.0
    margin = DOOR_FRONT_SIDE_MARGIN_M
    depth = DOOR_FRONT_DEPTH_M
    if abs(dx) > abs(dy):
        near = x1 if dx > 0.0 else x0
        far = near + (depth if dx > 0.0 else -depth)
        return (min(near, far), max(near, far), y0 - margin, y1 + margin)
    near = y1 if dy > 0.0 else y0
    far = near + (depth if dy > 0.0 else -depth)
    return (x0 - margin, x1 + margin, min(near, far), max(near, far))


class ElevatorMotion:
    """Advance the frozen elevator call, boarding, door, and travel state machine."""

    def __init__(
        self,
        record: ElevatorRecord,
        *,
        boarding_hold_s: Optional[float] = None,
        immediate_calls: bool = False,
    ) -> None:
        self.record = record
        self._immediate_calls = immediate_calls
        self._boarding_hold_s = boarding_hold_s
        self.state = ElevatorPhase.IDLE
        self._current_floor = record.home_floor
        self._door_u = 0.0
        self._last_t_s: float | None = None
        self._open_since_s: float | None = None
        self._exit_clear_since_s: float | None = None
        self._awaiting_exit = False
        self._close_reason: str | None = None
        self._move_start_s: float | None = None
        self._move_from_height = 0.0
        self._move_to_height = 0.0
        self._move_distance_m = 0.0
        self._lit: dict[str, bool] = {fid: False for fid in record.floors}
        self._trigger_since: dict[str, float | None] = {fid: None for fid in record.floors}
        self._called: dict[str, bool] = {fid: False for fid in record.floors}
        self._call_queue: list[str] = []
        self._direction_up = True
        self._pending_destination: str | None = None
        self._move_awaiting_exit = False

    @property
    def cab_floor(self) -> str:
        return self._current_floor

    def _height_of(self, floor_id: str) -> float:
        idx = self.record.floors.index(floor_id)
        return self.record.levels[idx] - self.record.levels[0]

    def _floor_of(self, pose: RobotPose) -> str | None:
        best: str | None = None
        best_dz = None
        for floor_id, level in zip(self.record.floors, self.record.levels):
            dz = pose.z - level
            if -0.3 <= dz <= 1.5 and (best_dz is None or abs(dz) < best_dz):
                best = floor_id
                best_dz = abs(dz)
        return best

    def positions(self) -> dict[str, float]:
        cab_height = self._move_from_height
        if self.state is ElevatorPhase.MOVING and self._last_t_s is not None:
            elapsed = self._last_t_s - (self._move_start_s or self._last_t_s)
            traveled = _elevator_trapezoid_distance(
                elapsed, self._move_distance_m, self.record.speed_mps
            )
            direction = 1.0 if self._move_to_height >= self._move_from_height else -1.0
            cab_height = self._move_from_height + direction * traveled
        elif self.state is not ElevatorPhase.MOVING:
            cab_height = self._height_of(self._current_floor)
        ease_u = _ease(self._door_u)
        result = {
            self.record.cab_part_id: cab_height,
            self.record.cab_door_l_id: -self.record.cab_door_open_m * ease_u,
            self.record.cab_door_r_id: self.record.cab_door_open_m * ease_u,
        }
        active_idx = (
            self.record.floors.index(self._current_floor)
            if self.state is not ElevatorPhase.MOVING
            else None
        )
        for idx, floor_id in enumerate(self.record.floors):
            door_u = ease_u if idx == active_idx else 0.0
            half = self.record.door_open_m[idx]
            result[self.record.door_l_ids[idx]] = -half * door_u
            result[self.record.door_r_ids[idx]] = half * door_u
            result[self.record.button_ids[idx]] = (
                -ELEV_BUTTON_PRESS_M if self._lit[floor_id] else 0.0
            )
        return result

    def lit(self) -> dict[str, bool]:
        return {
            self.record.button_ids[idx]: self._lit[floor_id]
            for (idx, floor_id) in enumerate(self.record.floors)
        }

    def step(self, now_s: float, robot: RobotPose | None) -> list:
        events: list = []
        dt = 0.0 if self._last_t_s is None else max(0.0, now_s - self._last_t_s)
        self._last_t_s = now_s
        events += self._update_calls(now_s, robot)
        if self.state is ElevatorPhase.DOORS_OPENING:
            events += self._step_door_opening(now_s, dt)
        elif self.state is ElevatorPhase.DOORS_OPEN:
            events += self._step_door_open(now_s, robot)
        elif self.state is ElevatorPhase.DOORS_CLOSING:
            events += self._step_door_closing(now_s, dt, robot)
        elif self.state is ElevatorPhase.MOVING:
            events += self._step_moving(now_s)
        return events

    def _event(self, now_s: float, kind: str, reason: str | None) -> PartEvent:
        return PartEvent(
            t_s=now_s,
            part_id=self.record.id,
            kind=kind,
            reason=reason,
            joint_position=0.0,
            robot_xy=None,
        )

    def _update_calls(self, now_s: float, robot: RobotPose | None) -> list:
        events: list = []
        if robot is None:
            return events
        floor_id = self._floor_of(robot)
        for fid in self.record.floors:
            if fid != floor_id:
                self._trigger_since[fid] = None
                self._called[fid] = False
                continue
            idx = self.record.floors.index(fid)
            points = _robot_points(robot)
            in_zone = _any_corner_in(points, self.record.trigger_rects[idx])
            if self._immediate_calls:
                in_zone = in_zone or _any_corner_in(points, _door_front_rect(self.record))
            if not in_zone:
                self._trigger_since[fid] = None
                self._called[fid] = False
                continue
            ready = self._immediate_calls
            if not ready:
                if self._trigger_since[fid] is None:
                    self._trigger_since[fid] = now_s
                ready = now_s - self._trigger_since[fid] >= self.record.trigger_dwell_s
            if ready and (not self._called[fid]):
                self._called[fid] = True
                self._lit[fid] = True
                if self._immediate_calls:
                    print(f"ELEVATOR_TRIGGER_CALL t={now_s:.3f} floor={fid}", flush=True)
                events.append(self._event(now_s, "call", fid))
                events += self._begin_service(now_s, fid)
        return events

    def _begin_service(self, now_s: float, floor_id: str) -> list:
        if self.state is not ElevatorPhase.IDLE:
            if floor_id not in self._call_queue:
                self._call_queue.append(floor_id)
            return []
        if floor_id == self._current_floor:
            return self._open_doors(now_s, awaiting_exit=False)
        return self._start_moving(now_s, floor_id, with_passenger=False)

    def _open_doors(self, now_s: float, awaiting_exit: bool) -> list:
        self.state = ElevatorPhase.DOORS_OPENING
        self._awaiting_exit = awaiting_exit
        self._exit_clear_since_s = None
        return [self._event(now_s, "door_open_start", self._current_floor)]

    def _step_door_opening(self, now_s: float, dt: float) -> list:
        events: list = []
        self._door_u = min(1.0, self._door_u + dt / ELEV_DOOR_MOVE_S)
        if self._door_u >= 1.0 - _EASE_TOL:
            self._door_u = 1.0
            self.state = ElevatorPhase.DOORS_OPEN
            self._open_since_s = now_s
            if self._lit.get(self._current_floor):
                self._lit[self._current_floor] = False
            events.append(self._event(now_s, "door_open_done", self._current_floor))
        return events

    def _step_door_open(self, now_s: float, robot: RobotPose | None) -> list:
        if self._open_since_s is None or robot is None:
            return []
        boarding = self._boarding_hold_s is not None and (not self._awaiting_exit)
        hold_s = self._boarding_hold_s if boarding else self.record.door_open_s
        elapsed_s = now_s - self._open_since_s
        if not boarding and elapsed_s < hold_s:
            return []
        cab_rect = self.record.cab_rect
        door_zone = self.record.door_zone
        corners = _robot_corners(robot)
        all_in = _all_corners_in(corners, cab_rect)
        if boarding:
            all_in = all_in and self._floor_of(robot) == self._current_floor
        any_in_cab = _any_corner_in(corners, cab_rect)
        any_in_zone = _any_corner_in(_robot_points(robot), door_zone)
        if all_in and self._awaiting_exit:
            return []
        if all_in:
            events = self._begin_closing(now_s, "travel")
            events.append(self._event(now_s, "robot_in_cab", self._current_floor))
            return events
        if elapsed_s < hold_s:
            return []
        all_out_clear = not any_in_cab and (not any_in_zone)
        if not all_out_clear:
            return []
        if self._awaiting_exit:
            if self._exit_clear_since_s is None:
                self._exit_clear_since_s = now_s
                return [self._event(now_s, "robot_out_of_cab", self._current_floor)]
            if now_s - self._exit_clear_since_s >= self.record.exit_close_delay_s:
                return self._begin_closing(now_s, "stay")
            return []
        return self._begin_closing(now_s, "stay")

    def _begin_closing(self, now_s: float, reason: str) -> list:
        self.state = ElevatorPhase.DOORS_CLOSING
        self._close_reason = reason
        return [self._event(now_s, "door_close_start", reason)]

    def _step_door_closing(self, now_s: float, dt: float, robot: RobotPose | None) -> list:
        if robot is None:
            return []
        if _any_corner_in(_robot_points(robot), self.record.door_zone):
            self.state = ElevatorPhase.DOORS_OPENING
            return [self._event(now_s, "door_reopen", self._current_floor)]
        events: list = []
        self._door_u = max(0.0, self._door_u - dt / ELEV_DOOR_MOVE_S)
        if self._door_u <= _EASE_TOL:
            self._door_u = 0.0
            events.append(self._event(now_s, "door_close_done", self._close_reason))
            if self._close_reason == "travel":
                destination = self._next_floor(self._current_floor)
                events += self._start_moving(now_s, destination, with_passenger=True)
            else:
                self.state = ElevatorPhase.IDLE
                events += self._drain_queue(now_s)
        return events

    def _next_floor(self, from_floor: str) -> str:
        floors = self.record.floors
        idx = floors.index(from_floor)
        if len(floors) == 2:
            return floors[1 - idx]
        if self._direction_up:
            if idx + 1 < len(floors):
                return floors[idx + 1]
            self._direction_up = False
            return floors[idx - 1]
        if idx - 1 >= 0:
            return floors[idx - 1]
        self._direction_up = True
        return floors[idx + 1]

    def _drain_queue(self, now_s: float) -> list:
        events: list = []
        while self._call_queue:
            fid = self._call_queue.pop(0)
            events += self._begin_service(now_s, fid)
            if self.state is not ElevatorPhase.IDLE:
                break
        return events

    def _start_moving(self, now_s: float, destination: str, with_passenger: bool) -> list:
        self.state = ElevatorPhase.MOVING
        self._move_start_s = now_s
        self._move_from_height = self._height_of(self._current_floor)
        self._move_to_height = self._height_of(destination)
        self._move_distance_m = abs(self._move_to_height - self._move_from_height)
        self._pending_destination = destination
        self._move_awaiting_exit = with_passenger
        return [self._event(now_s, "move_start", destination)]

    def _step_moving(self, now_s: float) -> list:
        if self._move_start_s is None or self._pending_destination is None:
            return []
        elapsed = now_s - self._move_start_s
        total_time = _elevator_trapezoid_time(self._move_distance_m, self.record.speed_mps)
        if elapsed < total_time:
            return []
        self._current_floor = self._pending_destination
        return self._open_doors(now_s, awaiting_exit=self._move_awaiting_exit) + [
            self._event(now_s, "move_done", self._current_floor)
        ]


def _axis_angle_quat(axis: str, angle_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(angle_deg) / 2.0
    (cos_h, sin_h) = (math.cos(half), math.sin(half))
    if axis == "X":
        return (cos_h, sin_h, 0.0, 0.0)
    if axis == "Y":
        return (cos_h, 0.0, sin_h, 0.0)
    return (cos_h, 0.0, 0.0, sin_h)


def joint_transform(
    joint_type: str, axis: str, q: float
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if joint_type == "revolute":
        return ((0.0, 0.0, 0.0), _axis_angle_quat(axis, q))
    translation = {"X": (q, 0.0, 0.0), "Y": (0.0, q, 0.0), "Z": (0.0, 0.0, q)}[axis]
    return (translation, (1.0, 0.0, 0.0, 0.0))


def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    (aw, ax, ay, az) = a
    (bw, bx, by, bz) = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate(
    quat: tuple[float, float, float, float], vec: tuple[float, float, float]
) -> tuple[float, float, float]:
    (w, x, y, z) = quat
    (vx, vy, vz) = vec
    uvx = y * vz - z * vy
    uvy = z * vx - x * vz
    uvz = x * vy - y * vx
    uuvx = y * uvz - z * uvy
    uuvy = z * uvx - x * uvz
    uuvz = x * uvy - y * uvx
    return (vx + 2.0 * (w * uvx + uuvx), vy + 2.0 * (w * uvy + uuvy), vz + 2.0 * (w * uvz + uuvz))


def compose_pose(
    base_pos: tuple[float, float, float],
    base_quat: tuple[float, float, float, float],
    local_pos: tuple[float, float, float],
    local_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rotated_local = _quat_rotate(base_quat, local_pos)
    world_pos = tuple(b + r for (b, r) in zip(base_pos, rotated_local))
    world_quat = _quat_mul(base_quat, local_quat)
    return (world_pos, world_quat)


def part_world_pose(
    closed_pose: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    joint_type: str,
    axis: str,
    q: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    (base_pos, base_quat) = closed_pose
    (local_pos, local_quat) = joint_transform(joint_type, axis, q)
    return compose_pose(base_pos, base_quat, local_pos, local_quat)


def elevator_cab_door_world_pose(
    closed_pose: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    axis: str,
    door_q: float,
    cab_q: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    (pos, quat) = part_world_pose(closed_pose, "prismatic", axis, door_q)
    return ((pos[0], pos[1], pos[2] + cab_q), quat)


def kinematic_target_row(
    position: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]
) -> tuple[float, float, float, float, float, float, float]:
    (x, y, z) = position
    (w, qx, qy, qz) = quat_wxyz
    return (x, y, z, qx, qy, qz, w)


def teleport_pose(
    position: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    return (position, quat_wxyz)


def _should_query_robot(now_s: float, next_decision_t_s: float | None, moving: bool) -> bool:
    return moving or next_decision_t_s is None or now_s >= next_decision_t_s


def _changed_part_ids(
    current: Mapping[str, float], previous: Mapping[str, float], eps: float = _EASE_TOL
) -> list[str]:
    return sorted(pid for (pid, q) in current.items() if abs(q - previous.get(pid, 0.0)) > eps)


def _with_moved_children(changed_ids: list[str], parents: Mapping[str, str | None]) -> list[str]:
    moved = set(changed_ids)
    moved.update(pid for (pid, parent) in parents.items() if parent is not None and parent in moved)
    return sorted(moved)


class PartMotionDriver:
    """Bind frozen moving-part state to Isaac physics and Fabric visual poses."""

    def __init__(
        self,
        scene_name: str,
        part_paths: Sequence[str],
        part_records: Sequence[PartRecord],
        obj_paths: Sequence[str],
        obj_records: Sequence[ObjRecord],
        elevator_records: Mapping[str, ElevatorRecord] | None = None,
    ) -> None:
        self._scene_name = scene_name
        self._part_paths = list(part_paths)
        self._part_records = list(part_records)
        self._part_ids = [r.id for r in part_records]
        self._part_records_by_id = {r.id: r for r in part_records}
        self._obj_paths = list(obj_paths)
        self._obj_records = list(obj_records)
        self._obj_ids = [r.id for r in obj_records]
        self._elevator_records: dict[str, ElevatorRecord] = dict(elevator_records or {})
        self._part_view = None
        self._part_visual_view = None
        self._visual_pending: set = set()
        self._last_visual_sync_s: float | None = None
        self._obj_view = None
        self._closed_pose: dict = {}
        self._motion_set: PartMotionSet | None = None
        self._object_watch: ObjectWatch | None = None
        self._elevator_motions: dict = {}
        self._last_elevator_visual_sync_s: float | None = None
        self._button_lit_state: dict = {}
        self._robot_pose_fn = None
        self._robot_body_size_m = ROBOT_BODY_SIZE_M
        self._last_positions: dict = {}
        self._start_pending = False
        self._next_decision_t_s: float | None = None
        self._last_object_check_s: float | None = None
        self._event_path: Path | None = None
        self._event_fh = None

    @classmethod
    def discover(cls, stage, scene_root: str, scene_name: str) -> PartMotionDriver | None:
        """Read runtime part and elevator attributes beneath the loaded scene root."""
        from pxr import Usd

        root_prim = stage.GetPrimAtPath(scene_root)
        if not root_prim.IsValid():
            return None
        part_marker = f"{USD_PART_ATTR_PREFIX}joint_type"
        obj_marker = f"{USD_OBJ_ATTR_PREFIX}class"
        elev_marker = f"{USD_ELEV_ATTR_PREFIX}id"
        part_paths: list[str] = []
        part_records: list[PartRecord] = []
        obj_paths: list[str] = []
        obj_records: list[ObjRecord] = []
        elevator_records: dict[str, ElevatorRecord] = {}
        for prim in Usd.PrimRange(root_prim):
            attrs = {
                attr.GetName(): attr.Get()
                for attr in prim.GetAttributes()
                if attr.HasAuthoredValue()
            }
            where = str(prim.GetPath())
            if part_marker in attrs:
                part_paths.append(where)
                part_records.append(parse_part_attrs(attrs, where=where))
            elif obj_marker in attrs:
                obj_paths.append(where)
                obj_records.append(parse_obj_attrs(attrs, where=where))
            if elev_marker in attrs:
                record = parse_elevator_attrs(attrs, where=where)
                elevator_records[record.id] = record
        if not part_paths and (not obj_paths):
            return None
        return cls(scene_name, part_paths, part_records, obj_paths, obj_records, elevator_records)

    def set_robot_pose_fn(self, fn, body_size_m: tuple = ROBOT_BODY_SIZE_M) -> None:
        """Provide full robot XY, height, yaw, and body footprint for elevator decisions."""
        self._robot_pose_fn = fn
        self._robot_body_size_m = body_size_m

    def register(self, world) -> None:
        """Register rigid-body views between initial and final world reset."""
        from isaacsim.core.prims import RigidPrim, XFormPrim

        if self._part_paths:
            self._part_view = RigidPrim(
                prim_paths_expr=list(self._part_paths), name=f"{self._scene_name}_parts"
            )
            world.scene.add(self._part_view)
            self._part_visual_view = XFormPrim(
                prim_paths_expr=list(self._part_paths),
                name=f"{self._scene_name}_parts_visual",
                reset_xform_properties=False,
            )
        if self._obj_paths:
            self._obj_view = RigidPrim(
                prim_paths_expr=list(self._obj_paths), name=f"{self._scene_name}_objects"
            )
            world.scene.add(self._obj_view)

    def after_reset(self) -> None:
        """Restore initial part poses and recreate elevator state before settling."""
        if self._part_view is not None:
            (positions, orientations) = self._part_view.get_world_poses()
            self._closed_pose = {
                part_id: (
                    tuple(float(v) for v in positions[i]),
                    tuple(float(v) for v in orientations[i]),
                )
                for (i, part_id) in enumerate(self._part_ids)
            }
        self._motion_set = PartMotionSet([r for r in self._part_records if not r.kind])
        self._next_decision_t_s = None
        self._last_object_check_s = None
        self._visual_pending = set()
        self._last_visual_sync_s = None
        self._last_elevator_visual_sync_s = None
        self._button_lit_state = {}
        self._object_watch = ObjectWatch()
        self._elevator_motions = {
            eid: ElevatorMotion(
                record,
                immediate_calls=self._scene_name == "env3_office_two_floors",
                boarding_hold_s=ENV3_BOARDING_HOLD_S
                if self._scene_name == "env3_office_two_floors"
                else None,
            )
            for (eid, record) in sorted(self._elevator_records.items())
        }
        self._last_positions = self._motion_set.positions()
        if not AUTO_MOTION and self._part_view is not None:
            self._last_positions = start_positions(self._part_records)
        for motion in self._elevator_motions.values():
            self._last_positions.update(motion.positions())
        if self._part_view is not None:
            self._write_physics_poses(list(self._part_ids), self._last_positions)
            self._start_pending = True
        self._sync_visual_poses(list(self._part_ids), self._last_positions)
        if self._event_fh is None:
            self._event_path = self._open_event_log()
        print(
            f"Moving parts: {len(self._part_records)}, objects: {len(self._obj_records)}; "
            f"events: {self._event_path}"
        )
        if not AUTO_MOTION:
            opened = sorted(pid for (pid, q) in self._last_positions.items() if q != 0.0)
            print(f"活动件不自动开关(AUTO_MOTION=False);开局开着的:{opened or '无'},其余关着")

    def step(self, now_s: float, robot_xy_fn) -> None:
        if self._start_pending:
            self._start_pending = False
            self._write_physics_poses(list(self._part_ids), self._last_positions)
            self._sync_visual_poses(list(self._part_ids), self._last_positions)
        if AUTO_MOTION:
            moving = self._motion_set.moving()
            robot_xy = None
            if _should_query_robot(now_s, self._next_decision_t_s, moving):
                robot_xy = robot_xy_fn()
                if self._next_decision_t_s is None or now_s >= self._next_decision_t_s:
                    self._next_decision_t_s = now_s + DECISION_PERIOD_S
            for event in self._motion_set.step(now_s, robot_xy):
                self._write_event(event.to_dict())
            self._write_changed_part_poses(now_s)
        if self._elevator_motions:
            self._step_elevators(now_s)
        if self._obj_view is not None and (
            self._last_object_check_s is None
            or now_s - self._last_object_check_s >= OBJECT_CHECK_PERIOD_S
        ):
            self._last_object_check_s = now_s
            (positions, _orientations) = self._obj_view.get_world_poses()
            obj_positions = {
                obj_id: tuple(float(v) for v in positions[i])
                for (i, obj_id) in enumerate(self._obj_ids)
            }
            for obj_event in self._object_watch.update(now_s, obj_positions):
                self._write_event(obj_event)

    def _part_pose(self, part_id: str, positions: Mapping[str, float]):
        record = self._part_records_by_id[part_id]
        if record.kind == "elevator_door" and record.parent is not None:
            cab_q = positions.get(record.parent, 0.0)
            return elevator_cab_door_world_pose(
                self._closed_pose[part_id], record.axis, positions[part_id], cab_q
            )
        return part_world_pose(
            self._closed_pose[part_id], record.joint_type, record.axis, positions[part_id]
        )

    def _step_elevators(self, now_s: float) -> None:
        """Advance elevators using the full robot pose and synchronize changed parts."""
        robot_pose = None
        if self._robot_pose_fn is not None:
            raw = self._robot_pose_fn()
            if raw is not None:
                (x, y, z, yaw_deg) = raw
                robot_pose = RobotPose(x, y, z, yaw_deg, self._robot_body_size_m)
        merged = dict(self._last_positions)
        changed_ids: list = []
        lit_ids: set = set()
        for motion in self._elevator_motions.values():
            for event in motion.step(now_s, robot_pose):
                self._write_event(event.to_dict())
            elev_positions = motion.positions()
            merged.update(elev_positions)
            lit_ids.update(pid for (pid, lit) in motion.lit().items() if lit)
        changed_ids = _with_moved_children(
            _changed_part_ids(merged, self._last_positions),
            {pid: record.parent for (pid, record) in self._part_records_by_id.items()},
        )
        self._last_positions = merged
        if changed_ids and self._part_view is not None:
            self._write_physics_poses(changed_ids, merged)
            self._visual_pending.update(changed_ids)
        due = (
            self._last_elevator_visual_sync_s is None
            or now_s - self._last_elevator_visual_sync_s >= VISUAL_SYNC_PERIOD_S
        )
        if self._visual_pending and due:
            self._sync_visual_poses(sorted(self._visual_pending), merged)
            self._visual_pending.clear()
            self._last_elevator_visual_sync_s = now_s
        self._sync_button_lights(lit_ids)

    def _sync_button_lights(self, lit_ids: set) -> None:
        for part_id in self._part_records_by_id:
            record = self._part_records_by_id[part_id]
            if record.kind != "call_button":
                continue
            want_lit = part_id in lit_ids
            if self._button_lit_state.get(part_id) == want_lit:
                continue
            self._button_lit_state[part_id] = want_lit
            self._set_button_emissive(part_id, want_lit)

    def _set_button_emissive(self, part_id: str, lit: bool) -> None:
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        idx = self._part_ids.index(part_id)
        prim = stage.GetPrimAtPath(self._part_paths[idx])
        if not prim.IsValid():
            return
        mesh_prim = next(
            (p for p in Usd.PrimRange(prim) if p != prim and p.IsA(UsdGeom.Mesh)), None
        )
        if mesh_prim is None:
            return
        (material, _binding_rel) = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
        if not material:
            return
        shader = next(
            (UsdShade.Shader(p) for p in Usd.PrimRange(material.GetPrim()) if UsdShade.Shader(p)),
            None,
        )
        if shader is None:
            return
        color = Gf.Vec3f(1.0, 0.55, 0.1) if lit else Gf.Vec3f(0.0, 0.0, 0.0)
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(color)

    def _write_changed_part_poses(self, now_s: float) -> None:
        if self._part_view is None:
            return
        positions = self._motion_set.positions()
        changed_ids = _changed_part_ids(positions, self._last_positions)
        self._last_positions = positions
        if changed_ids:
            self._write_physics_poses(changed_ids, positions)
            self._visual_pending.update(changed_ids)
        if not self._visual_pending:
            return
        due = (
            self._last_visual_sync_s is None
            or now_s - self._last_visual_sync_s >= VISUAL_SYNC_PERIOD_S
            or (not changed_ids)
            or (not self._motion_set.moving())
        )
        if due:
            self._sync_visual_poses(sorted(self._visual_pending), positions)
            self._visual_pending.clear()
            self._last_visual_sync_s = now_s

    def positions(self) -> dict:
        return dict(self._last_positions)

    def hold_positions(self, positions: Mapping[str, float]) -> None:
        if self._part_view is None:
            return
        unknown = sorted(set(positions) - set(self._part_ids))
        if unknown:
            raise KeyError(f"没有这些活动件:{unknown};场景里有 {self._part_ids}")
        merged = dict(self._last_positions)
        merged.update({pid: float(q) for (pid, q) in positions.items()})
        changed_ids = sorted(positions)
        self._write_physics_poses(changed_ids, merged)
        self._sync_visual_poses(changed_ids, merged)
        self._last_positions = merged

    def _write_physics_poses(self, changed_ids: list, positions: Mapping[str, float]) -> None:
        import numpy as np

        if POSE_WRITE_MODE == "kinematic_target":
            poses = [self._part_pose(pid, positions) for pid in self._part_ids]
            data = np.array(
                [kinematic_target_row(pos, quat) for (pos, quat) in poses], dtype=np.float32
            )
            indices = np.arange(len(self._part_ids), dtype=np.uint32)
            self._part_view._physics_view.set_kinematic_targets(data, indices)
        else:
            indices = [self._part_ids.index(pid) for pid in changed_ids]
            poses = [self._part_pose(pid, positions) for pid in changed_ids]
            teleported = [teleport_pose(pos, quat) for (pos, quat) in poses]
            positions_arr = np.array([p for (p, _q) in teleported], dtype=np.float32)
            orientations_arr = np.array([q for (_p, q) in teleported], dtype=np.float32)
            self._part_view.set_world_poses(positions_arr, orientations_arr, indices=indices)

    def _sync_visual_poses(self, part_ids: list, positions: Mapping[str, float]) -> None:
        if self._part_visual_view is None or not part_ids:
            return
        import numpy as np

        poses = [teleport_pose(*self._part_pose(pid, positions)) for pid in part_ids]
        self._part_visual_view.set_world_poses(
            np.array([p for (p, _q) in poses], dtype=np.float32),
            np.array([q for (_p, q) in poses], dtype=np.float32),
            indices=[self._part_ids.index(pid) for pid in part_ids],
            usd=VISUAL_SYNC_USD,
        )

    def _open_event_log(self) -> Path:
        import time

        event_dir = PART_EVENT_DIR.expanduser()
        event_dir.mkdir(parents=True, exist_ok=True)
        path = event_dir / f"{self._scene_name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._event_fh = path.open("w", encoding="utf-8")
        return path

    def _write_event(self, event: dict) -> None:
        event = dict(event)
        event["scene"] = self._scene_name
        if self._event_fh is not None:
            self._event_fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._event_fh.flush()

    def close(self) -> None:
        if self._event_fh is not None:
            self._event_fh.close()
            self._event_fh = None
