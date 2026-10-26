"""Typed USD attributes for frozen scene moving parts and elevators."""

# Adapted from nav_sim_isaac/nav_sim_isaac/scene/parts.py.
# Source SHA256: e38c30a482e19aac0362d56b664484f7546f6f8019a5d8e730687b4e7ee2387f.
# Copyright (c) 2026 Jianhao Jiao. MIT license; see LICENSE.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ROBOT_RADIUS_M = 0.4
HYSTERESIS_MIN_M = 0.4
PART_ATTR_PREFIX = "nav_part_"
OBJ_ATTR_PREFIX = "nav_obj_"
USD_PART_ATTR_PREFIX = "nav:part_"
USD_OBJ_ATTR_PREFIX = "nav:obj_"
_JOINT_TYPES = frozenset({"revolute", "prismatic"})
_AXES = frozenset({"X", "Y", "Z"})
_APPROXIMATIONS = frozenset({"convexHull", "none"})


class PartSpecError(Exception):
    pass


ELEV_ATTR_PREFIX = "nav_elev_"
USD_ELEV_ATTR_PREFIX = "nav:elev_"
ELEV_DOOR_MOVE_S = 1.5
ELEV_BUTTON_PRESS_M = 0.005


@dataclass(frozen=True)
class PartRecord:
    """Runtime joint limits, motion triggers, and parent relation decoded from USD."""

    id: str
    joint_type: str
    parent: str | None
    axis: str
    lower_limit: float
    upper_limit: float
    open_position: float
    approximation: str
    trigger_xy: tuple[float, float]
    open_radius_m: float
    close_radius_m: float
    hold_s: float
    move_s: float
    sweep_xy: tuple[float, float, float, float]
    kind: str = ""
    elevator: str | None = None
    floor: str | None = None


@dataclass(frozen=True)
class ElevatorRecord:
    """Frozen elevator geometry, floors, timings, and moving-part identifiers."""

    id: str
    floors: tuple[str, ...]
    levels: tuple[float, ...]
    home_floor: str
    speed_mps: float
    door_open_s: float
    exit_close_delay_s: float
    trigger_dwell_s: float
    cab_rect: tuple[float, float, float, float]
    door_zone: tuple[float, float, float, float]
    trigger_rects: tuple[tuple[float, float, float, float], ...]
    trigger_directions: tuple[str, ...]
    cab_door_open_m: float
    door_open_m: tuple[float, ...]
    cab_part_id: str
    cab_door_l_id: str
    cab_door_r_id: str
    door_l_ids: tuple[str, ...]
    door_r_ids: tuple[str, ...]
    button_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObjRecord:
    """Runtime identifier and physical metadata for a movable scene object."""

    id: str
    obj_class: str
    mass: float
    inside: str | None


def _strip_prefix(key: str) -> str | None:
    for prefix in (PART_ATTR_PREFIX, USD_PART_ATTR_PREFIX):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


def _strip_obj_prefix(key: str) -> str | None:
    for prefix in (OBJ_ATTR_PREFIX, USD_OBJ_ATTR_PREFIX):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


def _to_float(value: Any, where: str) -> float:
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PartSpecError(f"{where}:取值 {value!r} 不是数字") from exc


def _to_str(value: Any, where: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _to_float_pair(value: Any, where: str, count: int) -> tuple:
    if isinstance(value, str) or not hasattr(value, "__len__"):
        raise PartSpecError(f"{where}:应该是 {count} 个数,收到 {value!r}")
    seq = list(value)
    if len(seq) != count:
        raise PartSpecError(f"{where}:应该是 {count} 个数,收到 {len(seq)} 个")
    return tuple(_to_float(v, where) for v in seq)


def parse_part_attrs(attrs: Mapping[str, Any], where: str) -> PartRecord:
    """Validate USD joint attributes without importing scene construction code."""
    fields: dict = {}
    for key, value in attrs.items():
        name = _strip_prefix(key)
        if name is not None:
            fields[name] = value

    def _require(name: str) -> Any:
        if name not in fields:
            raise PartSpecError(f"{where}:缺属性 {PART_ATTR_PREFIX}{name}")
        return fields[name]

    part_id = _to_str(_require("id"), where)
    joint_type = _to_str(_require("joint_type"), where)
    if joint_type not in _JOINT_TYPES:
        raise PartSpecError(f"{where}:joint_type={joint_type!r} 只能是 {sorted(_JOINT_TYPES)}")
    parent_raw = _to_str(_require("parent"), where)
    parent = parent_raw or None
    axis = _to_str(_require("axis"), where)
    if axis not in _AXES:
        raise PartSpecError(f"{where}:axis={axis!r} 只能是 {sorted(_AXES)}")
    lower = _to_float(_require("lower_limit"), f"{where}.lower_limit")
    upper = _to_float(_require("upper_limit"), f"{where}.upper_limit")
    if lower > upper + 1e-09:
        raise PartSpecError(f"{where}:lower_limit={lower} 比 upper_limit={upper} 还大")
    open_position = _to_float(_require("open_position"), f"{where}.open_position")
    if not lower - 1e-09 <= open_position <= upper + 1e-09:
        raise PartSpecError(f"{where}:open_position={open_position} 不在 [{lower},{upper}] 范围内")
    if open_position == 0.0:
        raise PartSpecError(f"{where}:open_position 不能等于 0(0 表示关着)")
    approximation = _to_str(_require("approximation"), where)
    if approximation not in _APPROXIMATIONS:
        raise PartSpecError(
            f"{where}:approximation={approximation!r} 只能是 {sorted(_APPROXIMATIONS)}"
        )
    trigger_xy = _to_float_pair(_require("trigger_xy"), f"{where}.trigger_xy", 2)
    open_radius_m = _to_float(_require("open_radius_m"), f"{where}.open_radius_m")
    close_radius_m = _to_float(_require("close_radius_m"), f"{where}.close_radius_m")
    if close_radius_m < open_radius_m + HYSTERESIS_MIN_M - 1e-09:
        raise PartSpecError(
            f"{where}: close_radius_m must exceed open_radius_m by {HYSTERESIS_MIN_M} m"
        )
    hold_s = _to_float(_require("hold_s"), f"{where}.hold_s")
    move_s = _to_float(_require("move_s"), f"{where}.move_s")
    if hold_s <= 0.0 or move_s <= 0.0:
        raise PartSpecError(f"{where}:hold_s/move_s 必须大于 0,收到 {hold_s}/{move_s}")
    sweep_xy = _to_float_pair(_require("sweep_xy"), f"{where}.sweep_xy", 4)
    kind = _to_str(fields.get("kind", ""), where)
    elevator_raw = _to_str(fields.get("elevator", ""), where)
    floor_raw = _to_str(fields.get("floor", ""), where)
    return PartRecord(
        id=part_id,
        joint_type=joint_type,
        parent=parent,
        axis=axis,
        lower_limit=lower,
        upper_limit=upper,
        open_position=open_position,
        approximation=approximation,
        trigger_xy=(trigger_xy[0], trigger_xy[1]),
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        hold_s=hold_s,
        move_s=move_s,
        sweep_xy=sweep_xy,
        kind=kind,
        elevator=elevator_raw or None,
        floor=floor_raw or None,
    )


def parse_obj_attrs(attrs: Mapping[str, Any], where: str) -> ObjRecord:
    """Validate movable-object USD attributes."""
    fields: dict = {}
    for key, value in attrs.items():
        name = _strip_obj_prefix(key)
        if name is not None:
            fields[name] = value

    def _require(name: str) -> Any:
        if name not in fields:
            raise PartSpecError(f"{where}:缺属性 {OBJ_ATTR_PREFIX}{name}")
        return fields[name]

    obj_id = _to_str(_require("id"), where)
    obj_class = _to_str(_require("class"), where)
    mass = _to_float(_require("mass"), f"{where}.mass")
    if mass <= 0.0:
        raise PartSpecError(f"{where}:obj_mass 必须大于 0,收到 {mass}")
    inside_raw = _to_str(_require("inside"), where)
    inside = inside_raw or None
    return ObjRecord(id=obj_id, obj_class=obj_class, mass=mass, inside=inside)


def _strip_elev_prefix(key: str) -> str | None:
    for prefix in (ELEV_ATTR_PREFIX, USD_ELEV_ATTR_PREFIX):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


ELEV_STR_LIST_SEP = ","


def _to_str_tuple(value: Any, where: str, count: int | None = None) -> tuple:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        seq = value.split(ELEV_STR_LIST_SEP) if value else []
    elif hasattr(value, "__len__"):
        seq = list(value)
    else:
        raise PartSpecError(f"{where}:应该是一串字符串,收到 {value!r}")
    if count is not None and len(seq) != count:
        raise PartSpecError(f"{where}:应该是 {count} 个字符串,收到 {len(seq)} 个")
    return tuple(_to_str(v, where) for v in seq)


def _to_float_tuple(value: Any, where: str, count: int | None = None) -> tuple:
    if isinstance(value, str) or not hasattr(value, "__len__"):
        raise PartSpecError(f"{where}:应该是一串数字,收到 {value!r}")
    seq = list(value)
    if count is not None and len(seq) != count:
        raise PartSpecError(f"{where}:应该是 {count} 个数,收到 {len(seq)} 个")
    return tuple(_to_float(v, where) for v in seq)


def parse_elevator_attrs(attrs: Mapping[str, Any], where: str) -> ElevatorRecord:
    """Validate frozen elevator USD attributes and served-floor geometry."""
    fields: dict = {}
    for key, value in attrs.items():
        name = _strip_elev_prefix(key)
        if name is not None:
            fields[name] = value

    def _require(name: str) -> Any:
        if name not in fields:
            raise PartSpecError(f"{where}:缺属性 {ELEV_ATTR_PREFIX}{name}")
        return fields[name]

    elev_id = _to_str(_require("id"), where)
    floors = _to_str_tuple(_require("floors"), f"{where}.floors")
    n = len(floors)
    if n < 2:
        raise PartSpecError(f"{where}:floors 至少要 2 层,收到 {n} 个")
    levels = _to_float_tuple(_require("levels"), f"{where}.levels", n)
    home_floor = _to_str(_require("home_floor"), where)
    if home_floor not in floors:
        raise PartSpecError(f"{where}:home_floor={home_floor!r} 不在 floors={floors} 里")
    speed_mps = _to_float(_require("speed_mps"), f"{where}.speed_mps")
    if speed_mps <= 0.0:
        raise PartSpecError(f"{where}:speed_mps 必须大于 0,收到 {speed_mps}")
    door_open_s = _to_float(_require("door_open_s"), f"{where}.door_open_s")
    exit_close_delay_s = _to_float(_require("exit_close_delay_s"), f"{where}.exit_close_delay_s")
    trigger_dwell_s = _to_float(_require("trigger_dwell_s"), f"{where}.trigger_dwell_s")
    if door_open_s <= 0.0 or exit_close_delay_s <= 0.0 or trigger_dwell_s <= 0.0:
        raise PartSpecError(f"{where}:door_open_s/exit_close_delay_s/trigger_dwell_s 必须大于 0")
    cab_rect = _to_float_tuple(_require("cab_rect"), f"{where}.cab_rect", 4)
    door_zone = _to_float_tuple(_require("door_zone"), f"{where}.door_zone", 4)
    trigger_rects = tuple(
        _to_float_tuple(_require(f"trigger_rect_{i}"), f"{where}.trigger_rect_{i}", 4)
        for i in range(n)
    )
    trigger_directions = _to_str_tuple(
        _require("trigger_directions"), f"{where}.trigger_directions", n
    )
    cab_door_open_m = _to_float(_require("cab_door_open_m"), f"{where}.cab_door_open_m")
    if cab_door_open_m <= 0.0:
        raise PartSpecError(f"{where}:cab_door_open_m 必须大于 0,收到 {cab_door_open_m}")
    door_open_m = _to_float_tuple(_require("door_open_m"), f"{where}.door_open_m", n)
    if any(v <= 0.0 for v in door_open_m):
        raise PartSpecError(f"{where}:door_open_m 每一项都必须大于 0,收到 {door_open_m}")
    cab_part_id = _to_str(_require("cab_part_id"), where)
    cab_door_l_id = _to_str(_require("cab_door_l_id"), where)
    cab_door_r_id = _to_str(_require("cab_door_r_id"), where)
    door_l_ids = _to_str_tuple(_require("door_l_ids"), f"{where}.door_l_ids", n)
    door_r_ids = _to_str_tuple(_require("door_r_ids"), f"{where}.door_r_ids", n)
    button_ids = _to_str_tuple(_require("button_ids"), f"{where}.button_ids", n)
    return ElevatorRecord(
        id=elev_id,
        floors=floors,
        levels=levels,
        home_floor=home_floor,
        speed_mps=speed_mps,
        door_open_s=door_open_s,
        exit_close_delay_s=exit_close_delay_s,
        trigger_dwell_s=trigger_dwell_s,
        cab_rect=cab_rect,
        door_zone=door_zone,
        trigger_rects=trigger_rects,
        trigger_directions=trigger_directions,
        cab_door_open_m=cab_door_open_m,
        door_open_m=door_open_m,
        cab_part_id=cab_part_id,
        cab_door_l_id=cab_door_l_id,
        cab_door_r_id=cab_door_r_id,
        door_l_ids=door_l_ids,
        door_r_ids=door_r_ids,
        button_ids=button_ids,
    )
