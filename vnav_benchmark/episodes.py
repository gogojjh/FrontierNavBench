"""Immutable episode records, goal-image loading and dataset fingerprint verification."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from vnav_benchmark import frozen, oracle
from vnav_benchmark.interfaces import EpisodeSpec, Pose2D

SCHEMA = "vnav.episodes/2"
TIERS: Dict[str, Tuple[float, float]] = {
    "near": (3.0, 6.0),
    "mid": (6.0, 12.0),
    "far": (12.0, 25.0),
    "xfar": (25.0, math.inf),
}
TIER_OTHER = "other"  # Routes outside difficulty bands; affects report grouping only
TEST_SET = "env0_test_v3"  # Current official set: five v2 Go2 route maps, recomputed with 0.30 m
# grid inflation
KNOWN_SCENE_PREFIXES = ("env0", "env1", "env3")  # Frozen-set names use short prefixes rather than
# full scene names
DEPRECATED_SETS = {
    "env0_test_v1": "R3 退役重采(用户 10-20 定);目录原样保留不动",
    "env0_test_v2": (
        "真值栅格膨胀改 0.30 m,最短路重算,改用 env0_test_v3(用户 10-20 / 10-21 定);目录原样保留不动"
    ),
}
"""Deprecated frozen sets remain for reference, but campaigns reject them."""
DEV_SET = "env0_dev_v1"
POSE_EDGE_TYPE = "odom"  # Read keyframe poses only; every map has odometry edges
SNAP_MAX_M = oracle.ROUTE_SNAP_MAX_M  # Shared maximum endpoint snap distance.
SEQ_RGB, SEQ_DEPTH = ".color.jpg", ".depth.png"
ROUTE_DEPTH_ENCODING = "u16mm+png"  # Map seq/*.depth.png files use 16-bit millimeter PNG encoding


class EpisodeSetError(ValueError):
    pass




@dataclass(frozen=True)
class EpisodeRecord:
    """All ground truth for one episode, accessible only to the evaluation layer."""

    token: str
    route: str  # Route ID is the mapping-session directory name; runner uses it to select maps
    label: str  # Human label R1 / R2 / etc. from maps.yaml routes; hidden from methods
    start: Pose2D  # World frame; first keyframe
    goal: Pose2D  # World frame; last keyframe
    route_kf_first: int
    route_kf_last: int
    shortest_m: float
    straight_m: float
    tier: str
    same_room: bool
    start_room: Optional[str]
    goal_room: Optional[str]
    waypoints: Tuple[Tuple[float, float], ...]
    snap_m: float = 0.0  # Start/goal snapping distance; positive only inside the inflated exclusion
    # region
    goal_rgb: str = ""  # Filename inside the frozen directory
    goal_depth: str = ""
    goal_depth_median_m: Optional[float] = None
    start_floor: str = ""  # Set only for multistory episodes; legacy JSON keeps defaults for absent
    # fields
    goal_floor: str = ""
    grid_shortest_m: Optional[float] = None  # env1/env3 grid reference length, excluded from
    # PLR/SPL

    @property
    def detour_ratio(self) -> float:
        return self.shortest_m / self.straight_m if self.straight_m > 0 else float("inf")

    def to_json(self) -> Dict[str, Any]:
        out = asdict(self)
        for key in ("start", "goal"):
            pose = getattr(self, key)
            out[key] = {"x": pose.x, "y": pose.y, "yaw": pose.yaw}
        out["waypoints"] = [list(p) for p in self.waypoints]
        # Preserve the legacy single-floor JSON shape when loading and re-encoding old sets.
        if not self.start_floor and not self.goal_floor:
            out.pop("start_floor")
            out.pop("goal_floor")
        if self.grid_shortest_m is None:
            out.pop("grid_shortest_m")
        return out

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> EpisodeRecord:
        kw = dict(data)
        for key in ("start", "goal"):
            p = data[key]
            kw[key] = Pose2D(float(p["x"]), float(p["y"]), float(p["yaw"]))
        kw["waypoints"] = tuple(tuple(p) for p in data["waypoints"])
        return cls(**kw)


def tier_of(length_m: float) -> str:
    for name, (lo, hi) in TIERS.items():
        if lo <= length_m < hi:
            return name
    return TIER_OTHER


def episode_token(set_name: str, route: str) -> str:
    return "ep_" + hashlib.sha256(f"{set_name}|{route}".encode()).hexdigest()[:10]


# ---- Build episodes from route maps ----














# ---- Freeze / load ----


def start_spread(records: Sequence[EpisodeRecord]) -> Dict[str, Any]:
    """Measure the spread between each route's recorded first-keyframe truth pose.

    Preserve distinct starts and record their differences for frozen-set review.
    """
    starts = {r.route: {"x": r.start.x, "y": r.start.y, "yaw": r.start.yaw} for r in records}
    dist, yaw = 0.0, 0.0
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            dist = max(dist, math.hypot(a.start.x - b.start.x, a.start.y - b.start.y))
            yaw = max(
                yaw, abs(math.degrees(math.remainder(a.start.yaw - b.start.yaw, 2 * math.pi)))
            )
    return {"starts": starts, "max_dist_m": dist, "max_yaw_deg": yaw}


def episode_set_json(
    set_name: str,
    records: Sequence[EpisodeRecord],
    grid: Union[oracle.TruthGrid, oracle.MultiFloorGrid],
    depth_meta: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "name": set_name,
        "scene": grid.scene,
        "grid": (
            {
                "scene": grid.scene,
                "floors": {f: oracle.grid_digest(g) for f, g in grid.floors.items()},
                "transitions": [asdict(t) for t in grid.transitions],
            }
            if isinstance(grid, oracle.MultiFloorGrid)
            else oracle.grid_digest(grid)
        ),
        "depth": depth_meta,  # {"encoding", "width", "height", "K"}: goal-depth decoding metadata
        "params": {
            # JSON null denotes an unbounded tier; strict frozen JSON rejects Infinity.
            "tiers": {name: (lo, hi if math.isfinite(hi) else None)
                      for name, (lo, hi) in TIERS.items()},
            "snap_max_m": SNAP_MAX_M,
            "source": "route_maps",
        },
        "episodes": [r.to_json() for r in records],
        "start_spread": start_spread(records),
        **(extra or {}),
    }


def freeze_episode_set(target_dir: Path, payload: Dict[str, Any], assets: Dict[str, bytes]) -> str:
    """Create an exclusive frozen directory; require all goal images in assets. Return MANIFEST
    SHA256.
    """
    wanted = {e[k] for e in payload["episodes"] for k in ("goal_rgb", "goal_depth")}
    if "" in wanted or wanted - set(assets):
        raise EpisodeSetError("有回合缺目标照片,不能冻结")
    files = {"episodes.json": frozen.canonical_bytes(payload)}
    files.update({name: assets[name] for name in sorted(wanted)})
    meta = {"name": payload["name"], "episodes": len(payload["episodes"])}
    return frozen.freeze_dir(Path(target_dir), files, meta)


@dataclass
class EpisodeSet:
    name: str
    root: Path
    payload: Dict[str, Any]
    records: Dict[str, EpisodeRecord]
    manifest_sha256: str

    @property
    def is_test(self) -> bool:
        return any(self.name.startswith(f"{p}_test") for p in KNOWN_SCENE_PREFIXES)

    def tokens(self) -> List[str]:
        return [e["token"] for e in self.payload["episodes"]]

    def spec(self, token: str, repeat_index: int, budget_s: float) -> EpisodeSpec:
        """Expose goal images and a budget to methods, without any poses."""
        from vnav_benchmark.sim import codecs as nav_codecs

        rec = self.records[token]
        rgb_jpeg = (self.root / rec.goal_rgb).read_bytes()
        bgr = cv2.imdecode(np.frombuffer(rgb_jpeg, np.uint8), cv2.IMREAD_COLOR)
        d = self.payload["depth"]
        depth_mm = nav_codecs.decode_depth(
            (self.root / rec.goal_depth).read_bytes(), d["width"], d["height"], d["encoding"]
        )
        return EpisodeSpec(
            episode_token=token,
            repeat_index=repeat_index,
            scene=self.payload["scene"],
            goal_rgb=np.ascontiguousarray(bgr[:, :, ::-1]),
            goal_rgb_jpeg=rgb_jpeg,
            goal_depth_m=nav_codecs.millimeters_to_meters(depth_mm).astype(np.float32),
            goal_K=tuple(float(v) for v in d["K"]),
            budget_s=float(budget_s),
        )

    def check_usable(self) -> None:
        """Reject deprecated sets for campaigns; see DEPRECATED_SETS."""
        if self.name in DEPRECATED_SETS:
            raise EpisodeSetError(f"{self.name} 已作废:{DEPRECATED_SETS[self.name]}")

    def check_method_allowed(self, uses_truth: bool) -> None:
        """Reject truth-accessing debugging methods for official episode sets."""
        if self.is_test and uses_truth:
            raise EpisodeSetError(f"{self.name} 是正式集,不许跑调试方法")


def load_episode_set(root: Path) -> EpisodeSet:
    """Verify frozen fingerprints before loading."""
    root = Path(root)
    manifest = frozen.verify_dir(root)
    payload = json.loads((root / "episodes.json").read_bytes())
    if payload.get("schema") != SCHEMA:
        raise EpisodeSetError(f"{root}: schema 不对")
    records = {e["token"]: EpisodeRecord.from_json(e) for e in payload["episodes"]}
    return EpisodeSet(payload["name"], root, payload, records, manifest["sha256"])


def summary_rows(records: Iterable[EpisodeRecord]) -> List[Dict[str, Any]]:
    """Return a review table of token, route, band, l, straight distance, detour ratio and rooms."""
    return [
        {
            "label": r.label,
            "token": r.token,
            "route": r.route,
            "tier": r.tier,
            "shortest_m": round(r.shortest_m, 2),
            "straight_m": round(r.straight_m, 2),
            "detour": round(r.detour_ratio, 2),
            "same_room": r.same_room,
            "start_room": r.start_room,
            "goal_room": r.goal_room,
            "snap_m": round(r.snap_m, 2),
        }
        for r in records
    ]
