"""Copy allowlisted keyframe-map files into sanitized navigation views.

Raw mapping sessions contain poses_abs_gt.txt, gps_data.txt and truth JSON;
map_meta.json also exposes session paths that lead to truth. Methods receive only
SanitizedMap (interfaces.MapHandle), backed by an allowlisted directory.

Copy rather than symlink: upstream maps can change in place, and symlinks reveal
the original session. Content fingerprints pin the copied files independently of
modification times. Read-only views contain:
- seq/NNNNNN.color.jpg and depth.png (uint16 depth divided by 1000 gives meters).
- poses.txt, intrinsics.txt, timestamps.txt, database_descriptors.txt.
- edges_{odom,covis,trav}.txt and edges_trav.chain.txt when present; planning uses
  traversal edges only.
- view_meta.json with path-free keyframe counts, intrinsics, mapping thresholds
  and the opennavmap version extracted from map_meta.json.

view_sha256 hashes the sorted relative paths and content hashes of every file,
including view_meta.json, so identical maps always produce identical fingerprints.
"""

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from vnav_benchmark import oracle
from vnav_benchmark.interfaces import Pose2D
from vnav_benchmark.paths import expand_data_path
from vnav_benchmark.topo import TopoGraph, load_topo_graph, read_camera_poses

VIEW_SCHEMA = "vnav.mapview/1"
VIEW_META_NAME = "view_meta.json"
# Copy only allowlisted top-level files, edges and seq RGB/depth images.
TOP_LEVEL_FILES = ("poses.txt", "intrinsics.txt", "timestamps.txt", "database_descriptors.txt")
# Include odometry, covisibility, traversal shortcuts and chain backups.
# Exclude room/object edges: they connect names rather than keyframes.
EDGE_FILES = ("edges_odom.txt", "edges_covis.txt", "edges_trav.txt", "edges_trav.chain.txt")
SEQ_SUFFIXES = (".color.jpg", ".depth.png")
# Files forbidden in views; checked by tests and verify_view.
FORBIDDEN_NAMES = ("poses_abs_gt.txt", "gps_data.txt", "map_meta.json", "sim_truth.json")
# Copy mapping-threshold overrides only; omit service addresses and peers.
META_OVERRIDE_PREFIX = "keyframe."
DEPTH_SCALE_M = 1000.0  # Divide seq/*.depth.png pixels by 1000 to obtain meters.
READ_ONLY_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
READ_ONLY_DIR_MODE = READ_ONLY_MODE | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPS_CONFIG_PATH = REPO_ROOT / "configs" / "maps.yaml"
MAPS_SCHEMA = "vnav.maps/1"
DEFAULT_SCENE = "env0_office_single_floor"  # Older registry entries without scene belong to env0.
RIGID_TOL = 1e-6  # Orthogonality tolerance for the map_to_world rotation.


class MapViewError(Exception):
    """Missing/invalid map files, tampered fingerprints or malformed maps.yaml."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root: Path) -> Dict[str, str]:
    """Return relative POSIX paths and content SHA256 values for all root files."""
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(root).as_posix()] = _sha256_file(path)
    return manifest


def manifest_sha256(manifest: Mapping[str, str]) -> str:
    """Hash sorted path-tab-hash-newline entries to fingerprint a view."""
    text = "".join(f"{rel}\t{manifest[rel]}\n" for rel in sorted(manifest))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _keyframe_images(src: Path) -> List[str]:
    """Return RGB paths in poses.txt order, checking that each image exists."""
    poses_path = src / "poses.txt"
    if not poses_path.is_file():
        raise MapViewError(f"源图缺 poses.txt: {src}")
    names = []
    for line in poses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name = line.split()[0]
        if not (src / name).is_file():
            raise MapViewError(f"poses.txt 引用的图不存在: {name}")
        names.append(name)
    return names


def _view_meta(src: Path, num_keyframes: int, edge_files: List[str]) -> Dict[str, object]:
    meta_path = src / "map_meta.json"
    source_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    overrides = [
        item for item in source_meta.get("overrides", []) if item.startswith(META_OVERRIDE_PREFIX)
    ]
    return {
        "schema": VIEW_SCHEMA,
        "num_keyframes": num_keyframes,
        "edge_files": edge_files,
        "intrinsics": source_meta.get("intrinsics"),
        "keyframe_overrides": overrides,
        "map_schema_version": source_meta.get("schema_version"),
        "opennavmap_git_sha": source_meta.get("opennavmap_git_sha"),
    }


def _make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        os.chmod(path, READ_ONLY_DIR_MODE if path.is_dir() else READ_ONLY_MODE)
    os.chmod(root, READ_ONLY_DIR_MODE)


def build_sanitized_view(src: Path, dst: Path) -> str:
    """Copy src into a new sanitized dst and return its fingerprint.

    Build in a sibling temporary directory, make it read-only, then rename it
    to avoid leaving an incomplete view after failure.

    Raises:
        MapViewError: dst exists, or poses/edges/keyframe images are missing.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise MapViewError(f"视图目录已存在,拒绝覆盖: {dst}")
    images = _keyframe_images(src)
    edge_files = [name for name in EDGE_FILES if (src / name).is_file()]
    if "edges_trav.txt" not in edge_files:
        raise MapViewError(f"源图没有 edges_trav.txt: {src}")
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "seq").mkdir(parents=True)
    for name in TOP_LEVEL_FILES:
        if (src / name).is_file():
            shutil.copyfile(src / name, tmp / name)
    for name in edge_files:
        shutil.copyfile(src / name, tmp / name)
    for color_name in images:
        stem = color_name[: -len(SEQ_SUFFIXES[0])]
        for suffix in SEQ_SUFFIXES:
            if (src / (stem + suffix)).is_file():
                shutil.copyfile(src / (stem + suffix), tmp / (stem + suffix))
    meta = _view_meta(src, len(images), edge_files)
    (tmp / VIEW_META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    sha = manifest_sha256(file_manifest(tmp))
    _make_read_only(tmp)
    tmp.rename(dst)
    return sha


def verify_view(view_dir: Path, expected_sha256: Optional[str] = None) -> str:
    """Recompute a view fingerprint and optionally match expected_sha256.

    Raises:
        MapViewError: Forbidden files are present or the fingerprint differs.
    """
    view_dir = Path(view_dir)
    for name in FORBIDDEN_NAMES:
        if any(view_dir.rglob(name)):
            raise MapViewError(f"视图里出现了禁止文件 {name}: {view_dir}")
    sha = manifest_sha256(file_manifest(view_dir))
    if expected_sha256 is not None and sha != expected_sha256:
        raise MapViewError(f"视图指纹不符: {view_dir} 实际 {sha} ≠ 期望 {expected_sha256}")
    return sha


class SanitizedMap:
    """Read-only sanitized map implementing interfaces.MapHandle."""

    def __init__(
        self,
        view_dir: Path,
        map_id: str,
        variant: str,
        edge_type: str,
        camera_forward_m: float,
        expected_sha256: Optional[str] = None,
        *,
        scene: Optional[str] = None,
        map_to_world: Optional[Sequence[float]] = None,
    ) -> None:
        self._dir = Path(view_dir)
        self.map_id = map_id
        self.variant = variant
        self.edge_type = edge_type
        self.camera_forward_m = float(camera_forward_m)
        self.sha256 = verify_view(self._dir, expected_sha256)
        if not (self._dir / f"edges_{edge_type}.txt").is_file():
            raise MapViewError(f"视图里没有 edges_{edge_type}.txt: {self._dir}")
        self._graph: Optional[TopoGraph] = None
        self._scene = scene
        self._map_to_world = (
            np.asarray(map_to_world, dtype=float).reshape(4, 4)
            if map_to_world is not None else np.eye(4)
        )
        self._kf_floors: Optional[Dict[int, Optional[str]]] = None

    def floor_at_z(self, z: float) -> Optional[str]:
        """Resolve a world/odometry height through the scene's existing floor rule."""
        return oracle.floor_of_z(self._scene, z) if self._scene is not None else None

    def keyframe_floor(self, kf_id: int) -> Optional[str]:
        """Read map camera heights and transform them to world before classifying."""
        if self._scene is None:
            return None
        if self._kf_floors is None:
            self._kf_floors = {
                i: self.floor_at_z(float((self._map_to_world @ pose)[2, 3]))
                for i, pose in enumerate(read_camera_poses(self._dir))
            }
        return self._kf_floors[kf_id]

    def topo_graph(self) -> TopoGraph:
        """Build the keyframe-pose graph once using edges selected by edge_type."""
        if self._graph is None:
            self._graph = load_topo_graph(self._dir, self.camera_forward_m, self.edge_type)
        return self._graph

    @property
    def num_keyframes(self) -> int:
        return len(self.topo_graph())

    def keyframe_pose(self, kf_id: int) -> Pose2D:
        """Return the keyframe body pose in map coordinates, in meters/radians."""
        pose = self.topo_graph().poses[kf_id]
        return Pose2D(pose.x, pose.y, pose.yaw)

    def _seq_path(self, kf_id: int, suffix: str) -> Path:
        if kf_id not in self.topo_graph().poses:
            raise KeyError(f"地图里没有关键帧 {kf_id}")
        return self._dir / "seq" / f"{kf_id:06d}{suffix}"

    def keyframe_rgb_jpeg(self, kf_id: int) -> bytes:
        """Return the original JPEG bytes of a keyframe RGB image."""
        return self._seq_path(kf_id, SEQ_SUFFIXES[0]).read_bytes()

    def keyframe_depth_m(self, kf_id: int) -> np.ndarray:
        """Return float32 depth in meters, with zero marking invalid pixels."""
        path = self._seq_path(kf_id, SEQ_SUFFIXES[1])
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise MapViewError(f"读不出深度图: {path}")
        return raw.astype(np.float32) / DEPTH_SCALE_M


@dataclass(frozen=True)
class MapEntry:
    """One map entry from configs/maps.yaml."""

    map_id: str
    variant: str  # dense | sparse
    role: str  # formal: evaluated; dev_placeholder: excluded from formal tables.
    source: Path
    edge_type: str
    keyframe_trans_m: float
    keyframe_rot_deg: float
    map_to_world: np.ndarray  # 4x4 map-to-world transform used by OracleLocalizer.
    view_sha256: Optional[str]
    num_keyframes: Optional[int]
    route: Optional[str] = None  # Session name; absent for placeholders.
    trav_src_sha256: Optional[str] = None  # Source edges_trav.txt SHA; copied unchanged.
    scene: str = DEFAULT_SCENE  # Routes/maps can specify scene; omitted values default to env0.


EVAL_ROLES = ("formal", "optional")  # formal: evaluated; optional: mapped/archived but excluded.


@dataclass(frozen=True)
class RouteInfo:
    """One recorded route from maps.yaml routes, finalized on October 20."""

    label: str  # R1, R2, ... sorted by ground-truth shortest-path distance.
    session: str  # Session directory name, matching MapEntry.route.
    session_dir: Path  # Session directory with sensors/ and generated mapping outputs.
    end_room: str
    shortest_m: float  # Ground-truth shortest-path distance in meters.
    eval: str  # formal / optional
    scene: str = DEFAULT_SCENE


@dataclass(frozen=True)
class MapsConfig:
    camera_forward_m: float
    maps: Dict[str, MapEntry]
    defaults: Dict[str, str]  # Map variant to default map_id.
    route_table: Dict[str, RouteInfo] = field(default_factory=dict)  # Route label to routes entry.

    def route_info(self, route: str) -> RouteInfo:
        """Look up a route by label (R1) or session directory name."""
        if route in self.route_table:
            return self.route_table[route]
        hits = [r for r in self.route_table.values() if r.session == route]
        if len(hits) != 1:
            raise MapViewError(f"maps.yaml routes 里没有路线 {route!r}")
        return hits[0]

    def eval_routes(self, scene: str = DEFAULT_SCENE) -> List[RouteInfo]:
        """Return evaluated routes for one scene, sorted by label; default to env0."""
        return [
            r
            for _, r in sorted(self.route_table.items())
            if r.eval == "formal" and r.scene == scene
        ]

    def for_variant(self, variant: str) -> MapEntry:
        """Return development defaults; use for_route for formal route maps."""
        return self.maps[self.defaults[variant]]

    def for_route(self, route: str, variant: str) -> MapEntry:
        """Return a recorded route's dense or sparse map."""
        if route in self.route_table:  # Also accept route labels such as R1.
            route = self.route_table[route].session
        hits = [e for e in self.maps.values() if e.route == route and e.variant == variant]
        if len(hits) != 1:
            raise MapViewError(f"路线 {route!r} 的 {variant} 图有 {len(hits)} 张,应正好 1 张")
        return hits[0]

    def routes(self, scene: str = DEFAULT_SCENE) -> List[str]:
        """Return session names of routes with existing map entries."""
        return sorted({e.route for e in self.maps.values() if e.route and e.scene == scene})


def _rigid_matrix(values: List[float], map_id: str) -> np.ndarray:
    """Validate sixteen row-major values as a rigid 4x4 transform."""
    matrix = np.array(values, dtype=float).reshape(4, 4)
    rot = matrix[:3, :3]
    if not (
        np.allclose(rot @ rot.T, np.eye(3), atol=RIGID_TOL)
        and np.allclose(matrix[3], [0, 0, 0, 1], atol=RIGID_TOL)
    ):
        raise MapViewError(f"{map_id}: map_to_world 不是刚体变换")
    return matrix


def load_maps_config(path: Path = MAPS_CONFIG_PATH) -> MapsConfig:
    """Load configs/maps.yaml with optional per-route and per-map scene fields.

    Missing scene values default to env0_office_single_floor. Entries can select
    env1_garden_outdoor or env3_office_two_floors. Registry keys and map_id values
    remain globally unique; new scenes can use keys such as env1_R1 with matching
    display labels. Avoid a top-level scene default that could reclassify older
    entries accidentally.

    Raises:
        MapViewError: Invalid schema, variant, or missing default map.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != MAPS_SCHEMA:
        raise MapViewError(f"maps.yaml schema 应为 {MAPS_SCHEMA}: {data.get('schema')!r}")
    maps = {}
    for map_id, item in data["maps"].items():
        if item["variant"] not in ("dense", "sparse"):
            raise MapViewError(f"{map_id}: variant 只能是 dense / sparse")
        maps[map_id] = MapEntry(
            map_id=map_id,
            variant=item["variant"],
            role=item["role"],
            source=expand_data_path(item["source"]),
            edge_type=item["edge_type"],
            keyframe_trans_m=float(item["keyframe_thresh"]["trans_m"]),
            keyframe_rot_deg=float(item["keyframe_thresh"]["rot_deg"]),
            map_to_world=_rigid_matrix(item["map_to_world"], map_id),
            view_sha256=item.get("view_sha256"),
            num_keyframes=item.get("num_keyframes"),
            route=item.get("route"),
            trav_src_sha256=item.get("trav_src_sha256"),
            scene=item.get("scene", DEFAULT_SCENE),
        )
    defaults = dict(data["defaults"])
    for variant, map_id in defaults.items():
        if map_id not in maps or maps[map_id].variant != variant:
            raise MapViewError(f"默认 {variant} 图 {map_id!r} 不存在或密度不符")
    routes = {}
    for label, item in (data.get("routes") or {}).items():
        if item["eval"] not in EVAL_ROLES:
            raise MapViewError(f"路线 {label}: eval 只能是 {EVAL_ROLES}")
        session_dir = expand_data_path(item["session_dir"])
        routes[label] = RouteInfo(label, session_dir.name, session_dir, str(item["end_room"]),
                                  float(item["shortest_m"]), item["eval"],
                                  item.get("scene", DEFAULT_SCENE))  # fmt: skip
    return MapsConfig(float(data["camera_forward_m"]), maps, defaults, routes)


def materialize(entry: MapEntry, views_root: Path, camera_forward_m: float) -> SanitizedMap:
    """Build or reuse views_root/<map_id>/, verifying maps.yaml fingerprints."""
    view_dir = Path(views_root) / entry.map_id
    if not view_dir.exists():
        if (entry.source / VIEW_META_NAME).is_file():
            # Published views retain their original allowlisted metadata and fingerprint.
            verify_view(entry.source, entry.view_sha256)
            view_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".map-", dir=str(view_dir.parent)) as temp:
                staged = Path(temp) / "view"
                shutil.copytree(entry.source, staged)
                verify_view(staged, entry.view_sha256)
                # Moving across parents updates '..'; a read-only view needs temporary write access.
                os.chmod(staged, READ_ONLY_DIR_MODE | stat.S_IWUSR)
                os.rename(staged, view_dir)
                os.chmod(view_dir, READ_ONLY_DIR_MODE)
        else:
            build_sanitized_view(entry.source, view_dir)
    return SanitizedMap(
        view_dir,
        entry.map_id,
        entry.variant,
        entry.edge_type,
        camera_forward_m,
        expected_sha256=entry.view_sha256,
        scene=entry.scene,
        map_to_world=entry.map_to_world.ravel(),
    )


@lru_cache(maxsize=None)
def graph_stats(view_dir: str, camera_forward_m: float, edge_type: str) -> Tuple[int, int, int]:
    """Return keyframe, undirected-edge and connected-component counts."""
    graph = load_topo_graph(Path(view_dir), camera_forward_m, edge_type)
    edges = sum(len(neighbors) for neighbors in graph.adjacency.values()) // 2
    return len(graph), edges, len(graph.components())
