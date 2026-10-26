"""Keyframe topology and shortest-path planning, implemented without an external planner.

Adapted from litevloc_topo in litevloc_code revision a26caf6: graph.py, search.py,
route.py and planner.py under ``topo/litevloc_topo``. Retain only benchmark needs,
with equivalent algorithms and identical ``kf_path`` / ``length_m`` checked by
``test_vnav_topo.py``. Excluded features are object/image/text goal resolution
(goals here are keyframe IDs), ``advance`` (handled in ``methods/base_topo.py``),
and upstream pose-goal appending / pruning waypoints behind the start, which
only affect waypoints unused by the benchmark.

Read two map files:
- ``poses.txt``: rows ``<image> qw qx qy qz tx ty tz`` describe the task-to-camera
  transform. Invert it to get camera pose in the map frame; zero-based line
  numbers are keyframe IDs.
- ``edges_<edge_type>.txt``: undirected rows ``i j [w]``; default edge lengths
  are Euclidean distances between planar body positions.

Project the camera +z optical axis onto map XY and take atan2 for keyframe yaw.
Body position equals camera position minus
``camera_forward_m`` times (cos yaw, sin yaw).
"""

import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vnav_benchmark.interfaces import Pose2D

MIN_WAYPOINT_SPACING_M = 0.5  # Drop waypoints closer than this to the previous retained point;
# always retain the last
MAX_SEGMENT_M = 10.0  # Insert evenly spaced points across gaps longer than this distance


class TopoGraphError(Exception):
    """Unreadable map, malformed edges, missing referenced keyframes or an unplannable route."""


class NoPathError(TopoGraphError):
    """Start/goal absent from the graph, or no connection between them."""


# ---------------------------------------------------------------- Graph


def _quat_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise TopoGraphError("四元数范数接近 0,不是合法旋转")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def read_camera_poses(map_dir: Path) -> List[np.ndarray]:
    """Read poses.txt as inverted camera-to-map 4x4 transforms; line numbers are keyframe IDs."""
    path = Path(map_dir) / "poses.txt"
    if not path.is_file():
        raise TopoGraphError(f"找不到 poses.txt: {path}")
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise TopoGraphError(f"poses.txt 行格式错误: {line!r}")
        try:
            qw, qx, qy, qz, x, y, z = (float(v) for v in parts[1:])
        except ValueError as exc:
            raise TopoGraphError(f"poses.txt 行数值错误: {line!r}") from exc
        matrix = np.eye(4)
        matrix[:3, :3] = _quat_wxyz_to_matrix(qw, qx, qy, qz)
        matrix[:3, 3] = [x, y, z]
        poses.append(np.linalg.inv(matrix))
    return poses


def camera_to_body(camera_in_map: np.ndarray, camera_forward_m: float) -> Pose2D:
    """Convert a camera-to-map 4x4 transform into a planar body pose."""
    forward = camera_in_map[:3, 2]
    yaw = math.atan2(forward[1], forward[0])
    return Pose2D(
        float(camera_in_map[0, 3] - camera_forward_m * math.cos(yaw)),
        float(camera_in_map[1, 3] - camera_forward_m * math.sin(yaw)),
        float(yaw),
    )


@dataclass(frozen=True)
class TopoGraph:
    """Planar keyframe body poses and adjacency; default weights are Euclidean body distances."""

    poses: Dict[int, Pose2D] = field(default_factory=dict)
    adjacency: Dict[int, Tuple[Tuple[int, float], ...]] = field(default_factory=dict)

    def neighbors(self, kf: int) -> Tuple[Tuple[int, float], ...]:
        return self.adjacency.get(kf, ())

    def nearest(self, x: float, y: float) -> int:
        """Find the nearest keyframe body position to (x, y), keeping the first on ties."""
        if not self.poses:
            raise TopoGraphError("图里没有关键帧")
        return min(
            self.poses, key=lambda kf: math.hypot(self.poses[kf].x - x, self.poses[kf].y - y)
        )

    def components(self) -> List[List[int]]:
        """Return sorted connected components, ordered by their smallest keyframe ID."""
        visited = set()
        comps: List[List[int]] = []
        for kf in self.poses:
            if kf in visited:
                continue
            stack, comp = [kf], []
            visited.add(kf)
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor, _ in self.adjacency.get(node, ()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            comps.append(sorted(comp))
        comps.sort(key=lambda comp: comp[0])
        return comps

    def __len__(self) -> int:
        return len(self.poses)


def _adjacency(
    poses: Dict[int, Pose2D], edges: Iterable[Tuple[int, int, float]]
) -> Dict[int, Tuple[Tuple[int, float], ...]]:
    neighbors: Dict[int, List[Tuple[int, float]]] = {kf: [] for kf in poses}
    for i, j, weight in edges:
        if i not in poses or j not in poses:
            raise TopoGraphError(f"边引用了不存在的关键帧: {i} / {j}")
        neighbors[i].append((j, weight))
        neighbors[j].append((i, weight))
    return {kf: tuple(v) for kf, v in neighbors.items()}


def load_topo_graph(
    map_dir: Path, camera_forward_m: float, edge_type: str = "trav", use_file_weight: bool = False
) -> TopoGraph:
    """Read poses.txt and edges_<edge_type>.txt to compute body poses and graph adjacency."""
    map_dir = Path(map_dir)
    poses = {
        index: camera_to_body(pose, camera_forward_m)
        for index, pose in enumerate(read_camera_poses(map_dir))
    }
    edges_path = map_dir / f"edges_{edge_type}.txt"
    if not edges_path.is_file():
        raise TopoGraphError(f"找不到 edges_{edge_type}.txt: {edges_path}")
    edges: List[Tuple[int, int, float]] = []
    for line in edges_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise TopoGraphError(f"edges_{edge_type}.txt 行格式错误: {line!r}")
        i, j = int(parts[0]), int(parts[1])
        if i not in poses or j not in poses:
            raise TopoGraphError(f"edges_{edge_type}.txt 引用了不存在的关键帧: {i} / {j}")
        if use_file_weight:
            if len(parts) < 3:
                raise TopoGraphError(f"edges_{edge_type}.txt 缺权重列: {line!r}")
            weight = float(parts[2])
        else:
            weight = math.hypot(poses[i].x - poses[j].x, poses[i].y - poses[j].y)
        edges.append((i, j, weight))
    return TopoGraph(poses=poses, adjacency=_adjacency(poses, edges))


def from_edges(poses: Dict[int, Pose2D], edges: Iterable[Tuple[int, int]]) -> TopoGraph:
    """Build a graph from in-memory poses/edges for tests, using Euclidean edge weights."""
    weighted = (
        (i, j, math.hypot(poses[i].x - poses[j].x, poses[i].y - poses[j].y)) for i, j in edges
    )
    return TopoGraph(poses=dict(poses), adjacency=_adjacency(poses, weighted))


# ---------------------------------------------------------------- Shortest paths


def dijkstra(
    graph: TopoGraph, start: int, goal: int, blocked: Collection[Tuple[int, int]] = ()
) -> Tuple[List[int], float]:
    """Return (node path including endpoints, total length), blocking undirected edges in blocked.
    """
    if start not in graph.poses or goal not in graph.poses:
        raise NoPathError(f"关键帧 {start} 或 {goal} 不在图里")
    if start == goal:
        return [start], 0.0
    blocked_set = {tuple(sorted(pair)) for pair in blocked}
    dist = {start: 0.0}
    prev: Dict[int, int] = {}
    visited = set()
    heap: List[Tuple[float, int]] = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for neighbor, length in graph.adjacency.get(node, ()):
            if tuple(sorted((node, neighbor))) in blocked_set:
                continue
            nd = d + length
            if nd < dist.get(neighbor, math.inf):
                dist[neighbor] = nd
                prev[neighbor] = node
                heapq.heappush(heap, (nd, neighbor))
    if goal not in dist:
        raise NoPathError(f"关键帧 {start} 到 {goal} 不连通")
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[goal]


# ---------------------------------------------------------------- Routes


@dataclass(frozen=True)
class Route:
    kf_path: Tuple[int, ...]  # Keyframe sequence including endpoints
    length_m: float  # Total edge length, meters
    waypoints: Tuple[Pose2D, ...]  # Thinned/interpolated waypoints: zero yaw internally,
    # goal-keyframe yaw at the end
    waypoint_kfs: Tuple[Optional[int], ...]  # Associated keyframe per waypoint; None for
    # interpolated points


def _sparsify(
    pairs: Sequence[Tuple[Pose2D, Optional[int]]]
) -> List[Tuple[Pose2D, Optional[int]]]:
    if len(pairs) <= 2:
        return list(pairs)
    kept = [pairs[0]]
    for point, kf in pairs[1:-1]:
        last = kept[-1][0]
        if math.hypot(point.x - last.x, point.y - last.y) >= MIN_WAYPOINT_SPACING_M:
            kept.append((point, kf))
    kept.append(pairs[-1])
    return kept


def _subdivide(
    pairs: Sequence[Tuple[Pose2D, Optional[int]]]
) -> List[Tuple[Pose2D, Optional[int]]]:
    if not pairs:
        return []
    result = [pairs[0]]
    for point, kf in pairs[1:]:
        prev = result[-1][0]
        segments = max(1, math.ceil(math.hypot(point.x - prev.x, point.y - prev.y) / MAX_SEGMENT_M))
        for step in range(1, segments + 1):
            if step == segments:
                result.append((point, kf))
            else:
                ratio = step / segments
                mid = Pose2D(
                    prev.x + (point.x - prev.x) * ratio, prev.y + (point.y - prev.y) * ratio, 0.0
                )
                result.append((mid, None))
    return result


def make_route(graph: TopoGraph, kf_path: Sequence[int], length_m: float) -> Route:
    pairs = [(Pose2D(graph.poses[kf].x, graph.poses[kf].y, 0.0), kf) for kf in kf_path]
    pairs[-1] = (graph.poses[kf_path[-1]], kf_path[-1])
    full = _subdivide(_sparsify(pairs))
    return Route(
        tuple(kf_path), length_m, tuple(p for p, _ in full), tuple(k for _, k in full)
    )


# ---------------------------------------------------------------- Planner


class TopoPlanner:
    """Episode planner state with persistent blocked edges until clear_blocked is called.

    replan reuses the previous goal.
    """

    def __init__(self, graph: TopoGraph) -> None:
        self._graph = graph
        self._blocked: FrozenSet[Tuple[int, int]] = frozenset()
        self._current_route: Optional[Route] = None
        self._goal_kf: Optional[int] = None

    @property
    def blocked(self) -> FrozenSet[Tuple[int, int]]:
        return self._blocked

    @property
    def current_route(self) -> Optional[Route]:
        return self._current_route

    def block_edge(self, i: int, j: int) -> None:
        self._blocked = self._blocked | {(min(i, j), max(i, j))}

    def clear_blocked(self) -> None:
        self._blocked = frozenset()

    def _plan_to(self, start_kf: int, goal_kf: int) -> Route:
        if goal_kf not in self._graph.poses:
            raise TopoGraphError(f"目标关键帧不在地图里: {goal_kf}")
        kf_path, length_m = dijkstra(self._graph, start_kf, goal_kf, blocked=self._blocked)
        self._current_route = make_route(self._graph, kf_path, length_m)
        self._goal_kf = goal_kf
        return self._current_route

    def plan(self, start: Pose2D, goal_kf: int, start_kf: Optional[int] = None) -> Route:
        """Plan from the keyframe nearest start to goal_kf."""
        return self._plan_to(
            self._graph.nearest(start.x, start.y) if start_kf is None else start_kf, goal_kf
        )

    def replan(
        self, from_pose: Pose2D, block_edge: Optional[Tuple[int, int]] = None,
        start_kf: Optional[int] = None,
    ) -> Route:
        """Optionally block an edge, then replan from from_pose to the previous goal keyframe."""
        if block_edge is not None:
            self.block_edge(*block_edge)
        if self._goal_kf is None:
            raise TopoGraphError("还没有 plan 过,不能 replan")
        return self._plan_to(
            self._graph.nearest(from_pose.x, from_pose.y) if start_kf is None else start_kf,
            self._goal_kf,
        )
