"""Ground-truth grids and shortest paths, hidden from methods (frozen after T-I0).

The evaluation layer follows plan sections 3 and 6.7:
- Read frozen scene geometry and grids exported from the checked source layouts.
  Cells are 0.1 m, inflated by a 0.30 m body radius, retained when returning to Go2.
  env0 has 220x220 cells with 19,712 free cells. Scene XY equals simulation world XY.
- Compute shortest-path l with eight-neighbor Dijkstra without diagonal corner
  cutting, then straighten visible segments over free cells to remove the grid's
  roughly 8% overestimate. This is independent of every method's map.
- ``wall_line_of_sight`` checks only walls, treating door openings as clear, to
  test whether the goal is invisible from the start. ``free_line_of_sight`` uses
  inflated free cells for path straightening.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from frontier_nav_bench.scenes import scene_data, truth_arrays

SCENE_ENV0 = "env0_office_single_floor"
SCENE_ENV1 = "env1_garden_outdoor"  # Outdoor buildings share one XY navigation plane
SCENE_ENV3 = "env3_office_two_floors"  # First scene requiring explicit connections between floors
# Recorded routes are not theoretically optimal; use the approved 5% env1/env3 allowance.
MANUAL_ROUTE_SHORTEST_FACTOR = 0.95
STAIR_APPROACH_M = 0.4  # Matches the upstream reachability stair-entrance checkpoints
STAIR_WAYPOINT_SPACING_M = 1.0  # Maximum XY checkpoint spacing through flights and landings.
CELL_M = 0.1  # Grid cell edge length
ROUTE_SNAP_MAX_M = 0.5  # Maximum endpoint snap for route truth, shared with episodes.
ROBOT_RADIUS_M = 0.30  # Body inflation radius: Ranger half-width 0.25 m plus 0.05 m, retained for
# Go2
LOS_STEP_FRAC = 0.25  # Line-of-sight sampling step = cell edge length * 0.25
SQRT2 = math.sqrt(2.0)

Cell = Tuple[int, int]
XY = Tuple[float, float]


@dataclass(frozen=True)
class Room:
    id: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]

    def contains(self, x: float, y: float) -> bool:
        return self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]


@dataclass(frozen=True)
class WallRect:
    x0: float
    x1: float
    y0: float
    y1: float


@dataclass
class TruthGrid:
    """free[i, j] indicates whether the robot fits at ((i+0.5)*cell+x0, (j+0.5)*cell+y0)."""

    scene: str
    x0: float
    y0: float
    cell_m: float
    free: np.ndarray  # bool (nx, ny), inflated by body radius
    raw_blocked: np.ndarray  # bool (nx, ny), before inflation: walls, tall furniture and room
    # exteriors
    rooms: Tuple[Room, ...]
    walls: Tuple[WallRect, ...]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.free.shape  # type: ignore[return-value]

    def cell_of(self, x: float, y: float) -> Cell:
        return (
            int(math.floor((x - self.x0) / self.cell_m)),
            int(math.floor((y - self.y0) / self.cell_m)),
        )

    def center_of(self, cell: Cell) -> XY:
        return (self.x0 + (cell[0] + 0.5) * self.cell_m, self.y0 + (cell[1] + 0.5) * self.cell_m)

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.free.shape[0] and 0 <= cell[1] < self.free.shape[1]

    def is_free_xy(self, x: float, y: float) -> bool:
        cell = self.cell_of(x, y)
        return self.in_bounds(cell) and bool(self.free[cell])

    def room_of(self, x: float, y: float) -> Optional[str]:
        for room in self.rooms:
            if room.contains(x, y):
                return room.id
        return None

    def free_count(self) -> int:
        return int(self.free.sum())


def build_truth_grid(
    scene: str = SCENE_ENV0, cell_m: float = CELL_M, robot_radius_m: float = ROBOT_RADIUS_M
) -> TruthGrid:
    """Read the original single-plane truth, with optional geometry rerasterization."""
    data = scene_data(scene)
    floor = next(iter(data["floors"]))
    return _truth_grid_from_data(scene, floor, cell_m, robot_radius_m, single=True)


def _truth_grid_from_data(
    scene: str, floor: str, cell_m: float, robot_radius_m: float, single: bool = False
) -> TruthGrid:
    """Use exported arrays at frozen settings and the same geometry otherwise."""
    if not math.isfinite(cell_m) or cell_m <= 0:
        raise ValueError("cell_m must be finite and positive")
    if not math.isfinite(robot_radius_m) or robot_radius_m < 0:
        raise ValueError("robot_radius_m must be finite and nonnegative")
    data = scene_data(scene)
    record = data["floors"][floor]
    separate = single and "single_floor" in data
    if cell_m == data["cell_m"] and robot_radius_m == data["robot_radius_m"]:
        free, raw = truth_arrays(scene, "single" if separate else floor)
    elif data["outdoor"]:
        free, raw = _site_masks(data, cell_m, robot_radius_m)
    else:
        operations = data["single_floor"] if separate else record
        free, raw = _floor_masks(
            data["site_size_m"], operations["rect_operations"], cell_m, robot_radius_m
        )
    return TruthGrid(
        scene, record["x0"], record["y0"], cell_m, free, raw,
        tuple(Room(r["id"], r["x_range"], r["y_range"]) for r in record["rooms"]),
        tuple(WallRect(**dict(w)) for w in record["walls"]),
    )


def _dilate(blocked: np.ndarray, offsets: Sequence[Cell]) -> np.ndarray:
    """Inflate obstacles without treating the grid exterior as an obstacle."""
    nx, ny = blocked.shape
    dilated = blocked.copy()
    for di, dj in offsets:
        if abs(di) >= nx or abs(dj) >= ny:
            continue
        src = (slice(max(0, -di), min(nx, nx - di)), slice(max(0, -dj), min(ny, ny - dj)))
        dst = (slice(max(0, di), min(nx, nx + di)), slice(max(0, dj), min(ny, ny + dj)))
        dilated[dst] |= blocked[src]
    return dilated


def _floor_masks(
    size_m: Sequence[float], operations: Sequence, cell_m: float, radius_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay ordered room/wall/object/opening rectangles from upstream walk_grid.

    Keep index clipping and center comparisons in the original order: a floating
    boundary can otherwise change a cell at non-default resolutions.
    """
    tolerance = 1e-6
    nx, ny = [max(1, int(math.ceil(size / cell_m - tolerance))) for size in size_m]
    raw = np.ones((nx, ny), dtype=bool)
    for (x0, x1, y0, y1), value in operations:
        i0, i1 = max(0, int(math.floor(x0 / cell_m))), min(nx, int(math.ceil(x1 / cell_m)))
        j0, j1 = max(0, int(math.floor(y0 / cell_m))), min(ny, int(math.ceil(y1 / cell_m)))
        for i in range(i0, i1):
            cx = (i + 0.5) * cell_m
            if cx < x0 - tolerance or cx > x1 + tolerance:
                continue
            for j in range(j0, j1):
                cy = (j + 0.5) * cell_m
                if y0 - tolerance <= cy <= y1 + tolerance:
                    raw[i, j] = value
    k = max(0, int(math.ceil(radius_m / cell_m - tolerance)))
    radius_cells_sq = (radius_m / cell_m) ** 2 + tolerance
    offsets = [(di, dj) for di in range(-k, k + 1) for dj in range(-k, k + 1)
               if di * di + dj * dj <= radius_cells_sq]
    return ~_dilate(raw, offsets), raw


def _polygon_mask(
    polygon: Sequence[XY], xx: np.ndarray, yy: np.ndarray, half_cell: float = 0.0
) -> np.ndarray:
    """Rasterize public Solid/WalkTop convex polygons at cell centers without private grid APIs."""
    positive, negative = np.ones_like(xx, dtype=bool), np.ones_like(xx, dtype=bool)
    for a, b in zip(polygon, tuple(polygon[1:]) + (polygon[0],)):
        dx, dy = b[0] - a[0], b[1] - a[1]
        cross = dx * (yy - a[1]) - dy * (xx - a[0])
        margin = half_cell * (abs(dx) + abs(dy)) + 1e-9
        positive &= cross >= -margin
        negative &= cross <= margin
    xs, ys = zip(*polygon)
    return (positive | negative) & (
        (xx >= min(xs) - half_cell)
        & (xx <= max(xs) + half_cell)
        & (yy >= min(ys) - half_cell)
        & (yy <= max(ys) + half_cell)
    )


def _site_masks(data: Any, cell_m: float, robot_radius_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Project exported outdoor surfaces and solids using the original grid rules."""
    tolerance = 1e-6
    site = data["site"]
    margin = site["margin_m"]
    nx, ny = [max(1, int(round((size - 2 * margin) / cell_m)))
              for size in data["site_size_m"]]
    xx, yy = np.meshgrid(
        margin + (np.arange(nx) + 0.5) * cell_m,
        margin + (np.arange(ny) + 0.5) * cell_m, indexing="ij",
    )
    height = _terrain_heights(site["heightfield"], xx, yy)
    dx, dy = np.gradient(height, cell_m)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    owner = np.full((nx, ny), -1, dtype=int)
    for k, wt in enumerate(site["walktops"]):
        a, b, c = wt["plane"]
        top = a * xx + b * yy + c
        covered = _polygon_mask(wt["polygon"], xx, yy) & (top >= height - tolerance)
        height[covered] = top[covered]
        slope[covered] = math.degrees(math.atan(math.hypot(a, b)))
        owner[covered] = k
    blocked = np.zeros((nx, ny), dtype=bool)
    for solid in site["solids"]:
        xs, ys = zip(*solid["polygon"])
        thin = min(max(xs) - min(xs), max(ys) - min(ys)) < 2 * cell_m
        covered = _polygon_mask(solid["polygon"], xx, yy, cell_m / 2.0 if thin else 0.0)
        if solid["kind"] != "water":
            z0, z1 = solid["z_range"]
            if z1 - z0 <= 0.02:  # Upstream flat-item cutoff.
                continue
            covered &= (z1 > height + 0.02) & (z0 < height + 0.45)
            if solid["id"] in site["support_owners"]:
                covered &= ~np.isin(owner, site["support_owners"][solid["id"]])
        blocked |= covered
    for axis in (0, 1):
        jump = np.abs(np.diff(height, axis=axis)) > 0.18 + tolerance
        if axis == 0:
            blocked[:-1, :] |= jump
            blocked[1:, :] |= jump
        else:
            blocked[:, :-1] |= jump
            blocked[:, 1:] |= jump
    blocked |= slope > 20.0 + tolerance  # Upstream outdoor slope limit, degrees.
    sx, sy = data["site_size_m"]
    blocked |= (xx < margin) | (xx > sx - margin) | (yy < margin) | (yy > sy - margin)
    r = int(math.ceil(robot_radius_m / cell_m))
    offsets = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
               if math.hypot(di, dj) * cell_m <= robot_radius_m + tolerance]
    return ~_dilate(blocked, offsets), blocked


def _terrain_heights(heightfield: Any, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Interpolate on the original fixed-diagonal terrain triangles."""
    tolerance = 1e-6
    xs, ys = np.asarray(heightfield["xs"]), np.asarray(heightfield["ys"])
    z = np.asarray(heightfield["z"])
    i = np.clip(np.searchsorted(xs[1:] + tolerance, xx, side="left"), 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys[1:] + tolerance, yy, side="left"), 0, len(ys) - 2)
    u = (xx - xs[i]) / (xs[i + 1] - xs[i])
    v = (yy - ys[j]) / (ys[j + 1] - ys[j])
    z00, z10, z01, z11 = z[j, i], z[j, i + 1], z[j + 1, i], z[j + 1, i + 1]
    return np.where(u >= v, z00 + u * (z10 - z00) + v * (z11 - z10),
                    z00 + v * (z01 - z00) + u * (z11 - z01))


@dataclass(frozen=True)
class Transition:
    """Bidirectional floor connection, separate from the legacy single-floor algorithm.

    Elevator endpoints are cab centers and length_m covers vertical travel only.
    Elevator waypoints guide boarding on the floor; they do not define the edge cost.
    """

    kind: str
    a_floor: str
    a_xy: XY
    b_floor: str
    b_xy: XY
    length_m: float
    waypoints: Tuple[XY, ...] = ()  # Stairs: a -> b; elevators: hall door front -> cab center.


@dataclass
class MultiFloorGrid:
    """Reuse TruthGrid per floor and store cross-floor edges separately, preserving single-floor
    APIs.
    """

    scene: str
    floors: Dict[str, TruthGrid]
    transitions: Tuple[Transition, ...]


def _stair_endpoint(stair: Any, bottom: bool) -> XY:
    """Choose exterior checkpoints along stair direction, matching reachability entrance
    definitions.
    """
    flight = stair.flights[0] if bottom else stair.flights[-1]
    x0, x1, y0, y1 = flight.rect
    along = (x0, x1) if flight.ascend.endswith("x") else (y0, y1)
    across = (y0 + y1) / 2.0 if flight.ascend.endswith("x") else (x0 + x1) / 2.0
    sign = 1 if flight.ascend.startswith("+") else -1
    entry, exit_ = along if sign == 1 else along[::-1]
    s = entry - sign * STAIR_APPROACH_M if bottom else exit_ + sign * STAIR_APPROACH_M
    return (s, across) if flight.ascend.endswith("x") else (across, s)


def _stair_waypoints(stair: Any) -> List[XY]:
    """Follow ordered flights and landing centers with at most 1 m between checkpoints.

    Args:
        stair: StairSpec with flights in ascending order and intervening landing rectangles.

    Returns:
        Approach, flight entry/exit, landing center and departure checkpoints, with
        each segment subdivided while preserving every turn and both outer endpoints.
    """
    return [xy for xy, _ in _stair_waypoints_with_rise(stair)]


def _landing_turn_point(stair: Any, index: int) -> XY:
    """Choose a length-only turn inset from the two flights' adjacent landing edges.

    Same-side mouths share one inset edge; perpendicular mouths use an inset
    corner. Opposite mouths use the midpoint of their inset edges as a symmetric
    compromise. Along an unconstrained axis, use the midpoint of flight centers.
    Clamp to the inset rectangle; if a dimension is narrower than the robot
    diameter, use its midpoint rather than putting the turn outside the landing.
    This fallback does not certify that the narrow landing is traversable.

    Args:
        stair: StairSpec with flights and intervening landing rectangles.
        index: Landing index between flights index and index + 1.

    Returns:
        Turn XY inside the landing with radius clearance where its dimensions permit.
    """
    landing = stair.landings[index]
    flights = (stair.flights[index], stair.flights[index + 1])
    point = []
    for axis in (0, 1):
        low, high = landing[2 * axis:2 * axis + 2]
        inset = min(ROBOT_RADIUS_M, (high - low) / 2.0)
        near_edges = []
        for flight, exiting in zip(flights, (True, False)):
            flight_axis = 0 if flight.ascend.endswith("x") else 1
            if flight_axis == axis:
                # Ascending exits follow ascend into the landing; entries reverse it.
                into_positive = flight.ascend.startswith("+") == exiting
                near_edges.append(low + inset if into_positive else high - inset)
        coordinate = (
            sum(near_edges) / len(near_edges) if near_edges else
            sum((f.rect[2 * axis] + f.rect[2 * axis + 1]) / 2.0 for f in flights) / 2.0
        )
        point.append(max(low + inset, min(high - inset, coordinate)))
    return point[0], point[1]


def _stair_waypoints_with_rise(
    stair: Any, *, shortest_landings: bool = False,
) -> List[Tuple[XY, float]]:
    """Pair XY checkpoints with cumulative riser counts.

    Flight interiors interpolate the number of risers climbed; landings and the
    outer approach/departure segments keep a constant count. Navigation retains
    landing centers by default; only length calculation opts into inset turns.
    Subdivision uses XY distance and preserves each anchor.
    """
    rise = 0.0
    anchors = [(_stair_endpoint(stair, bottom=True), rise)]
    for i, flight in enumerate(stair.flights):
        # Equivalent to scene.multistory._ascend_geometry (lines 452-465), using
        # the same local rect/direction derivation as _stair_endpoint, no private import.
        x0, x1, y0, y1 = flight.rect
        along_x = flight.ascend.endswith("x")
        along = (x0, x1) if along_x else (y0, y1)
        across = (y0 + y1) / 2.0 if along_x else (x0 + x1) / 2.0
        entry, exit_ = along if flight.ascend.startswith("+") else along[::-1]
        anchors.append(((entry, across) if along_x else (across, entry), rise))
        rise += flight.n_risers
        anchors.append(((exit_, across) if along_x else (across, exit_), rise))
        if i < len(stair.landings):
            lx0, lx1, ly0, ly1 = stair.landings[i]
            turn = ((lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0)
            if shortest_landings and i + 1 < len(stair.flights):
                turn = _landing_turn_point(stair, i)
            anchors.append((turn, rise))
    anchors.append((_stair_endpoint(stair, bottom=False), rise))

    points = [anchors[0]]
    for (a, a_rise), (b, b_rise) in zip(anchors, anchors[1:]):
        if a == b:
            continue
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(length / STAIR_WAYPOINT_SPACING_M)))
        points.extend(
            ((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n),
             a_rise + (b_rise - a_rise) * k / n)
            for k in range(1, n)
        )
        points.append((b, b_rise))
    return points


def _stair_length_m(spec: Any, stair: Any) -> float:
    """Measure a 3D stair polyline with inset landing turns and scene floor levels."""
    levels = {floor.id: floor.level_m for floor in spec.floors}
    riser = (levels[stair.to_floor] - levels[stair.from_floor]) / sum(
        flight.n_risers for flight in stair.flights
    )
    total_risers = sum(flight.n_risers for flight in stair.flights)
    points = []
    for (x, y), rise in _stair_waypoints_with_rise(stair, shortest_landings=True):
        # Use exact floor levels at both ends, including the entire approach segment.
        z = levels[stair.from_floor] + rise * riser
        if rise == total_risers:
            z = levels[stair.to_floor]
        points.append((x, y, z))
    return sum(
        math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)
        for a, b in zip(points, points[1:])
    )


def build_multifloor_truth_grid(
    scene: str = SCENE_ENV3, cell_m: float = CELL_M, robot_radius_m: float = ROBOT_RADIUS_M
) -> MultiFloorGrid:
    """Read frozen per-floor truth and exact stair/elevator connection costs."""
    data = scene_data(scene)
    if not data["transitions"]:
        raise ValueError(f"Scene has no multistory topology: {scene!r}")
    floors = {floor: _truth_grid_from_data(scene, floor, cell_m, robot_radius_m)
              for floor in data["floors"]}
    transitions = tuple(
        Transition(t["kind"], t["a_floor"], t["a_xy"], t["b_floor"], t["b_xy"],
                   t["length_m"], t["waypoints"])
        for t in data["transitions"]
    )
    return MultiFloorGrid(scene, floors, transitions)


def floor_of_z(spec_or_scene: Union[Any, str], z: float) -> str:
    """Return the nearest floor elevation, assigning midpoint ties to the higher floor.

    Outdoor buildings share one navigation plane. This serves multistory callers
    and leaves the original env0 coordinate handling unchanged.
    """
    if isinstance(spec_or_scene, str):
        data = scene_data(spec_or_scene)
        levels = [(key, value["level_m"]) for key, value in data["floors"].items()]
        if data["outdoor"]:
            return levels[0][0]
    else:
        spec = spec_or_scene
        if spec.outdoor is not None:
            return spec.floors[0].id
        levels = [(floor.id, floor.level_m) for floor in spec.floors]
    return min(levels, key=lambda item: (abs(item[1] - z), -item[1]))[0]


# ---- Line of sight ----


def _segment_samples(a: XY, b: XY, step: float) -> np.ndarray:
    n = max(2, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step)) + 1)
    t = np.linspace(0.0, 1.0, n)
    return np.stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])], axis=1)


def free_line_of_sight(grid: TruthGrid, a: XY, b: XY) -> bool:
    """Check that every sample along a segment lies in a free grid cell."""
    pts = _segment_samples(a, b, grid.cell_m * LOS_STEP_FRAC)
    i = np.floor((pts[:, 0] - grid.x0) / grid.cell_m).astype(int)
    j = np.floor((pts[:, 1] - grid.y0) / grid.cell_m).astype(int)
    nx, ny = grid.free.shape
    if (i < 0).any() or (j < 0).any() or (i >= nx).any() or (j >= ny).any():
        return False
    return bool(grid.free[i, j].all())


def _segment_hits_rect(a: XY, b: XY, r: WallRect) -> bool:
    """Test segment intersection with an axis-aligned rectangle using Liang-Barsky clipping."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - r.x0), (dx, r.x1 - a[0]), (-dy, a[1] - r.y0), (dy, r.y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def wall_line_of_sight(grid: TruthGrid, a: XY, b: XY) -> bool:
    """Check visibility through wall gaps; door openings are clear and furniture does not occlude.
    """
    return not any(_segment_hits_rect(a, b, w) for w in grid.walls)


# ---- Shortest paths ----

_NEIGHBORS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
)  # fmt: skip


def distance_field(grid: TruthGrid, source: Cell) -> np.ndarray:
    """Eight-neighbor Dijkstra from source in cell units, with diagonal cost sqrt(2).

    Forbid diagonal corner cutting and assign infinity to unreachable cells.
    """
    free = grid.free
    nx, ny = free.shape
    dist = np.full((nx, ny), np.inf)
    if not grid.in_bounds(source) or not free[source]:
        return dist
    dist[source] = 0.0
    heap = [(0.0, source[0], source[1])]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj, w in _NEIGHBORS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not free[ni, nj]:
                continue
            if di and dj and not (free[i + di, j] and free[i, j + dj]):
                continue
            nd = d + w
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                heapq.heappush(heap, (nd, ni, nj))
    return dist


def _descend(dist: np.ndarray, grid: TruthGrid, start: Cell) -> List[Cell]:
    """Descend the distance field from start to its zero-distance source, returning a cell polyline.
    """
    path = [start]
    cur = start
    nx, ny = dist.shape
    free = grid.free
    while dist[cur] > 0.0:
        best, best_d = None, dist[cur]
        for di, dj, w in _NEIGHBORS:
            ni, nj = cur[0] + di, cur[1] + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if di and dj and not (free[cur[0] + di, cur[1]] and free[cur[0], cur[1] + dj]):
                continue
            cand = dist[ni, nj] + w
            if dist[ni, nj] < best_d and cand <= dist[cur] + 1e-9:
                best, best_d = (ni, nj), dist[ni, nj]
        if best is None:
            raise RuntimeError("距离场下降卡住(不应发生)")
        path.append(best)
        cur = best
    return path


def pull_string(grid: TruthGrid, points: Sequence[XY]) -> List[XY]:
    """Extend from each anchor until visibility breaks, then choose the last visible point as
    anchor.
    """
    if len(points) <= 2:
        return list(points)
    out = [points[0]]
    k = 0
    while k < len(points) - 1:
        nxt = k + 1
        while nxt + 1 < len(points) and free_line_of_sight(grid, points[k], points[nxt + 1]):
            nxt += 1
        out.append(points[nxt])
        k = nxt
    return out


def polyline_length(points: Sequence[XY]) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])))


@dataclass(frozen=True)
class GeodesicPath:
    length_m: float
    waypoints: Tuple[XY, ...]  # Navigation polyline, including dense transition checkpoints.
    grid_length_m: float  # Eight-neighbor path length before straightening, for diagnostics
    elevator_rides: int = 0
    elevator_vertical_m: float = 0.0


def shortest_path(
    grid: TruthGrid, start: XY, goal: XY, field: Optional[np.ndarray] = None
) -> Optional[GeodesicPath]:
    """Return the truth shortest path, or None for blocked or disconnected endpoints.

    Reuse an optional goal-sourced ``field`` when evaluating multiple starts.
    """
    if not (grid.is_free_xy(*start) and grid.is_free_xy(*goal)):
        return None
    if free_line_of_sight(grid, start, goal):
        d = math.hypot(goal[0] - start[0], goal[1] - start[1])
        return GeodesicPath(d, (start, goal), d)
    goal_cell = grid.cell_of(*goal)
    dist = field if field is not None else distance_field(grid, goal_cell)
    start_cell = grid.cell_of(*start)
    if not np.isfinite(dist[start_cell]):
        return None
    cells = _descend(dist, grid, start_cell)
    pts: List[XY] = [start] + [grid.center_of(c) for c in cells[1:-1]] + [goal]
    pulled = pull_string(grid, pts)
    return GeodesicPath(polyline_length(pulled), tuple(pulled), polyline_length(pts))


def nearest_free(grid: TruthGrid, xy: XY, max_m: float) -> Optional[XY]:
    """Keep a free xy point; otherwise snap to the nearest free center within max_m, or return None.
    """
    if grid.is_free_xy(*xy):
        return (float(xy[0]), float(xy[1]))
    ci, cj = grid.cell_of(*xy)
    r = int(math.ceil(max_m / grid.cell_m))
    best: Optional[Tuple[float, XY]] = None
    for i in range(ci - r, ci + r + 1):
        for j in range(cj - r, cj + r + 1):
            if not grid.in_bounds((i, j)) or not grid.free[i, j]:
                continue
            c = grid.center_of((i, j))
            d = math.hypot(c[0] - xy[0], c[1] - xy[1])
            if d <= max_m and (best is None or d < best[0]):
                best = (d, c)
    return None if best is None else best[1]


def multifloor_shortest_path(
    mgrid: MultiFloorGrid, start_floor: str, start: XY, goal_floor: str, goal: XY
) -> Optional[GeodesicPath]:
    """Use the legacy same-floor algorithm or Dijkstra over stair/elevator endpoint nodes.

    Isolate single-floor callers. Blocked endpoints get no edges; return None
    without a fully connected route. Include connector lengths in both length
    fields. waypoints retain XY only; floor identities stay in internal graph nodes.
    """
    if start_floor not in mgrid.floors or goal_floor not in mgrid.floors:
        return None
    if start_floor == goal_floor:
        return shortest_path(mgrid.floors[start_floor], start, goal)
    if not (
        mgrid.floors[start_floor].is_free_xy(*start) and mgrid.floors[goal_floor].is_free_xy(*goal)
    ):
        return None
    nodes = [(start_floor, start), (goal_floor, goal)]
    edges: Dict[int, Dict[int, GeodesicPath]] = {0: {}, 1: {}}
    boarding: Dict[int, Tuple[XY, ...]] = {}
    for t in mgrid.transitions:
        if t.a_floor not in mgrid.floors or t.b_floor not in mgrid.floors:
            continue
        checkpoints = t.waypoints or (t.a_xy, t.b_xy)
        if not (
            mgrid.floors[t.a_floor].is_free_xy(*t.a_xy)
            and mgrid.floors[t.b_floor].is_free_xy(*t.b_xy)
        ):
            continue
        a, b = len(nodes), len(nodes) + 1
        nodes.extend([(t.a_floor, t.a_xy), (t.b_floor, t.b_xy)])
        rides, vertical = (1, t.length_m) if t.kind == "elevator" else (0, 0.0)
        if t.kind == "elevator":
            # Cost endpoints stay at the cab center. Boarding belongs to same-floor travel;
            # its dense checkpoints guide navigation independently of the distance metric.
            checkpoints = (t.a_xy, t.b_xy)
            if t.waypoints:
                boarding[a] = boarding[b] = t.waypoints
        edges[a] = {b: GeodesicPath(t.length_m, checkpoints, t.length_m, rides, vertical)}
        edges[b] = {a: GeodesicPath(
            t.length_m, tuple(reversed(checkpoints)), t.length_m, rides, vertical,
        )}
    for i, (floor, xy) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            other_floor, other_xy = nodes[j]
            if floor != other_floor:
                continue
            path = shortest_path(mgrid.floors[floor], xy, other_xy)
            if path is not None:
                if xy != other_xy and (i in boarding or j in boarding):
                    departure = tuple(reversed(boarding[i])) if i in boarding else (xy,)
                    arrival = boarding.get(j, (other_xy,))
                    hall = shortest_path(mgrid.floors[floor], departure[-1], arrival[0])
                    if hall is None:
                        continue
                    navigation = departure + hall.waypoints[1:] + arrival[1:]
                    path = GeodesicPath(path.length_m, navigation, path.grid_length_m)
                edges[i][j] = path
                edges[j][i] = GeodesicPath(
                    path.length_m, tuple(reversed(path.waypoints)), path.grid_length_m
                )
    dist = {0: 0.0}
    previous: Dict[int, int] = {}
    heap = [(0.0, 0)]
    while heap:
        length, node = heapq.heappop(heap)
        if length > dist[node]:
            continue
        if node == 1:
            chain = [1]
            while chain[-1] != 0:
                chain.append(previous[chain[-1]])
            chain.reverse()
            segments = [edges[a][b] for a, b in zip(chain, chain[1:])]
            points = [start]
            for segment in segments:
                for point in segment.waypoints[1:]:
                    if point != points[-1] or len(points) == 1:
                        points.append(point)
            return GeodesicPath(
                length, tuple(points), sum(s.grid_length_m for s in segments),
                sum(s.elevator_rides for s in segments),
                sum(s.elevator_vertical_m for s in segments),
            )
        for neighbor, path in edges[node].items():
            candidate = length + path.length_m
            if candidate < dist.get(neighbor, float("inf")):
                dist[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))
    return None


def route_shortest_path(
    grid: Union[TruthGrid, MultiFloorGrid], start: XY, goal: XY, *,
    start_floor: str = "", goal_floor: str = "", snap_max_m: float = ROUTE_SNAP_MAX_M,
) -> Optional[GeodesicPath]:
    """Compute route truth with bounded endpoint snapping and all available transitions.

    Args:
        grid: Single-floor or multifloor truth grid.
        start: World XY start, before snapping.
        goal: World XY goal, before snapping.
        start_floor: Required start floor ID for a multifloor grid.
        goal_floor: Required goal floor ID for a multifloor grid.
        snap_max_m: Maximum endpoint displacement; zero preserves legacy strict checks.

    Returns:
        The shortest path, or None for unknown floors, unsnappable or disconnected endpoints.
        Distance uses cab-center grid paths plus vertical floor differences; navigation
        waypoints retain door-front checkpoints. No transition-kind filtering is applied.
    """
    if isinstance(grid, MultiFloorGrid):
        if start_floor not in grid.floors or goal_floor not in grid.floors:
            return None
        start_grid, goal_grid = grid.floors[start_floor], grid.floors[goal_floor]
    else:
        start_grid = goal_grid = grid
    a = nearest_free(start_grid, start, snap_max_m)
    b = nearest_free(goal_grid, goal, snap_max_m)
    if a is None or b is None:
        return None
    if isinstance(grid, MultiFloorGrid):
        return multifloor_shortest_path(grid, start_floor, a, goal_floor, b)
    return shortest_path(grid, a, b)


def grid_digest(grid: TruthGrid) -> Dict[str, object]:
    """Grid fingerprint stored in frozen episode sets and run.json."""
    import hashlib

    return {
        "scene": grid.scene,
        "cell_m": grid.cell_m,
        "shape": list(grid.free.shape),
        "free_cells": grid.free_count(),
        "free_sha256": hashlib.sha256(np.packbits(grid.free).tobytes()).hexdigest(),
    }
