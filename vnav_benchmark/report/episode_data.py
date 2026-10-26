"""Read an episode directory into EpisodeData for videos and reports, without rendering.

The loader is read-only and uses no simulator or network. Missing fields receive empty values
rather than inferred data.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vnav_benchmark.fairness import parse_method_id
from vnav_benchmark.report.results_io import METHOD_CODES
from vnav_benchmark.rundir import TRACE_COLUMNS
from vnav_benchmark.vlm.memory import HISTORY_SIZE
from vnav_benchmark.vlm.prompt import SUBGOAL_LABEL

HIST_WIDTH = HISTORY_SIZE[
    0
]  # History frames are 320 px wide, distinguishing them from goal, subgoal and current frames.
_SUBGOAL_PREFIX = SUBGOAL_LABEL.split(" (subgoal")[
    0
]  # The subgoal image immediately follows the CURRENT SUBGOAL IMAGE text block.


@dataclass(frozen=True)
class DecisionInfo:
    """One decision assembled from a steps.jsonl row and optional VLM-side logs."""

    i: int
    t_sim: float
    applied: Tuple[float, float, float]  # Actual clipped command (vx, vy, wz).
    hold_s: float
    done: bool
    note: str
    debug: Dict[str, Any]
    act_wall_s: float
    # Same-decision row from method/vlm/steps.jsonl; None for point-goal methods.
    vlm: Optional[Dict[str, Any]]
    # History frames shown to the model, oldest first; empty without a VLM.
    history_jpegs: Tuple[bytes, ...]
    subgoal_jpeg: Optional[bytes]  # Subgoal image shown to the model, or None.
    usage_usd: float  # Total billed_usd across this decision's attempts.
    tokens_in: int
    tokens_out: int
    subgoal_kf: Optional[int] = None  # Current subgoal keyframe ID, or None.


@dataclass(frozen=True)
class RouteKeyframes:
    """Planned route keyframes: order, world poses and photographs for video annotations."""

    kf_ids: Tuple[int, ...]  # Planned route order.
    # Keyframe body poses transformed into simulation world coordinates.
    poses_world: Dict[int, Tuple[float, float, float]]
    jpeg: Callable[[int], bytes]  # Keyframe photographs.


@dataclass
class EpisodeData:
    """All read-only data loaded from one episode directory."""

    run_dir: Path
    run: Dict[str, Any]
    method_id: str
    code: str  # B0 / M1 / M2 / M3 / M4
    planner: str  # "ptgoal" | "imggoal"
    vlm: Optional[str]  # "astra" | "qwen" | None
    # Shape (N, 8): t_sim, x, y, yaw, up_z, vx_body, wz_body, z, matching TRACE_COLUMNS.
    trace: np.ndarray
    ego_frames: List[Tuple[float, Path]]  # Time-sorted (time in seconds, path) pairs.
    depth_frames: List[Tuple[float, Path]]
    decisions: List[DecisionInfo]
    goal_jpeg: bytes
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    # Ground-truth shortest-path polyline in world coordinates.
    waypoints: List[Tuple[float, float]]
    outcome: Dict[str, Any]
    # oracle.TruthGrid or a compatible test grid, injectable when loading.
    grid: Optional[Any] = None
    route: Optional[RouteKeyframes] = None  # None for mapless methods or unavailable map views.

    def visited_kfs(self, t: float) -> List[int]:
        """Return previously replaced subgoal keyframes up to time t, in traversal order."""
        current = self.decision_at(t)
        cur_kf = current.subgoal_kf if current is not None else None
        out: List[int] = []
        for d in self.decisions:
            if d.t_sim > t:
                break
            if d.subgoal_kf is not None and d.subgoal_kf != cur_kf and d.subgoal_kf not in out:
                out.append(d.subgoal_kf)
        return out

    def frame_at(self, t: float) -> Optional[Path]:
        """Return the latest first-person frame at or before t, or None before the first frame."""
        return _path_at_or_before(self.ego_frames, t)

    def depth_at(self, t: float) -> Optional[Path]:
        """Select a depth frame using the same timestamp rule as frame_at."""
        return _path_at_or_before(self.depth_frames, t)

    def decision_at(self, t: float) -> Optional[DecisionInfo]:
        """Return the last decision with t_sim <= t, or None before the first decision."""
        times = [d.t_sim for d in self.decisions]
        idx = bisect.bisect_right(times, t) - 1
        return self.decisions[idx] if idx >= 0 else None

    def pose_at(self, t: float) -> Tuple[float, float, float]:
        """Linearly interpolate trace x, y and unwrapped yaw, clamping at either endpoint."""
        if self.trace.shape[0] == 0:
            return (0.0, 0.0, 0.0)
        ts = self.trace[:, 0]
        x = float(np.interp(t, ts, self.trace[:, 1]))
        y = float(np.interp(t, ts, self.trace[:, 2]))
        yaw_unwrapped = np.unwrap(self.trace[:, 3])
        yaw = float(np.interp(t, ts, yaw_unwrapped))
        return (x, y, yaw)


def _path_at_or_before(frames: Sequence[Tuple[float, Path]], t: float) -> Optional[Path]:
    times = [f[0] for f in frames]
    idx = bisect.bisect_right(times, t) - 1
    return frames[idx][1] if idx >= 0 else None


def _frames_in(dir_path: Path, suffix: str) -> List[Tuple[float, Path]]:
    """List <dir_path>/<8-digit-t_ms><suffix> files as time-sorted (seconds, path) pairs."""
    if not dir_path.is_dir():
        return []
    out: List[Tuple[float, Path]] = []
    for p in dir_path.glob(f"*{suffix}"):
        try:
            t_ms = int(p.stem)
        except ValueError:
            continue
        out.append((t_ms / 1000.0, p))
    out.sort(key=lambda item: item[0])
    return out


def _load_trace(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.empty((0, len(TRACE_COLUMNS)), dtype=np.float64)
    import csv

    rows: List[List[float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # Historical campaigns predate the z column; retain their planar plots.
            rows.append([float(row.get(c, 0.0) if c == "z" else row[c]) for c in TRACE_COLUMNS])
    if not rows:
        return np.empty((0, len(TRACE_COLUMNS)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _read_image(images_dir: Path, sha: str) -> bytes:
    return (images_dir / f"{sha}.jpg").read_bytes()


def _extract_vlm_images(
    last_call: Dict[str, Any], images_dir: Path
) -> Tuple[Tuple[bytes, ...], Optional[bytes]]:
    """Extract history and subgoal images from the last attempt's calls.jsonl parts.

    History image width equals HIST_WIDTH. The subgoal image follows the CURRENT SUBGOAL IMAGE
    text block.
    """
    history: List[bytes] = []
    subgoal: Optional[bytes] = None
    want_subgoal_next = False
    for part in last_call.get("parts", []):
        if "img" in part:
            if want_subgoal_next:
                subgoal = _read_image(images_dir, part["img"])
                want_subgoal_next = False
            elif int(part.get("w", 0)) == HIST_WIDTH:
                history.append(_read_image(images_dir, part["img"]))
        else:
            want_subgoal_next = str(part.get("t", "")).startswith(_SUBGOAL_PREFIX)
    return tuple(history), subgoal


def _vlm_usage(calls_for_decision: Sequence[Dict[str, Any]]) -> Tuple[float, int, int]:
    """Sum billed_usd, input_tokens and output_tokens over all attempts for a decision."""
    usd = 0.0
    tin = 0
    tout = 0
    for row in calls_for_decision:
        usage = row.get("usage")
        if not usage:
            continue
        usd += float(usage.get("billed_usd", 0.0))
        tin += int(usage.get("input_tokens", 0))
        tout += int(usage.get("output_tokens", 0))
    return usd, tin, tout


def _decision_info(
    row: Dict[str, Any],
    vlm_steps: Dict[int, Dict[str, Any]],
    vlm_calls: Dict[int, List[Dict[str, Any]]],
    images_dir: Path,
) -> DecisionInfo:
    i = int(row["i"])
    applied = row.get("applied") or {}
    action = row.get("action") or {}
    vlm_row = vlm_steps.get(i)
    calls_for_i = vlm_calls.get(i, [])
    history: Tuple[bytes, ...] = ()
    subgoal_jpeg: Optional[bytes] = None
    usage_usd, tokens_in, tokens_out = 0.0, 0, 0
    if calls_for_i:
        usage_usd, tokens_in, tokens_out = _vlm_usage(calls_for_i)
        history, subgoal_jpeg = _extract_vlm_images(calls_for_i[-1], images_dir)
    return DecisionInfo(
        i=i,
        t_sim=float(row["t_sim"]),
        applied=(
            float(applied.get("vx", 0.0)),
            float(applied.get("vy", 0.0)),
            float(applied.get("wz", 0.0)),
        ),
        hold_s=float(applied.get("hold_s", 0.0)),
        done=bool(action.get("done", False)),
        note=str(row.get("note", "")),
        debug=dict(row.get("debug") or {}),
        act_wall_s=float(row.get("act_wall_s", 0.0)),
        vlm=vlm_row,
        history_jpegs=history,
        subgoal_jpeg=subgoal_jpeg,
        usage_usd=usage_usd,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        subgoal_kf=_subgoal_kf(row, vlm_row),
    )


def _subgoal_kf(row: Dict[str, Any], vlm_row: Optional[Dict[str, Any]]) -> Optional[int]:
    """Read point-goal debug[kf] or image-goal debug[subgoal_kf] keyframe IDs."""
    debug = row.get("debug") or {}
    for value in (debug.get("subgoal_kf"), debug.get("kf"), (vlm_row or {}).get("subgoal_kf")):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _route_kf_ids(run_dir: Path) -> List[int]:
    """Read route order from method/route.json or image-goal method/vlm/summary.json."""
    route_json = run_dir / "method" / "route.json"
    if route_json.is_file():
        return [int(k) for k in json.loads(route_json.read_text(encoding="utf-8"))["kf_path"]]
    summary = run_dir / "method" / "vlm" / "summary.json"
    if summary.is_file():
        info = json.loads(summary.read_text(encoding="utf-8")).get("route") or {}
        return [int(k) for k in info.get("kf_ids", [])]
    return []


def load_route_keyframes(run_dir: Path, run: Dict[str, Any]) -> Optional[RouteKeyframes]:
    """Load planned keyframes from campaign map views and maps.yaml map_to_world transforms.

    Return None for mapless methods or missing views/entries; videos still render without the
    route.
    """
    map_info = (run.get("settings") or {}).get("map")
    kf_ids = _route_kf_ids(run_dir)
    if not map_info or not kf_ids:
        return None
    from vnav_benchmark import mapview

    campaign = run_dir.parents[3]  # runs/<method>/<token>/r<k>
    view = campaign / "maps" / str(map_info["map_id"])
    try:
        config = mapview.load_maps_config()
        entry = config.maps[str(map_info["map_id"])]
        handle = mapview.SanitizedMap(
            view, entry.map_id, entry.variant, str(map_info.get("edge_type", entry.edge_type)),
            config.camera_forward_m,
        )  # fmt: skip
    except (KeyError, OSError, mapview.MapViewError):
        return None
    m2w = np.asarray(entry.map_to_world, dtype=np.float64)
    yaw_off = float(np.arctan2(m2w[1, 0], m2w[0, 0]))
    poses: Dict[int, Tuple[float, float, float]] = {}
    for kf in kf_ids:
        p = handle.keyframe_pose(kf)
        world = m2w @ np.array([p.x, p.y, 0.0, 1.0])
        poses[kf] = (float(world[0]), float(world[1]), float(p.yaw + yaw_off))
    return RouteKeyframes(tuple(kf_ids), poses, handle.keyframe_rgb_jpeg)


def _load_vlm_logs(
    method_dir: Path,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    steps: Dict[int, Dict[str, Any]] = {
        int(row["decision"]): row for row in _load_jsonl(method_dir / "vlm" / "steps.jsonl")
    }
    calls: Dict[int, List[Dict[str, Any]]] = {}
    for row in _load_jsonl(method_dir / "vlm" / "calls.jsonl"):
        calls.setdefault(int(row["decision"]), []).append(row)
    for rows in calls.values():
        rows.sort(key=lambda r: int(r["attempt"]))
    return steps, calls


def load_episode(
    run_dir: Path,
    grid_fn: Optional[Callable[[str], Any]] = None,
    route_fn: Optional[Callable[[Path, Dict[str, Any]], Optional[RouteKeyframes]]] = None,
) -> EpisodeData:
    """Load a read-only episode directory.

    Inject grid_fn to supply a test grid; otherwise build the true grid for run.json
    episode.scene. Inject route_fn similarly; the default is load_route_keyframes.
    """
    run_dir = Path(run_dir)
    run: Dict[str, Any] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    method_id = str(run["method_id"])
    profile = parse_method_id(method_id)
    code = METHOD_CODES["-".join(method_id.split("-")[:3])]

    method_dir = run_dir / "method"
    vlm_steps, vlm_calls = _load_vlm_logs(method_dir)
    images_dir = method_dir / "vlm" / "images"
    decisions = [
        _decision_info(row, vlm_steps, vlm_calls, images_dir)
        for row in _load_jsonl(run_dir / "steps.jsonl")
    ]

    ep = run["episode"]
    start = (float(ep["start"]["x"]), float(ep["start"]["y"]), float(ep["start"]["yaw"]))
    goal = (float(ep["goal"]["x"]), float(ep["goal"]["y"]), float(ep["goal"]["yaw"]))
    waypoints = [(float(p[0]), float(p[1])) for p in ep.get("waypoints", [])]

    goal_path = run_dir / "goal" / "goal.jpg"
    goal_jpeg = goal_path.read_bytes() if goal_path.is_file() else b""

    if grid_fn is not None:
        grid: Optional[Any] = grid_fn(str(ep["scene"]))
    else:
        from vnav_benchmark.oracle import build_truth_grid

        grid = build_truth_grid(str(ep["scene"]))

    return EpisodeData(
        run_dir=run_dir,
        run=run,
        method_id=method_id,
        code=code,
        planner=profile.planner,
        vlm=profile.vlm,
        trace=_load_trace(run_dir / "trace.csv"),
        ego_frames=_frames_in(run_dir / "frames" / "ego", ".jpg"),
        depth_frames=_frames_in(run_dir / "depth", ".png"),
        decisions=decisions,
        goal_jpeg=goal_jpeg,
        start=start,
        goal=goal,
        waypoints=waypoints,
        outcome=dict(run.get("outcome") or {}),
        grid=grid,
        route=(route_fn or load_route_keyframes)(run_dir, run),
    )
