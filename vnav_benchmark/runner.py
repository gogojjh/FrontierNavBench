"""Campaign execution in the frozen evaluation layer.

Each episode starts and settles a fresh simulator outside its budget, resets the
method, then alternates act()/step() until done, timeout, fall, method exception,
act wall time over 300 s or episode cost over $5. Evaluate using truth and write
run.json atomically.
InfraError from simulation, providers or SRU is unscored and retried with a fresh
simulator up to three times, then recorded as unscored_infra. FatalInfraError,
including exhausted provider credit, stops the campaign. Other exceptions count
as scored method failures.
Job order is deterministic: group by endpoint pair and rotate the first method.
Resume skips cells with run.json. Three consecutive infrastructure failures trip
the campaign circuit breaker and alert. BudgetGuard blocks predicted overspend.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import yaml

from vnav_benchmark import fairness, metrics, oracle
from vnav_benchmark.budget import VLM_PROVIDER, BudgetGuard, EpisodeUsage
from vnav_benchmark.episodes import EpisodeRecord, EpisodeSet
from vnav_benchmark.interfaces import (
    Action,
    FatalInfraError,
    InfraError,
    MapHandle,
    Method,
    MethodContext,
    Observation,
    Pose2D,
)
from vnav_benchmark.localizer import OracleLocalizer
from vnav_benchmark.rundir import RUN_SCHEMA_ID, RunDir, write_json_atomic
from vnav_benchmark.sim import ctrl_protocol as cp
from vnav_benchmark.sim.stage_check import sim_dir_sha256
from vnav_benchmark.sim_env import SimEnv, SimLink, pose2d_of

REPO = Path(__file__).resolve().parents[1]
BENCH_CONFIG = REPO / "configs" / "bench.yaml"
METHOD_CONFIG_DIR = REPO / "configs" / "methods"
CAMPAIGN_SCHEMA = "vnav.campaign/1"
COVERAGE_SCHEMA = "vnav.coverage/1"
LOCK_NAME = ".campaign.lock"  # Exclusive lock released on process exit.

LOG = logging.getLogger("vnav.runner")


# ---- Configuration ----


@dataclass(frozen=True)
class BenchConfig:
    budget_s: float = 90.0  # User-approved value from October 21.
    seed: int = 20261020
    act_wall_limit_s: float = 300.0
    max_attempts: int = 3
    consecutive_infra_break: int = 3
    robot: str = "go2"  # Evaluation platform: Go2, selected on October 21.
    budget_s_by_scene: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = BENCH_CONFIG) -> BenchConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != "vnav.bench/1":
            raise ValueError(f"{path}: schema 不对")
        check = data.get("fairness", {})
        expect = {
            "vx_min": fairness.VX_MIN, "vx_max": fairness.VX_MAX, "vy_max": fairness.VY_MAX,
            "wz_max": fairness.WZ_MAX, "hold_quantum_s": fairness.HOLD_QUANTUM_S,
            "hold_min_s": fairness.HOLD_MIN_S, "hold_max_s": fairness.HOLD_MAX_S,
            "success_dist_m": metrics.SUCCESS_DIST_M, "success_yaw_deg": metrics.SUCCESS_YAW_DEG,
            "success_loose_m": metrics.SUCCESS_DIST_LOOSE_M,
            "success_loose_yaw_deg": metrics.SUCCESS_YAW_LOOSE_DEG,
            "grid_robot_radius_m": oracle.ROBOT_RADIUS_M,
        }  # fmt: skip
        diff = {k: (check.get(k), v) for k, v in expect.items() if check.get(k) != v}
        if diff:
            raise ValueError(f"bench.yaml 与代码常量不一致(yaml, 代码): {diff}")
        run = data["run"]
        return cls(
            budget_s=float(run["budget_s"]),
            seed=int(run["seed"]),
            act_wall_limit_s=float(run["act_wall_limit_s"]),
            max_attempts=int(run["max_attempts"]),
            consecutive_infra_break=int(run["consecutive_infra_break"]),
            robot=str(run["robot"]),
            budget_s_by_scene={
                scene: float(budget) for scene, budget in run.get("budget_s_by_scene", {}).items()
            },
        )

    def budget_s_for(self, scene: str) -> float:
        """Return the scene override or the default episode budget in seconds."""
        return self.budget_s_by_scene.get(scene, self.budget_s)

    def settings(self, scene: Optional[str] = None) -> Dict[str, Any]:
        return {
            "robot": self.robot,
            "budget_s": self.budget_s_for(scene) if scene is not None else self.budget_s,
            "seed": self.seed,
            "act_wall_limit_s": self.act_wall_limit_s,
            "limits": {"vx_min": fairness.VX_MIN, "vx_max": fairness.VX_MAX,
                       "vy_max": fairness.VY_MAX, "wz_max": fairness.WZ_MAX},
            "hold": {"quantum_s": fairness.HOLD_QUANTUM_S, "min_s": fairness.HOLD_MIN_S,
                     "max_s": fairness.HOLD_MAX_S},
            "success": {"dist_m": metrics.SUCCESS_DIST_M, "yaw_deg": metrics.SUCCESS_YAW_DEG,
                        "loose_dist_m": metrics.SUCCESS_DIST_LOOSE_M,
                        "loose_yaw_deg": metrics.SUCCESS_YAW_LOOSE_DEG},
        }  # fmt: skip


def method_config(
    method_id: str, directory: Path = METHOD_CONFIG_DIR
) -> Tuple[Dict[str, Any], str]:
    """Load method config and SHA256; missing files return ({}, "")."""
    path = Path(directory) / f"{method_id}.yaml"
    if not path.is_file():
        return {}, ""
    raw = path.read_bytes()
    return dict(yaml.safe_load(raw) or {}), hashlib.sha256(raw).hexdigest()


# ---- Injectable external dependencies ----


class SimSession(Protocol):
    link: SimLink
    identity: Dict[str, Any]

    def close(self) -> None: ...


class SimProvider(Protocol):
    def open(self, scene: str, start: Pose2D, third: bool, log_dir: Path) -> SimSession: ...


class MapSource(Protocol):
    """Select a route map by route and density.

    Returns a sanitized map, sixteen map_to_world values and run.json metadata.
    """

    def get(
        self, route: str, variant: str
    ) -> Tuple[MapHandle, Sequence[float], Dict[str, Any]]: ...


@dataclass
class _Session:
    link: Any
    identity: Dict[str, Any]
    closer: Callable[[], None] = lambda: None

    def close(self) -> None:
        self.closer()


class KinematicProvider:
    """Kinematic provider with grid obstacles and injectable per-attempt faults."""

    def __init__(self, grid: Optional[oracle.TruthGrid] = None) -> None:
        self.grid = grid
        self.fail_next: List[str] = []  # InfraError messages for subsequent opens.
        self.opened = 0

    def open(self, scene: str, start: Pose2D, third: bool, log_dir: Path) -> SimSession:
        from vnav_benchmark.sim.backend import KinematicBackend
        from vnav_benchmark.sim.ctrl_client import InProcessClient
        from vnav_benchmark.sim.ctrl_server import CtrlServer

        self.opened += 1
        if self.fail_next:
            raise InfraError(self.fail_next.pop(0))
        # Use uninflated walls/furniture; inflated clearance is not a collision.
        occ = None if self.grid is None else self.grid.raw_blocked.T
        origin = (0.0, 0.0) if self.grid is None else (self.grid.x0, self.grid.y0)
        cell = 0.1 if self.grid is None else self.grid.cell_m
        backend = KinematicBackend(scene, occupancy=occ, origin_xy=origin, cell_m=cell)
        client = InProcessClient(CtrlServer(backend, "inproc://kinematic"))
        identity = client.hello()
        return _Session(client, identity)


class IsaacProvider:
    """Start a fresh Isaac service per episode through the orchestrator."""

    def __init__(self, orchestrator: Any, robot: Optional[str] = None) -> None:
        self.orch = orchestrator
        self.robot = robot  # If supplied, validate the robot field in server hello.

    def open(self, scene: str, start: Pose2D, third: bool, log_dir: Path) -> SimSession:
        from vnav_benchmark.sim.ctrl_client import CtrlClient

        endpoint = self.orch.start(
            scene, (start.x, start.y), start.yaw, log_dir / "isaac.log", third
        )
        client = CtrlClient(endpoint)
        try:
            hello = client.hello()
            local_sha = sim_dir_sha256()
            served_sha = hello.get("sim_dir_sha256")
            if served_sha != local_sha:
                raise FatalInfraError(
                    "台式机仿真代码不一致或服务缺少指纹,拒绝跑批: "
                    f"sim_dir_sha256 客户端(Jetson)={local_sha}, "
                    f"服务端(台式机)={served_sha!r}。"
                    "请在客户端提交需要同步的代码后运行 bash scripts/desktop_stage.sh,"
                    "再重新跑批(脚本只同步已提交代码)。"
                )
        except Exception:
            client.close()
            self.orch.stop()
            raise

        served = self.orch.identity().get("robot")
        if self.robot is not None and served != self.robot:  # Wrong robot: retry unscored.
            client.close()
            self.orch.stop()
            raise InfraError(f"台式机服务的机器人是 {served!r},跑批要 {self.robot!r}")

        def closer() -> None:
            client.close()
            self.orch.stop()  # Request control shutdown, then clean up the owned process group.

        return _Session(client, self.orch.identity(), closer)


class MapsYamlSource:
    """Provide sanitized maps from maps.yaml, building each view once per campaign."""

    def __init__(self, views_root: Path) -> None:
        from vnav_benchmark import mapview

        self._mapview = mapview
        self._config = mapview.load_maps_config()
        self.views_root = Path(views_root)
        self._cache: Dict[Tuple[str, str], Tuple[MapHandle, Sequence[float], Dict[str, Any]]] = {}

    def get(self, route: str, variant: str) -> Tuple[MapHandle, Sequence[float], Dict[str, Any]]:
        key = (route, variant)
        if key not in self._cache:
            entry = self._config.for_route(route, variant)
            handle = self._mapview.materialize(
                entry, self.views_root, self._config.camera_forward_m
            )
            info = {
                "map_id": entry.map_id,
                "route": route,
                "variant": entry.variant,
                "role": entry.role,
                "sha256": handle.sha256,
                "edge_type": entry.edge_type,
            }
            self._cache[key] = (handle, entry.map_to_world.reshape(-1).tolist(), info)
        return self._cache[key]


# ---- One episode ----


@dataclass(frozen=True)
class Job:
    method_id: str
    token: str
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.method_id}/{self.token}/r{self.repeat}"


@dataclass
class RunContext:
    root: Path
    episode_set: EpisodeSet
    sim: SimProvider
    maps: Optional[MapSource]
    guard: BudgetGuard
    bench: BenchConfig = field(default_factory=BenchConfig)
    method_factory: Optional[Callable[[str], Method]] = None
    want_third: Callable[[Job], bool] = lambda job: False
    git: Dict[str, Any] = field(default_factory=dict)
    grid: Optional[oracle.TruthGrid] = None  # Lazy load for debug methods.
    multifloor_grid: Optional[oracle.MultiFloorGrid] = None  # Lazy load for env3 debug methods.

    def truth_grid(self) -> oracle.TruthGrid:
        if self.grid is None:
            self.grid = oracle.build_truth_grid(self.episode_set.payload["scene"])
        return self.grid

    def truth_multifloor_grid(self) -> Optional[oracle.MultiFloorGrid]:
        scene = self.episode_set.payload["scene"]
        if scene != oracle.SCENE_ENV3:
            return None
        if self.multifloor_grid is None:
            self.multifloor_grid = oracle.build_multifloor_truth_grid(scene)
        return self.multifloor_grid

    def make_method(self, method_id: str) -> Method:
        if self.method_factory is not None:
            return self.method_factory(method_id)
        from vnav_benchmark.methods import make_method

        return make_method(method_id)


class _Abort(Exception):
    """Leave the episode loop while carrying its outcome."""

    def __init__(self, termination: str) -> None:
        super().__init__(termination)
        self.termination = termination


def provider_of(method_id: str) -> Optional[str]:
    vlm = fairness.parse_method_id(method_id).vlm
    return VLM_PROVIDER.get(vlm) if vlm else None


def _rng_for(seed: int, job: Job) -> np.random.Generator:
    digest = hashlib.sha256(f"{job.method_id}|{job.token}|{job.repeat}".encode()).digest()
    return np.random.default_rng([seed, int.from_bytes(digest[:8], "little")])


def _trace_rows(samples: Sequence[Tuple[float, cp.WirePose]]) -> List[Dict[str, float]]:
    rows = []
    for t, pose in samples:
        p = pose2d_of(pose)
        rows.append({"t_sim": t, "x": p.x, "y": p.y, "z": pose.position_xyz[2],
                     "yaw": p.yaw, "up_z": pose.up_z,
                     "vx_body": pose.v_body[0], "wz_body": pose.w_body[2]})  # fmt: skip
    return rows


def _pose_json(p: Pose2D) -> Dict[str, float]:
    return {"x": p.x, "y": p.y, "yaw": p.yaw}


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys: Dict[str, List[float]] = {}
    for row in rows:
        lat = (row.get("debug") or {}).get("latency") or {}
        for k, v in lat.items():
            if isinstance(v, (int, float)):
                keys.setdefault(k, []).append(float(v))
    return {k: metrics.latency_stats(v) for k, v in keys.items()}


def run_episode(job: Job, ctx: RunContext) -> Dict[str, Any]:
    """Run a cell with infrastructure retries, or load its existing run.json."""
    rd = RunDir(ctx.root, job.method_id, job.token, job.repeat)
    if rd.is_complete():
        return rd.load()
    profile = fairness.parse_method_id(job.method_id)
    ctx.episode_set.check_method_allowed(profile.uses_truth)
    errors: List[str] = []
    while True:
        attempt = rd.begin_attempt()
        if attempt > ctx.bench.max_attempts:
            doc = _infra_doc(job, ctx, attempt - 1, errors or ["之前的尝试被中断"])
            rd.finish(doc)
            return doc
        try:
            doc = _attempt(job, ctx, rd, attempt)
        except FatalInfraError:
            rd.park_attempt(traceback.format_exc())
            raise
        except InfraError as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            LOG.warning("%s 基础设施失败(第 %d 次): %s", job.key, attempt, exc)
            rd.park_attempt(traceback.format_exc())
            continue
        doc["infra"] = {"attempts": attempt, "errors": errors}
        rd.finish(doc)
        return doc


def _infra_doc(job: Job, ctx: RunContext, attempts: int, errors: List[str]) -> Dict[str, Any]:
    rec = ctx.episode_set.records[job.token]
    return {
        "schema": RUN_SCHEMA_ID, "method_id": job.method_id,
        "episode_set": ctx.episode_set.name, "episode_token": job.token,
        "repeat_index": job.repeat, "status": "unscored_infra", "attempt": max(attempts, 1),
        "git": ctx.git, "server": {},
        "settings": ctx.bench.settings(scene=ctx.episode_set.payload["scene"]),
        "episode": _episode_json(ctx, rec), "timing": {}, "outcome": None, "metrics": None,
        "method_stats": {}, "usage": {"calls": 0, "billed_usd": 0.0, "list_usd": 0.0},
        "infra": {"attempts": attempts, "errors": errors},
    }  # fmt: skip


def _episode_json(ctx: RunContext, rec: EpisodeRecord) -> Dict[str, Any]:
    return {
        "scene": ctx.episode_set.payload["scene"], "route": rec.route,
        "route_kf_first": rec.route_kf_first, "route_kf_last": rec.route_kf_last,
        "start": _pose_json(rec.start),
        "goal": _pose_json(rec.goal), "shortest_m": rec.shortest_m, "straight_m": rec.straight_m,
        "detour": rec.detour_ratio, "tier": rec.tier, "same_room": rec.same_room,
        "start_room": rec.start_room, "goal_room": rec.goal_room,
        "waypoints": [list(p) for p in rec.waypoints],
        "episode_set_sha256": ctx.episode_set.manifest_sha256,
    }  # fmt: skip


def _attempt(job: Job, ctx: RunContext, rd: RunDir, attempt: int) -> Dict[str, Any]:
    es = ctx.episode_set
    rec = es.records[job.token]
    profile = fairness.parse_method_id(job.method_id)
    third = bool(ctx.want_third(job))
    rd.write_bytes("goal/goal.jpg", (es.root / rec.goal_rgb).read_bytes())
    rd.write_bytes("goal/goal.depth.png", (es.root / rec.goal_depth).read_bytes())
    wall_start = time.time()
    session = ctx.sim.open(es.payload["scene"], rec.start, third, rd.path("logs"))
    method: Optional[Method] = None
    try:
        env = SimEnv(session.link, ctx.bench.budget_s_for(es.payload["scene"]), want_third=third)
        try:
            obs = env.start(rec.start)
        except cp.CtrlError as exc:
            raise InfraError(f"reset 失败: {exc}") from exc
        rd.write_frames("ego", obs.frames)
        rd.write_depth(0.0, env.last_wire.depth)  # type: ignore[union-attr]
        rd.append_trace(_trace_rows(env.truth.samples))
        written = len(env.truth.samples)

        config, config_sha = method_config(job.method_id)
        map_handle, map_info, localizer = None, None, None
        if profile.map_variant is not None:
            if ctx.maps is None:
                raise InfraError("这个方法要地图,但没配 MapSource")
            map_handle, m2w, map_info = ctx.maps.get(rec.route, profile.map_variant)
            localizer = fairness.QuotaLocalizer(
                OracleLocalizer(
                    env.truth_pose, rec.goal, m2w, rng=_rng_for(ctx.bench.seed, job),
                    scene=es.payload["scene"], truth_z_now=lambda: env.truth_wire().position_xyz[2],
                    goal_floor=rec.goal_floor,
                ),
                profile,
            )
        usage: EpisodeUsage = ctx.guard.sink(job.method_id, job.token, job.repeat)
        method = ctx.make_method(job.method_id)
        if profile.uses_truth:
            from vnav_benchmark.methods.debug import DebugTruth

            method.attach_truth(  # type: ignore[attr-defined]
                DebugTruth(
                    grid=ctx.truth_grid(), truth_now=env.truth_pose, goal_world=rec.goal,
                    multifloor=ctx.truth_multifloor_grid(),
                    truth_z_now=lambda: env.truth_wire().position_xyz[2],
                )
            )
        mctx = MethodContext(
            profile=profile,
            config=config,
            map=map_handle,
            localizer=localizer,
            rng=_rng_for(ctx.bench.seed, job),
            log=logging.getLogger(f"vnav.method.{job.method_id}"),
            usage=usage,
            artifact_dir=rd.path("method"),
        )
        spec = es.spec(job.token, job.repeat, ctx.bench.budget_s_for(es.payload["scene"]))

        termination = ""
        declared = False
        act_walls: List[float] = []
        proximity = 0
        clamp_counts = {"vx": 0, "vy": 0, "wz": 0, "nonfinite": 0, "hold": 0}
        step_rows: List[Dict[str, Any]] = []
        fall = metrics.FallMonitor()
        reset_wall = 0.0
        try:
            t0 = time.perf_counter()
            method.reset(spec, obs, mctx)
            reset_wall = time.perf_counter() - t0
            while True:
                if localizer is not None:
                    localizer.on_frame(obs.step_index)
                near = metrics.is_proximity(obs.depth_m)
                proximity += int(near)
                t0 = time.perf_counter()
                action = method.act(obs)
                act_wall = time.perf_counter() - t0
                act_walls.append(act_wall)
                if not isinstance(action, Action):
                    raise TypeError(f"act 返回的不是 Action: {type(action).__name__}")
                applied, rep = fairness.clamp_action(action)
                for key, hit in (("vx", rep.clamped_vx), ("vy", rep.clamped_vy),
                                 ("wz", rep.clamped_wz), ("nonfinite", rep.nonfinite),
                                 ("hold", rep.hold_changed)):  # fmt: skip
                    clamp_counts[key] += int(hit)
                hold = 0.0 if action.done else env.clip_hold(applied.hold_s)
                row = _step_row(obs, action, applied, hold, rep, act_wall, near)
                step_rows.append(row)
                rd.append_step(row)
                if act_wall > ctx.bench.act_wall_limit_s:
                    raise _Abort("act_timeout")
                if usage.over_cap():
                    raise _Abort("cost_guard")
                if action.done:
                    declared = True
                    raise _Abort("done")
                try:
                    obs, reply = env.step(applied)
                except cp.CtrlError as exc:
                    raise InfraError(f"step 失败: {exc}") from exc
                rd.write_frames("ego", obs.frames)
                if third:
                    rd.write_frames("third", reply.third)
                rd.write_depth(obs.sim_time_s, reply.obs.depth)
                new = env.truth.samples[written:]
                written = len(env.truth.samples)
                rd.append_trace(_trace_rows(new))
                if any(fall.update(t, p.up_z) for t, p in new):
                    raise _Abort("fallen")
                if env.out_of_budget:
                    raise _Abort("timeout")
        except _Abort as stop:
            termination = stop.termination
        except InfraError:
            raise
        except Exception:
            termination = "method_error"
            rd.write_bytes("logs/method_error.txt", traceback.format_exc().encode())
            LOG.warning("%s 方法异常:\n%s", job.key, traceback.format_exc())

        trace = [metrics.TraceRow(**{k: r[k] for k in ("t_sim", "x", "y", "yaw", "up_z", "z")})
                 for r in _trace_rows(env.truth.samples)]  # fmt: skip
        outcome = metrics.evaluate_run(
            trace, rec.goal, rec.shortest_m, declared, termination,
            decisions=len(act_walls), act_wall_s=act_walls, proximity_events=proximity,
            goal_scene=es.payload["scene"], goal_floor=rec.goal_floor,
        )  # fmt: skip
        summary = getattr(method, "summary", None)
        method_summary = summary() if callable(summary) else None
    finally:
        if method is not None:
            try:
                method.close()
            except Exception:
                LOG.warning("%s method.close 出错:\n%s", job.key, traceback.format_exc())
        session.close()

    if profile.uses_truth:
        status = "excluded_debug"
    elif outcome.termination in ("act_timeout", "cost_guard"):
        status = "unscored_guard"
    else:
        status = "scored"
    out = outcome.to_dict()
    return {
        "schema": RUN_SCHEMA_ID,
        "method_id": job.method_id,
        "episode_set": es.name,
        "episode_token": job.token,
        "repeat_index": job.repeat,
        "status": status,
        "attempt": attempt,
        "git": ctx.git,
        "server": session.identity,
        "settings": {
            **ctx.bench.settings(scene=es.payload["scene"]),
            "method_config_sha256": config_sha,
            "map": map_info,
            "localizer": None
            if localizer is None
            else {"kind": "oracle", "sigma_xy_m": 0.0, "sigma_yaw_rad": 0.0},  # fmt: skip
            "third_person": third,
        },
        "episode": _episode_json(ctx, rec),
        "timing": {
            "wall_start": wall_start,
            "wall_end": time.time(),
            "sim_s": env.sim_time,
            "decisions": len(act_walls),
            "reset_wall_s": reset_wall,
            "act_wall_s": out["act_wall_s"],
            "compute_per_sim_s": out["compute_per_sim_s"],
            "method_latency": _latency_summary(step_rows),
        },
        "outcome": out,
        "metrics": {
            k: out[k]
            for k in ("success", "success_loose", "spl", "plr", "path_length_m", "shortest_path_m")
        },  # fmt: skip
        "method_stats": {
            "loc": None if localizer is None else localizer.stats(),
            "clamped": clamp_counts,
            "server_clamped": env.server_clamped,
            "summary": method_summary,
        },
        "usage": usage.summary(),
        "infra": {},
    }


def _step_row(
    obs: Observation, action: Action, applied: Action, hold: float, rep: Any, wall: float,
    near: bool,
) -> Dict[str, Any]:  # fmt: skip
    return {
        "i": obs.step_index,
        "t_sim": obs.sim_time_s,
        "budget_left_s": obs.budget_left_s,
        "act_wall_s": wall,
        "action": {
            "vx": action.vx,
            "vy": action.vy,
            "wz": action.wz,
            "hold_s": action.hold_s,
            "done": action.done,
        },  # fmt: skip
        "applied": {"vx": applied.vx, "vy": applied.vy, "wz": applied.wz, "hold_s": hold},
        "clamp": {
            "vx": rep.clamped_vx,
            "vy": rep.clamped_vy,
            "wz": rep.clamped_wz,
            "nonfinite": rep.nonfinite,
            "hold": rep.hold_changed,
        },  # fmt: skip
        "note": action.note,
        "debug": dict(action.debug),
        "odom": {"x": obs.odom.pose.x, "y": obs.odom.pose.y, "yaw": obs.odom.pose.yaw},
        "proximity": near,
    }


# ---- One campaign ----


Plan = Dict[str, Tuple[Tuple[str, ...], int]]
"""Episode tokens and repeat counts for each method."""

V1_PTGOAL_REPEATS = 3  # B0/M1 have no API cost: N routes, three repeats.


def uniform_plan(methods: Sequence[str], tokens: Sequence[str], repeats: int) -> Plan:
    return {m: (tuple(tokens), int(repeats)) for m in methods}


def v1_plan(es: EpisodeSet) -> Plan:
    """Return the approved v1 plan across all N routes for twelve configurations.

    Image methods (astra/qwen) run once; B0/M1 run three times without API cost.
    """
    every = tuple(es.tokens())
    return {
        mid: (every, V1_PTGOAL_REPEATS if mid.endswith("-ptgoal") else 1)
        for mid in fairness.V1_METHOD_IDS
    }


def job_order(
    methods: Sequence[str], tokens: Sequence[str], repeats: int, plan: Optional[Plan] = None
) -> List[Job]:
    """Order jobs by repeat, pair, then method; rotate which method runs first.

    When plan is supplied, include only method/pair/repeat cells in that plan.
    """
    plan = plan or uniform_plan(methods, tokens, repeats)
    max_repeats = max((plan[m][1] for m in methods), default=0)
    jobs = []
    for repeat in range(max_repeats):
        for i, token in enumerate(tokens):
            eligible = [m for m in methods if token in plan[m][0] and repeat < plan[m][1]]
            n = len(eligible)
            for k in range(n):
                jobs.append(Job(eligible[(i + k) % n], token, repeat))
    return jobs


class CampaignStopped(RuntimeError):
    pass


@dataclass
class Campaign:
    ctx: RunContext
    methods: Tuple[str, ...]
    tokens: Tuple[str, ...]
    repeats: int
    alert: Callable[[str], None] = lambda msg: LOG.error(msg)
    plan: Optional[Plan] = None  # Per-method plan; tokens set ordering.

    @property
    def root(self) -> Path:
        return self.ctx.root

    def header(self) -> Dict[str, Any]:
        es = self.ctx.episode_set
        return {
            "schema": CAMPAIGN_SCHEMA,
            "episode_set": es.name,
            "episode_set_sha256": es.manifest_sha256,
            "methods": list(self.methods),
            "tokens": list(self.tokens),
            "repeats": self.repeats,
            "plan": None
            if self.plan is None
            else {m: [list(t), r] for m, (t, r) in sorted(self.plan.items())},
            "bench": self.ctx.bench.settings(scene=self.ctx.episode_set.payload["scene"]),
        }

    def init(self) -> None:
        """Write campaign.json initially; require identical parameters on resume."""
        self.ctx.episode_set.check_usable()
        path = self.root / "campaign.json"
        header = self.header()
        if path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            if {k: old.get(k) for k in header} != json.loads(json.dumps(header)):
                raise CampaignStopped(f"{path} 的参数和这次不同:续跑不许改方法 / 起终点 / 遍数")
            return
        for mid in self.methods:
            prof = fairness.parse_method_id(mid)
            self.ctx.episode_set.check_method_allowed(prof.uses_truth)
        write_json_atomic(path, {**header, "created": time.time(), "git": self.ctx.git})

    def jobs(self) -> List[Job]:
        return job_order(self.methods, self.tokens, self.repeats, self.plan)

    def run(self, max_jobs: Optional[int] = None) -> Dict[str, Any]:
        """Run unfinished cells under an exclusive .campaign.lock for this campaign."""
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / LOCK_NAME, "a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CampaignStopped(f"另一个进程正在跑 {self.root},不能同时跑同一批") from exc
            return self._run_locked(max_jobs)

    def _run_locked(self, max_jobs: Optional[int]) -> Dict[str, Any]:
        self.init()
        streak = 0
        ran = 0
        blocked: Dict[str, str] = {}
        try:
            for job in self.jobs():
                rd = RunDir(self.root, job.method_id, job.token, job.repeat)
                if rd.is_complete():
                    continue
                if max_jobs is not None and ran >= max_jobs:
                    break
                ok, why = self.ctx.guard.can_start(provider_of(job.method_id))
                if not ok:
                    blocked[job.key] = why
                    continue
                try:
                    doc = run_episode(job, self.ctx)
                except FatalInfraError as exc:
                    self.alert(f"[vnav] 跑批停:致命基础设施错误 {exc}({job.key})")
                    raise CampaignStopped(str(exc)) from exc
                ran += 1
                streak = streak + 1 if doc["status"] == "unscored_infra" else 0
                self.write_coverage(blocked)
                if streak >= self.ctx.bench.consecutive_infra_break:
                    self.alert(f"[vnav] 跑批熔断:连续 {streak} 格基础设施失败(最后 {job.key})")
                    raise CampaignStopped("连续基础设施失败")
        finally:
            cov = self.write_coverage(blocked)
        return cov

    def write_coverage(self, blocked: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        cov = coverage(self.root, self.methods, self.tokens, self.repeats, blocked or {}, self.plan)
        write_json_atomic(self.root / "coverage.json", cov)
        return cov


def coverage(
    root: Path,
    methods: Sequence[str],
    tokens: Sequence[str],
    repeats: int,
    blocked: Optional[Mapping[str, str]] = None,
    plan: Optional[Plan] = None,
) -> Dict[str, Any]:
    """Return cell states, method counts and fully scored planned paired cells."""
    cells = []
    counts: Dict[str, Dict[str, int]] = {m: {} for m in methods}
    scored: Dict[Tuple[str, int], set] = {}
    planned: Dict[Tuple[str, int], set] = {}
    for job in job_order(methods, tokens, repeats, plan):
        planned.setdefault((job.token, job.repeat), set()).add(job.method_id)
        rd = RunDir(root, job.method_id, job.token, job.repeat)
        if rd.is_complete():
            status = rd.load()["status"]
        elif blocked and job.key in blocked:
            status = "budget_blocked"
        else:
            status = "pending"
        cells.append({"method_id": job.method_id, "token": job.token, "repeat": job.repeat,
                      "status": status})  # fmt: skip
        counts[job.method_id][status] = counts[job.method_id].get(status, 0) + 1
        if status == "scored":
            scored.setdefault((job.token, job.repeat), set()).add(job.method_id)
    paired = sorted(
        f"{t}/r{r}" for (t, r), ms in scored.items() if ms >= planned[(t, r)] and len(ms) > 1
    )
    total = len(cells)
    done = sum(c["status"] not in ("pending", "budget_blocked") for c in cells)
    return {
        "schema": COVERAGE_SCHEMA,
        "total": total,
        "done": done,
        "by_method": counts,
        "paired_cells": paired,
        "blocked": dict(blocked or {}),
        "cells": cells,
    }
