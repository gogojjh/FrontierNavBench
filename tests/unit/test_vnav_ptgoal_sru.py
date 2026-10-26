"""Run full B0/M1 episodes with KinematicBackend, SimEnv, ideal localization, quotas and clamps.

Replace the point-goal controller with a kinematic fake; no SRU sidecar starts.
Real SRU parity is covered by test_vnav_sru.py and integration tests.
"""

import logging
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import yaml
from vnav_t_fixtures import CAMERA_FORWARD_M, L_CORRIDOR, make_map, unicycle_to

from vnav_benchmark import fairness
from vnav_benchmark import mapview as mv
from vnav_benchmark.interfaces import EpisodeSpec, MethodContext, OdomState, Pose2D
from vnav_benchmark.localizer import OracleLocalizer
from vnav_benchmark.methods import make_method
from vnav_benchmark.methods import ptgoal_sru as pg
from vnav_benchmark.methods.base_topo import FINAL_POS_TOL_M
from vnav_benchmark.methods.sru.client import PointGoalCommand
from vnav_benchmark.sim.backend import KinematicBackend
from vnav_benchmark.sim.ctrl_client import InProcessClient
from vnav_benchmark.sim.ctrl_server import CtrlServer
from vnav_benchmark.sim_env import SimEnv

REPO = Path(__file__).resolve().parents[2]
IDENTITY16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
CELL_M = 0.1
ORIGIN = (-2.0, -2.0)
GOAL = (4.1, 4.3, math.pi / 2 + 1.0)  # Goal yaw differs by 1 rad from the approach direction,
# requiring in-place alignment
START = (0.2, 0.1, math.pi)


def l_corridor_occupancy() -> np.ndarray:
    """A 10x10 m grid with an L corridor: [-1,5]x[-1,1] union [3,5]x[-1,5.5]."""
    occ = np.ones((100, 100), dtype=bool)
    ys = ORIGIN[1] + (np.arange(100) + 0.5) * CELL_M
    xs = ORIGIN[0] + (np.arange(100) + 0.5) * CELL_M
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    free = ((xx > -1) & (xx < 5) & (yy > -1) & (yy < 1)) | (
        (xx > 3) & (xx < 5) & (yy > -1) & (yy < 5.5)
    )
    occ[free] = False
    return occ


class UnicycleController:
    """Kinematic point controller: turn then move; report arrival within 0.25 m like SRU."""

    def __init__(self) -> None:
        self.resets = 0
        self.closed = False
        self.targets: List[Tuple[float, float]] = []

    def reset(self) -> None:
        self.resets += 1

    def step(
        self, depth_m: np.ndarray, odom: OdomState, target_odom_xy: Tuple[float, float]
    ) -> PointGoalCommand:
        self.targets.append(target_odom_xy)
        pose = odom.pose
        if math.hypot(target_odom_xy[0] - pose.x, target_odom_xy[1] - pose.y) <= 0.25:
            return PointGoalCommand(0.0, 0.0, 0.0, True, 0.001)
        vx, vy, wz = unicycle_to((pose.x, pose.y, pose.yaw), target_odom_xy)
        return PointGoalCommand(vx, vy, wz, False, 0.001)

    def close(self) -> None:
        self.closed = True


class _Sink:
    def add(self, record: object) -> None:
        raise AssertionError("B0 / M1 不该有付费调用")

    def episode_billed_usd(self) -> float:
        return 0.0


def run_episode(
    tmp: Path,
    method_id: str,
    trav_edges: Sequence[Tuple[int, int]] = (),
    budget_s: float = 120.0,
    poses: Sequence[Tuple[float, float, float]] = L_CORRIDOR,
) -> Tuple[SimEnv, pg.PtGoalSruMethod, fairness.QuotaLocalizer, UnicycleController, bool]:
    """Minimal runner-equivalent loop: on_frame, act, clamp, step for every observation."""
    src = make_map(tmp / "src", poses=poses, trav_edges=trav_edges)
    mv.build_sanitized_view(src, tmp / "view")
    profile = fairness.parse_method_id(method_id)
    handle = mv.SanitizedMap(tmp / "view", "m", profile.map_variant, "trav", CAMERA_FORWARD_M)
    backend = KinematicBackend(occupancy=l_corridor_occupancy(), origin_xy=ORIGIN, cell_m=CELL_M)
    client = InProcessClient(CtrlServer(backend, "inproc://t3"))
    client.hello()
    env = SimEnv(client, budget_s)
    obs = env.start(Pose2D(*START))
    oracle = OracleLocalizer(env.truth_pose, Pose2D(*GOAL), IDENTITY16)
    loc = fairness.QuotaLocalizer(oracle, profile)
    config = yaml.safe_load((REPO / "configs/methods" / f"{method_id}.yaml").read_text())
    ctx = MethodContext(
        profile=profile, config=config, map=handle, localizer=loc,
        rng=np.random.default_rng(0), log=logging.getLogger("t3"), usage=_Sink(),
        artifact_dir=tmp / "method",
    )
    controller = UnicycleController()
    method = pg.PtGoalSruMethod(method_id, controller_factory=lambda _log: controller)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    depth = np.zeros((48, 64), np.float32)
    spec = EpisodeSpec("tok", 0, "env0", img, b"g", depth, (1.0,), budget_s)
    loc.on_frame(obs.step_index)
    method.reset(spec, obs, ctx)
    done = False
    while not env.out_of_budget:
        loc.on_frame(obs.step_index)
        action, _ = fairness.clamp_action(method.act(obs))
        if action.done:
            done = True
            break
        obs, _ = env.step(action)
    method.close()
    return env, method, loc, controller, done


@pytest.mark.parametrize("method_id", ["dense-contloc-ptgoal", "sparse-contloc-ptgoal"])
def test_episode_reaches_and_aligns(tmp_path: Path, method_id: str) -> None:
    env, method, loc, controller, done = run_episode(tmp_path, method_id)
    assert done
    truth = env.truth_pose()
    assert math.hypot(truth.x - GOAL[0], truth.y - GOAL[1]) <= FINAL_POS_TOL_M
    assert abs(math.remainder(truth.yaw - GOAL[2], 2 * math.pi)) <= math.radians(30)
    assert loc.stats()["used"] == {"goal": 1, "self": 1}  # Continuous localization propagates
    # odometry
    assert controller.resets == 1 and controller.closed
    summary = method.summary()
    assert summary["termination"] == "done" and summary["align_decisions"] >= 1
    assert summary["subgoal_advances"] >= 5 and summary["replans"] == 0
    out = tmp_path / "method"
    assert {"route.json", "steps.jsonl", "summary.json"} <= {p.name for p in out.iterdir()}
    assert len((out / "steps.jsonl").read_text().splitlines()) == summary["decisions"]


def test_shortcut_through_wall_is_blocked_and_replanned(tmp_path: Path) -> None:
    """A shorter 1->9->8 shortcut points into a wall at keyframe 9.

    Direct perpendicular contact cannot slide; after 10 s without progress,
    block edge 1-9, switch to the chain and still reach the goal.
    """
    poses = tuple(L_CORRIDOR) + ((1.0, 2.0, math.pi / 2),)
    env, method, _, _, done = run_episode(
        tmp_path, "dense-contloc-ptgoal", trav_edges=[(1, 9), (9, 8)], poses=poses
    )
    assert done
    summary = method.summary()
    assert summary["replans"] >= 1 and summary["blocked_edges"] >= 1 and summary["degraded"] == 0


def test_budget_runs_out_without_done(tmp_path: Path) -> None:
    env, method, _, controller, done = run_episode(tmp_path, "dense-contloc-ptgoal", budget_s=3.0)
    assert not done and env.out_of_budget and controller.closed
    assert method.summary()["termination"] is None


def test_registry_and_config() -> None:
    for method_id in ("dense-contloc-ptgoal", "sparse-contloc-ptgoal"):
        method = make_method(method_id)  # Construct only; do not start the sidecar
        assert isinstance(method, pg.PtGoalSruMethod) and method.method_id == method_id
        config = yaml.safe_load((REPO / "configs/methods" / f"{method_id}.yaml").read_text())
        assert config["method_id"] == method_id and config["controller"] in pg.CONTROLLERS
    with pytest.raises(ValueError):
        pg.PtGoalSruMethod("dense-initloc-imggoal-astra")


def test_method_files_pass_leakage_check() -> None:
    for path in (Path(pg.__file__), Path(pg.__file__).with_name("base_topo.py")):
        assert fairness.leakage_violations(path.read_text(), path.name) == []
