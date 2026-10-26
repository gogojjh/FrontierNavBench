"""Go2 MoE-CTS inference and Isaac joint control.

Policy observations contain ten 45-value frames at 50 Hz. Commands expire after
0.5 s; zero-speed walking lasts another 2 s before standing at gains 60/5.
MoE-CTS policy conventions derive from go2_rl_robotlab revision 28b4516, Apache-2.0;
see assets/LICENSES/Apache-2.0-go2-policy.txt for the retained license."""

# Adapted from nav_loco/nav_loco/go2/mcts_policy.py.
# Source SHA256: ca26042f6c952adda2770681baebf6cd595b20c907cd8c81cb32cd7377a74ab1.
# Adapted from nav_sim_isaac/nav_sim_isaac/go2_walk.py.
# Source SHA256: ab3e5b361952f8b615636c692b7f59dc29f26a99511515f7ed57b4b87940df07.
# Adapted from nav_sim_isaac/nav_sim_isaac/unitree_policy.py.
# Source SHA256: 25e933d18df651f1140716d36f7e4d6c5f4af41594db7c329d84403d029edfb4.
# Adapted from nav_sim_isaac/nav_sim_isaac/go2_rig.py.
# Source SHA256: 43d0351f6434fb09f85b9289d98bc909b834d5c98acd9f3faf18e48204c524d5.
# Adapted from nav_sim_isaac/scripts/sim_sensor_publisher.py.
# Source SHA256: 18ea3a70d64720fb7ddd2b39a5937347b89bdc4bdb855cd369fe86195509d2da.
# Copyright (c) 2026 Jianhao Jiao. MIT license; see LICENSE.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from vnav_benchmark.paths import asset_root

JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
DEFAULT_JOINT_POS = np.array(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=float
)
ACTION_SCALE = 0.25
ACTION_CLIP = 100.0
OBS_CLIP = 100.0
OBS_TERMS: tuple[tuple[str, int, float], ...] = (
    ("base_ang_vel", 3, 0.25),
    ("projected_gravity", 3, 1.0),
    ("velocity_commands", 3, 1.0),
    ("joint_pos_rel", 12, 1.0),
    ("joint_vel_rel", 12, 0.05),
    ("last_action", 12, 1.0),
)
FRAME_DIM = sum(size for (_, size, _) in OBS_TERMS)
HISTORY_LENGTH = 10
POLICY_HZ = 50.0
KP = 25.0
KD = 0.5
COMMAND_RANGE = ((-1.0, 1.0), (-1.0, 1.0), (-1.5, 1.5))
POLICY_DIR = asset_root() / "weights/go2"
DEFAULT_POLICY = POLICY_DIR / "go2_moe_cts_176k_0.6984.pt"
_GRAVITY_DIR = np.array([0.0, 0.0, -1.0])


class MctsPolicyError(ValueError):
    pass


class JointMap:
    """Map Isaac joint order to the frozen leg-major policy order."""

    def __init__(self, sim_joint_names: Sequence[str]) -> None:
        names = tuple(sim_joint_names)
        if sorted(names) != sorted(JOINT_NAMES):
            raise MctsPolicyError(
                f"Joint names differ from MoE-CTS: {list(names)}; expected {list(JOINT_NAMES)}"
            )
        self.sim_names = names
        self.policy_from_sim = np.array([names.index(n) for n in JOINT_NAMES], dtype=int)
        "``policy_vec = sim_vec[policy_from_sim]``。"
        self.sim_from_policy = np.array([JOINT_NAMES.index(n) for n in names], dtype=int)
        "``sim_vec = policy_vec[sim_from_policy]``。"

    def to_policy(self, sim_values: Sequence[float]) -> np.ndarray:
        return np.asarray(sim_values, dtype=float)[self.policy_from_sim]

    def to_sim(self, policy_values: Sequence[float]) -> np.ndarray:
        return np.asarray(policy_values, dtype=float)[self.sim_from_policy]


def quat_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    (w, x, y, z) = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_frame(
    ang_vel_world: Sequence[float],
    orientation_wxyz: Sequence[float],
    command: Sequence[float],
    joint_pos: Sequence[float],
    joint_vel: Sequence[float],
    last_action: Sequence[float],
) -> np.ndarray:
    """Build one 45-value observation with the original scales and clipping."""
    rotation = quat_wxyz_to_matrix(orientation_wxyz)
    raw = (
        rotation.T @ np.asarray(ang_vel_world, dtype=float),
        rotation.T @ _GRAVITY_DIR,
        np.asarray(command, dtype=float),
        np.asarray(joint_pos, dtype=float) - DEFAULT_JOINT_POS,
        np.asarray(joint_vel, dtype=float),
        np.asarray(last_action, dtype=float),
    )
    parts = []
    for values, (name, size, scale) in zip(raw, OBS_TERMS):
        if values.shape != (size,):
            raise MctsPolicyError(f"观测项 {name} 应是 {size} 个数,拿到 {values.shape}")
        parts.append(np.clip(values, -OBS_CLIP, OBS_CLIP) * scale)
    return np.concatenate(parts).astype(np.float32)


def action_to_targets(action: Sequence[float]) -> np.ndarray:
    """Convert clipped policy actions to absolute joint position targets."""
    target = DEFAULT_JOINT_POS + ACTION_SCALE * np.asarray(action, dtype=float)
    return np.clip(target, -ACTION_CLIP, ACTION_CLIP)


class MctsPolicy:
    """CPU TorchScript inference with ten chronological observation frames."""

    def __init__(self, path: Path = DEFAULT_POLICY) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 MoE-CTS 策略文件 {path}")
        import torch

        self._torch = torch
        self._module = torch.jit.load(str(path), map_location="cpu").eval()
        self.path = path
        single = int(getattr(self._module, "num_single_obs", -1))
        history = int(getattr(self._module, "history_len", -1))
        if (single, history) != (FRAME_DIM, HISTORY_LENGTH):
            raise MctsPolicyError(
                f"{path}: observation/history {single}/{history}; "
                f"expected {FRAME_DIM}/{HISTORY_LENGTH}"
            )

    def reset(self) -> None:
        self._module.reset()

    def __call__(self, frame: Sequence[float]) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32)
        if x.shape != (FRAME_DIM,):
            raise MctsPolicyError(f"策略要 {FRAME_DIM} 个数的一帧,拿到 {x.shape}")
        with self._torch.no_grad():
            out = self._module(self._torch.from_numpy(x)[None])
        return out[0].numpy().astype(float)

    def prime(self, frame: Sequence[float]) -> np.ndarray:
        """Fill all ten history slots with the first observation before inference."""
        self.reset()
        action = None
        for _ in range(HISTORY_LENGTH):
            action = self(frame)
        return action


def clamp_command(
    command: Sequence[float], limits: Sequence[tuple[float, float]]
) -> tuple[float, float, float]:
    return tuple(min(high, max(low, float(value))) for (value, (low, high)) in zip(command, limits))


EFFORT_LIMITS_NM = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}
MAX_DEPENETRATION_MPS = 1.0
CMD_TIMEOUT_S = 0.5
SETTLE_AFTER_CMD_S = 2.0
MODE_STAND = "stand"
MODE_WALK = "walk"


def effort_limits(joint_names: Sequence[str]) -> np.ndarray:
    """Return Go2 URDF torque limits in the requested joint order."""
    limits = []
    for name in joint_names:
        kind = next((k for k in EFFORT_LIMITS_NM if name.endswith(f"_{k}_joint")), None)
        if kind is None:
            raise ValueError(f"不是 Go2 的关节名:{name!r}(应当以 _hip/_thigh/_calf_joint 结尾)")
        limits.append(EFFORT_LIMITS_NM[kind])
    return np.array(limits, dtype=float)


@dataclass
class GaitState:
    mode: str
    command: tuple[float, float, float]


class GaitScheduler:
    """Switch between standing and walking using simulation-time command expiry."""

    def __init__(self) -> None:
        self._last_cmd_at: float | None = None
        self._last_cmd = (0.0, 0.0, 0.0)
        self._walking = False

    def on_command(self, now: float, command: tuple[float, float, float]) -> None:
        self._last_cmd_at = now
        self._last_cmd = command
        self._walking = True

    def update(self, now: float) -> GaitState:
        age = None if self._last_cmd_at is None else now - self._last_cmd_at
        fresh = age is not None and age <= CMD_TIMEOUT_S
        if fresh:
            return GaitState(MODE_WALK, self._last_cmd)
        if not self._walking:
            return GaitState(MODE_STAND, (0.0, 0.0, 0.0))
        if age is not None and age <= CMD_TIMEOUT_S + SETTLE_AFTER_CMD_S:
            return GaitState(MODE_WALK, (0.0, 0.0, 0.0))
        self._walking = False
        return GaitState(MODE_STAND, (0.0, 0.0, 0.0))


STAND_KP = 60.0
STAND_KD = 5.0
SPAWN_CLEAR_M = 0.6
STAND_SPAWN_GAP_M = 0.008
SETTLE_S = 1.5
GROUND_HEADROOM_M = 0.8
UPWARD_NORMAL_MIN = 0.7
(_HIP, _THIGH, _CALF) = ("_hip_joint", "_thigh_joint", "_calf_joint")


class StanceError(ValueError):
    pass


def pick_ground(hits: Sequence[tuple[float, float]], near_z: float | None = None) -> float | None:
    """Choose an upward surface with standing headroom, optionally near a floor."""
    surfaces = sorted({round(z, 3) for (z, _) in hits})
    candidates = []
    for z, normal_z in hits:
        if normal_z < UPWARD_NORMAL_MIN:
            continue
        above = [s for s in surfaces if s > z + 0.01]
        headroom = min(above) - z if above else float("inf")
        if headroom >= GROUND_HEADROOM_M:
            candidates.append(z)
    if not candidates:
        return None
    if near_z is not None:
        return min(candidates, key=lambda z: abs(z - near_z))
    return min(candidates)


def stance_angle(joint_name: str) -> float:
    if len(joint_name) < 2 or joint_name[0] not in "FR" or joint_name[1] not in "LR":
        raise StanceError(f"不是 Go2 的关节名:{joint_name!r}(应当像 FL_hip_joint)")
    if joint_name.endswith(_HIP):
        return 0.1 if joint_name[1] == "L" else -0.1
    if joint_name.endswith(_THIGH):
        return 0.8 if joint_name[0] == "F" else 1.0
    if joint_name.endswith(_CALF):
        return -1.5
    raise StanceError(f"不是 Go2 的关节名:{joint_name!r}(应当以 _hip/_thigh/_calf_joint 结尾)")


def default_stance(joint_names: Sequence[str]) -> list[float]:
    """Return standard Go2 standing angles in Isaac joint order."""
    return [stance_angle(name) for name in joint_names]


class Go2Driver:
    """Apply standing targets or 50 Hz policy targets before each physics tick."""

    def __init__(self, robot, stance, policy_path: Path, physics_hz: float, np) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self._np = np
        self._action_type = ArticulationAction
        self._robot = robot
        self._controller = robot.get_articulation_controller()
        self._stance = stance
        self._count = len(stance)
        self._joints = JointMap(list(robot.dof_names))
        self._decimation = max(1, round(physics_hz / POLICY_HZ))
        self._scheduler = GaitScheduler()
        self._mode = MODE_STAND
        self._last_action = np.zeros(self._count)
        self._primed = False
        self._target = stance.copy()
        self.command_range = COMMAND_RANGE
        limits = effort_limits(list(robot.dof_names))
        robot._articulation_view.set_max_efforts(np.asarray(limits, dtype=float)[None])
        self._set_gains(STAND_KP, STAND_KD)
        self._policy = MctsPolicy(Path(policy_path))
        stance_gap = float(np.max(np.abs(self._joints.to_policy(stance) - DEFAULT_JOINT_POS)))
        ranges = " ".join(f"[{lo:+.1f},{hi:+.1f}]" for (lo, hi) in self.command_range)
        print(
            f"MoE-CTS {policy_path}; decimation={self._decimation}, "
            f"ranges={ranges}, stance gap={stance_gap:.3f} rad",
            flush=True,
        )

    @property
    def can_walk(self) -> bool:
        return True

    def on_command(self, now: float, command: tuple[float, float, float]) -> None:
        self._scheduler.on_command(now, command)

    def step(self, now: float, step_index: int) -> None:
        np = self._np
        state = self._scheduler.update(now)
        if state.mode != self._mode:
            if state.mode == MODE_WALK:
                self._set_gains(KP, KD)
                self._last_action = np.zeros(self._count)
                self._primed = False
            else:
                self._set_gains(STAND_KP, STAND_KD)
            self._mode = state.mode
        if state.mode == MODE_WALK:
            if step_index % self._decimation == 0:
                self._target = self._run_policy(state.command)
        else:
            self._target = self._stance
        self._controller.apply_action(self._action_type(joint_positions=self._target))

    def _run_policy(self, command):
        robot = self._robot
        (_, orientation) = robot.get_world_pose()
        frame = build_frame(
            robot.get_angular_velocity(),
            orientation,
            command,
            self._joints.to_policy(robot.get_joint_positions()),
            self._joints.to_policy(robot.get_joint_velocities()),
            self._last_action,
        )
        if self._primed:
            self._last_action = self._policy(frame)
        else:
            self._last_action = self._policy.prime(frame)
            self._primed = True
        return self._joints.to_sim(action_to_targets(self._last_action))

    def _set_gains(self, kp, kd) -> None:
        np = self._np
        self._controller.set_gains(
            kps=np.broadcast_to(np.asarray(kp, dtype=float), (self._count,)).copy(),
            kds=np.broadcast_to(np.asarray(kd, dtype=float), (self._count,)).copy(),
        )
