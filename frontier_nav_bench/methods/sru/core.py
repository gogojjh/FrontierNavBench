"""Pure-numpy SRU preprocessing/postprocessing shared by sidecar and parity checks.

SRU is a learned local controller receiving depth, body state and a target point
and returning body-frame velocity. Import upstream sru_nav_go2.model.LearningModel
for ONNX inference without ROS. Approximately 40 preprocessing/postprocessing lines
are copied from the ROS-bound node class, with source line references below.

The ONNX model wrapper, constants and NumPy helpers are retained under
third_party/SRU-Odin. SIM_PARAMS preserves the audited simulation launch values;
the original ROS node and launch configuration remain on the development branch.

The sole behavioral difference is inference scheduling: the node uses odometry
timestamps at 5 Hz, whereas callers here invoke step every 0.2 simulation seconds
(one hold), equivalent to accepting each scheduling check.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class SruParams:
    """Node parameters after simulation launch overrides; tests compare the launch/YAML sources."""

    min_depth_m: float  # Treat shallower depths as missing
    max_depth_m: float  # Treat deeper depths as missing
    policy_scale: Vec3  # Scale tanh network outputs into forward m/s, lateral m/s and yaw rad/s
    lateral_velocity_scale: float  # Additional lateral velocity multiplier
    low_pass_filter_coef: Vec3  # First-order low-pass: new = a * current + (1-a) * previous
    arrive_goal_threshold_m: float  # At or below this planar goal distance, zero velocity and clear
    # the target
    control_frequency_hz: float  # Node cadence, matched here with a 0.2 s hold


SIM_PARAMS = SruParams(
    min_depth_m=0.0,  # sim_sru_nav_go2.launch sets min_depth=0.0, preserving near depths in
    # simulation
    max_depth_m=10.0,  # sru_nav.yaml max_depth
    policy_scale=(0.8, 0.3, 0.7),  # Approved policy_scale override from sim_sru_nav_go2.launch
    lateral_velocity_scale=1.0,  # sru_nav.yaml
    low_pass_filter_coef=(0.5, 0.5, 0.5),  # sru_nav.yaml
    arrive_goal_threshold_m=0.25,  # sru_nav.yaml arrive_goal_threshold
    control_frequency_hz=5.0,  # sru_nav.yaml control_frequency
)
"""Simulation uses use_sim=false: world-frame network velocities are converted to body frame.

Do not reset the LSTM on target jumps.
"""

GRAVITY_MAGNITUDE = 9.81  # Source: sru_nav_go2/constants.py GRAVITY_MAGNITUDE


class SruModel(Protocol):
    """Subset of the sru_nav_go2.model.LearningModel interface used here."""

    def predict(
        self,
        linear_vel: Sequence[float],
        angular_vel: Sequence[float],
        gravity_vector: Sequence[float],
        last_action: Sequence[float],
        target_pos_w: Sequence[float],
        robot_pos_w: Sequence[float],
        robot_orientation_w: Sequence[float],
        depth_image: np.ndarray,
        is_reset: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def reset_hidden_state(self) -> None: ...


# ---------------------------------------------------------------- Quaternions, node order wxyz


def _rotation_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Normalized body-to-world rotation matrix, equivalent to node scipy R.from_quat(xyzw)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def world_to_body(vec_w: Sequence[float], quat_wxyz: Sequence[float]) -> list:
    """Rotate world vectors into the body frame; node convert_vel_frame lines 292-298."""
    return (_rotation_wxyz(quat_wxyz).T @ np.asarray(vec_w, dtype=float)).tolist()


def projected_gravity(quat_wxyz: Sequence[float]) -> list:
    """Unit gravity vector in the body frame; node projected_gravity_vector lines 300-309."""
    gravity = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
    proj = _rotation_wxyz(quat_wxyz).T @ gravity
    proj = proj / (np.linalg.norm(proj) + 1e-6)
    return proj.tolist()


def preprocess_depth(depth_m: np.ndarray, params: SruParams) -> np.ndarray:
    """Clean depth as in node depth_callback lines 187-191.

    Replace NaN/-inf with zero and +inf with twice max_depth, then zero values
    outside [min_depth, max_depth]. The network treats zero as missing data.
    """
    depth = np.nan_to_num(
        depth_m, nan=0.0, posinf=params.max_depth_m * 2.0, neginf=0.0
    ).astype(np.float32, copy=False)
    depth = depth.copy()
    depth[depth > params.max_depth_m] = 0.0
    depth[depth < params.min_depth_m] = 0.0
    return depth


@dataclass(frozen=True)
class SruStep:
    cmd: Vec3  # Low-pass-filtered body velocity sent to the robot
    arrived: bool  # Whether this step zeroed velocity because the goal was within arrival tolerance
    raw_action: Optional[Vec3]  # Raw network output before tanh; None on arrival
    vae_ms: float  # Depth-encoding latency
    policy_ms: float  # Policy-network latency


class SruCore:
    """Episode SRU state: model LSTM state, previous action and previous low-pass output."""

    def __init__(self, model: SruModel, params: SruParams = SIM_PARAMS) -> None:
        self._model = model
        self._params = params
        self.reset()

    def reset(self) -> None:
        """Reset LSTM, previous action and low-pass state at episode start; the node never resets
        LSTM.
        """
        self._model.reset_hidden_state()
        self._last_action = [0.0, 0.0, 0.0]  # Node _reset_last_action, lines 385-387
        self._prev_cmd = np.zeros(3)

    def step(
        self,
        depth_m: np.ndarray,
        robot_pos_w: Sequence[float],
        robot_quat_wxyz: Sequence[float],
        linear_vel_b: Sequence[float],
        angular_vel_b: Sequence[float],
        target_pos_w: Sequence[float],
    ) -> SruStep:
        """Infer one step with poses/target in a shared world frame (episode odometry here) and body
        velocities.
        """
        params = self._params
        target = [float(v) for v in target_pos_w]
        if abs(target[2]) < 1e-3:  # Node target_position_callback lines 355-358: use body z when
        # target z is near zero
            target[2] = float(robot_pos_w[2])
        # Node _check_goal_reached lines 336-343 and generate_cmd_vel arrival branch lines 211-222
        dist = np.linalg.norm(np.array(target[:2]) - np.array(robot_pos_w[:2]))
        if dist <= params.arrive_goal_threshold_m:
            self._last_action = [0.0, 0.0, 0.0]
            self._prev_cmd = np.zeros(3)
            return SruStep((0.0, 0.0, 0.0), True, None, 0.0, 0.0)
        # Node generate_cmd_vel inference branch, lines 225-267
        cmd, action, _ = self._model.predict(
            list(linear_vel_b),
            list(angular_vel_b),
            projected_gravity(robot_quat_wxyz),
            self._last_action,
            target,
            list(robot_pos_w),
            list(robot_quat_wxyz),
            preprocess_depth(depth_m, params),
            False,
        )
        self._last_action = action.tolist()
        model_cmd = np.array(
            [cmd[0].item(), cmd[1].item() * params.lateral_velocity_scale, cmd[2].item()]
        )
        coef = np.array(params.low_pass_filter_coef)
        filtered = coef * model_cmd + (1 - coef) * self._prev_cmd
        self._prev_cmd = filtered
        vae_ms = float(getattr(self._model, "last_vae_ms", 0.0))
        policy_ms = float(getattr(self._model, "last_policy_ms", 0.0))
        return SruStep(
            (float(filtered[0]), float(filtered[1]), float(filtered[2])),
            False,
            (float(action[0]), float(action[1]), float(action[2])),
            vae_ms,
            policy_ms,
        )
