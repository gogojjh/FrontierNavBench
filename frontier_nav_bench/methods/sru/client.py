"""Point-goal controller protocol and SRU sidecar client.

B0/M1 pass depth, odometry and an odometry-frame target to PointGoalController,
which returns body-frame velocity. SruSidecarController starts a fresh process
in the SRU environment per episode and closes it afterward (plan section 6.8).
Future ROS implementations can use the same protocol.
"""

import os
import secrets
import signal
import subprocess
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from frontier_nav_bench import paths, pins
from frontier_nav_bench.interfaces import InfraError, OdomState
from frontier_nav_bench.methods.sru.sidecar import ENV_AUTHKEY, ENV_PORT, READY_TAG

REPO_ROOT = Path(__file__).resolve().parents[3]
WS_ROOT = REPO_ROOT
SRU_PYTHON = Path(paths.setting(
    "VNAV_SRU_PYTHON", str(Path.home() / "miniconda3/envs/vnav-sru/bin/python"),
))
SRU_SRC = WS_ROOT / "third_party/SRU-Odin/Deployment/src"  # Directory containing the sru_nav_go2
# package
SIDECAR_PORT = 9982  # Allocated sidecar port; 9983 is the fallback in the README port table
STARTUP_TIMEOUT_S = 60.0  # Startup/model-load wall timeout; measured Jetson CPU startup takes a few
# seconds
STEP_TIMEOUT_S = 10.0  # Per-step response wall timeout; normal inference is below 0.2 s
SHUTDOWN_TIMEOUT_S = 5.0
POLL_S = 0.05


@dataclass(frozen=True)
class PointGoalCommand:
    vx: float  # Body-frame velocity, m/s
    vy: float
    wz: float  # rad/s
    arrived: bool  # Controller arrival flag; SRU sends zero velocity within 0.25 m
    controller_s: float  # Round-trip wall seconds, recorded in Action.debug["latency"]


class PointGoalController(Protocol):
    def reset(self) -> None: ...

    def step(
        self, depth_m: np.ndarray, odom: OdomState, target_odom_xy: Tuple[float, float]
    ) -> PointGoalCommand: ...

    def close(self) -> None: ...


def odom_inputs(odom: OdomState) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Convert odometry to SRU position, node-order wxyz quaternion and body linear/angular
    velocity.
    """
    qx, qy, qz, qw = odom.quat_xyzw
    return list(odom.position_xyz), [qw, qx, qy, qz], list(odom.v_body), list(odom.w_body)


class SruSidecarController:
    """Start, handshake, synchronously reset/step and close; connection/timeouts raise InfraError.
    """

    def __init__(
        self,
        log_path: Path,
        command: Optional[Sequence[str]] = None,
        port: int = SIDECAR_PORT,
        verify_pins: bool = True,
    ) -> None:
        if verify_pins:
            pins.require_pins(pins.SRU_PINS)
        self._authkey = secrets.token_bytes(16)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            {
                ENV_PORT: str(port),
                ENV_AUTHKEY: self._authkey.hex(),
                "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(SRU_SRC)]),
            }
        )
        cmd = list(command) if command else [str(SRU_PYTHON), "-m", __package__ + ".sidecar"]
        with self._log_path.open("ab") as log:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Create a separate process group so cleanup targets only
                # this sidecar group
            )
        self._conn: Optional[Connection] = None
        try:
            self._conn = Client(("127.0.0.1", self._wait_ready()), authkey=self._authkey)
            self.hello = self._call({"op": "hello"})
        except InfraError:
            self.close()
            raise
        except OSError as exc:
            self.close()
            raise InfraError(f"连不上 SRU 旁车: {exc}") from exc

    def _wait_ready(self) -> int:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            for line in self._log_path.read_text(errors="replace").splitlines():
                if line.startswith(READY_TAG):
                    return int(line.split()[1])
            if self._proc.poll() is not None:
                code = self._proc.returncode
                raise InfraError(f"SRU 旁车启动即退出(码 {code}),见 {self._log_path}")
            time.sleep(POLL_S)
        raise InfraError(f"SRU 旁车 {STARTUP_TIMEOUT_S:.0f} s 没就绪,见 {self._log_path}")

    def _call(self, request: Dict[str, Any], timeout_s: float = STEP_TIMEOUT_S) -> Dict[str, Any]:
        if self._conn is None:
            raise InfraError("SRU 旁车已关闭")
        try:
            self._conn.send(request)
            if not self._conn.poll(timeout_s):
                raise InfraError(f"SRU 旁车 {timeout_s:.0f} s 没回复({request.get('op')})")
            reply = self._conn.recv()
        except (EOFError, OSError) as exc:
            raise InfraError(f"SRU 旁车断开: {exc}") from exc
        if not reply.get("ok"):
            raise RuntimeError(f"SRU 旁车报错: {reply.get('error')}")
        return reply

    def reset(self) -> None:
        self._call({"op": "reset"})

    def step(
        self, depth_m: np.ndarray, odom: OdomState, target_odom_xy: Tuple[float, float]
    ) -> PointGoalCommand:
        pos, quat, lin, ang = odom_inputs(odom)
        start = time.perf_counter()
        reply = self._call(
            {
                "op": "step",
                "depth_m": np.ascontiguousarray(depth_m, dtype=np.float32),
                "robot_pos_w": pos,
                "robot_quat_wxyz": quat,
                "linear_vel_b": lin,
                "angular_vel_b": ang,
                "target_pos_w": [float(target_odom_xy[0]), float(target_odom_xy[1]), pos[2]],
            }
        )
        vx, vy, wz = reply["cmd"]
        return PointGoalCommand(vx, vy, wz, bool(reply["arrived"]), time.perf_counter() - start)

    def close(self) -> None:
        """Request shutdown, wait, then SIGTERM/SIGKILL the owned process group if needed;
        idempotent.
        """
        if self._conn is not None:
            try:
                self._call({"op": "shutdown"}, timeout_s=SHUTDOWN_TIMEOUT_S)
            except (InfraError, RuntimeError):
                pass
            self._conn.close()
            self._conn = None
        if self._proc.poll() is None:
            try:
                self._proc.wait(SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(self._proc.pid, sig)
                        self._proc.wait(SHUTDOWN_TIMEOUT_S)
                        break
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        continue
