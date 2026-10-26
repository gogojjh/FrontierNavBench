"""Benchmark simulator control protocol vnav.ctrl/1 (frozen interface layer).

ZMQ REQ/REP provides one reply per request. The server advances simulation only
inside step(); computation time outside it consumes no simulation budget.
The outer envelope uses frontier_nav_bench.sim.protocol.encode(TOPIC_CTRL, ...).
Request payloads contain schema, session, req, op and args. Response payloads
contain schema, session, req, ok and either result or error with code/message.
Request IDs increase strictly within each session. The last (session, req)
response is cached, so retransmission cannot advance physics twice.

Units are meters, radians and simulation seconds; quaternions use xyzw order.
Wire poses contain world truth in scene coordinates. sim_env.py projects away
truth before methods receive observations; methods cannot access this layer.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from frontier_nav_bench.sim import protocol as nav_protocol

SCHEMA = "vnav.ctrl/1"

OP_HELLO = "hello"  # Versions, scene, intrinsics and pinned source hashes.
OP_RESET = "reset"  # New episode: spawn/settle at the start and reset simulation time.
OP_STEP = "step"  # Apply velocity for hold_s seconds; return observation and 10 Hz trace.
OP_SET_POSE = "set_pose"  # Teleport and settle only the robot for goal-image generation.
OP_CAPTURE = "capture"  # Render without physics or advancing simulation time.
OP_TRUTH = "truth"  # Current ground-truth pose.
OP_STATUS = "status"  # Service status, available even in the poisoned state.
OP_SHUTDOWN = "shutdown"  # Exit after replying.
ALL_OPS = (OP_HELLO, OP_RESET, OP_STEP, OP_SET_POSE, OP_CAPTURE, OP_TRUTH, OP_STATUS, OP_SHUTDOWN)

# Error codes.
E_BAD_REQUEST = "bad_request"  # Invalid encoding, missing fields or unknown operation.
E_SESSION_MISMATCH = "session_mismatch"  # Request belongs to another session/client.
E_STALE_REQUEST = "stale_request"  # Request ID predates the cached request.
E_NO_EPISODE = "no_episode"  # Step requested before reset.
E_HOLD_INVALID = "hold_invalid"  # Hold is out of range or not a multiple of 0.1 s.
E_POSE_INVALID = "pose_invalid"  # Pose is out of bounds or inside an obstacle.
E_TELEPORT_UNSTABLE = "teleport_unstable"  # Drift or unstable teleport.
E_RENDER_EMPTY = "render_empty"  # Captured image is empty.
E_SIM_FAULT = "sim_fault"  # Backend failure poisoned the service; restart required.
ALL_ERRORS = (
    E_BAD_REQUEST,
    E_SESSION_MISMATCH,
    E_STALE_REQUEST,
    E_NO_EPISODE,
    E_HOLD_INVALID,
    E_POSE_INVALID,
    E_TELEPORT_UNSTABLE,
    E_RENDER_EMPTY,
    E_SIM_FAULT,
)

TRACE_HZ = 10.0  # Truth/up_z sampling and first-person frame rate match the camera.
HOLD_STEP_S = 0.1  # Hold must be a multiple of this period (20 ticks at 200 Hz).


class CtrlError(Exception):
    """Server-reported error; codes are listed in ALL_ERRORS."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ---- Wire data ----


@dataclass(frozen=True)
class WirePose:
    """Ground-truth base_link pose in world coordinates."""

    position_xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]
    v_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    w_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up_z: float = 1.0  # Body up-axis component; evaluation declares falls after 1 s below 0.5.


@dataclass(frozen=True)
class WireObs:
    """One observation with camera and pose sharing a timestamp."""

    t_sim: float
    rgb_jpeg: bytes
    depth: bytes
    depth_encoding: str  # frontier_nav_bench.sim.protocol.DEPTH_ENC_*
    width: int
    height: int
    K: Tuple[float, ...]  # Nine values in row-major order.
    pose: WirePose


@dataclass(frozen=True)
class TraceSample:
    t_sim: float
    pose: WirePose


@dataclass(frozen=True)
class StepReply:
    obs: WireObs
    trace: List[TraceSample]  # 0.1 s samples; final pose matches obs.
    frames: List[Tuple[float, bytes]] = field(default_factory=list)  # First-person JPEG samples.
    third: List[Tuple[float, bytes]] = field(default_factory=list)  # Third-person JPEG samples.
    applied_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Applied after clamping.
    clamped: bool = False
    fallen_now: bool = False  # Instantaneous up_z < 0.5; evaluation makes the final fall decision.


def _tuple(values: Any) -> Tuple[float, ...]:
    return tuple(float(v) for v in values)


def pose_to_wire(pose: WirePose) -> Dict[str, Any]:
    return asdict(pose)


def pose_from_wire(data: Dict[str, Any]) -> WirePose:
    return WirePose(
        position_xyz=_tuple(data["position_xyz"]),  # type: ignore[arg-type]
        quat_xyzw=_tuple(data["quat_xyzw"]),  # type: ignore[arg-type]
        v_body=_tuple(data.get("v_body", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        w_body=_tuple(data.get("w_body", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        up_z=float(data.get("up_z", 1.0)),
    )


def obs_to_wire(obs: WireObs) -> Dict[str, Any]:
    out = asdict(obs)
    out["pose"] = pose_to_wire(obs.pose)
    return out


def obs_from_wire(data: Dict[str, Any]) -> WireObs:
    return WireObs(
        t_sim=float(data["t_sim"]),
        rgb_jpeg=bytes(data["rgb_jpeg"]),
        depth=bytes(data["depth"]),
        depth_encoding=str(data["depth_encoding"]),
        width=int(data["width"]),
        height=int(data["height"]),
        K=_tuple(data["K"]),
        pose=pose_from_wire(data["pose"]),
    )


def step_to_wire(reply: StepReply) -> Dict[str, Any]:
    return {
        "obs": obs_to_wire(reply.obs),
        "trace": [{"t_sim": s.t_sim, "pose": pose_to_wire(s.pose)} for s in reply.trace],
        "frames": [[t, jpeg] for t, jpeg in reply.frames],
        "third": [[t, jpeg] for t, jpeg in reply.third],
        "applied_cmd": list(reply.applied_cmd),
        "clamped": reply.clamped,
        "fallen_now": reply.fallen_now,
    }


def step_from_wire(data: Dict[str, Any]) -> StepReply:
    return StepReply(
        obs=obs_from_wire(data["obs"]),
        trace=[TraceSample(float(s["t_sim"]), pose_from_wire(s["pose"])) for s in data["trace"]],
        frames=[(float(t), bytes(j)) for t, j in data.get("frames", [])],
        third=[(float(t), bytes(j)) for t, j in data.get("third", [])],
        applied_cmd=_tuple(data.get("applied_cmd", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        clamped=bool(data.get("clamped", False)),
        fallen_now=bool(data.get("fallen_now", False)),
    )


# ---- Envelope ----


@dataclass(frozen=True)
class CtrlRequest:
    session: str
    req: int
    op: str
    args: Dict[str, Any]


def encode_request(session: str, req: int, op: str, args: Optional[Dict[str, Any]] = None) -> bytes:
    payload = {"schema": SCHEMA, "session": session, "req": int(req), "op": op, "args": args or {}}
    return nav_protocol.encode(nav_protocol.TOPIC_CTRL, int(req), 0.0, time.time(), "", payload)


def decode_request(raw: bytes) -> CtrlRequest:
    """Decode a request; reject invalid input with CtrlError(bad_request)."""
    try:
        payload = nav_protocol.decode(raw)["payload"]
        if payload.get("schema") != SCHEMA:
            raise CtrlError(E_BAD_REQUEST, f"schema 应为 {SCHEMA},收到 {payload.get('schema')!r}")
        op = payload["op"]
        if op not in ALL_OPS:
            raise CtrlError(E_BAD_REQUEST, f"不认识的 op {op!r}")
        return CtrlRequest(str(payload["session"]), int(payload["req"]), op, dict(payload["args"]))
    except CtrlError:
        raise
    except Exception as exc:  # Normalize envelope and field errors to bad_request.
        raise CtrlError(E_BAD_REQUEST, str(exc)) from exc


def encode_reply(
    session: str,
    req: int,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[CtrlError] = None,
    server_ms: Optional[float] = None,
) -> bytes:
    """server_ms measures backend execution; client wall time minus it is overhead."""
    payload: Dict[str, Any] = {"schema": SCHEMA, "session": session, "req": int(req)}
    if server_ms is not None:
        payload["server_ms"] = float(server_ms)
    if error is None:
        payload.update(ok=True, result=result or {})
    else:
        payload.update(ok=False, error={"code": error.code, "message": error.message})
    return nav_protocol.encode(
        nav_protocol.TOPIC_CTRL, max(int(req), 0), 0.0, time.time(), "", payload
    )


def decode_reply(raw: bytes) -> Dict[str, Any]:
    payload = nav_protocol.decode(raw)["payload"]
    if payload.get("schema") != SCHEMA:
        raise CtrlError(E_BAD_REQUEST, f"回复 schema 不对: {payload.get('schema')!r}")
    return payload
