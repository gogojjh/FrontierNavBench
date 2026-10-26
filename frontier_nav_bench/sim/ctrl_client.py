"""Control client with Lazy Pirate retries for benchmark execution.

On timeout, discard the socket, reconnect and resend the same request. The server
returns its cached response, so retries do not advance physics twice. Exhausting
RETRIES raises SimUnavailable; sim_fault raises SimFault. Both are InfraError
failures requiring a fresh service. Other codes raise ctrl_protocol.CtrlError.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

import zmq

from frontier_nav_bench.interfaces import InfraError
from frontier_nav_bench.sim import ctrl_protocol as cp

RETRIES = 3  # Maximum resends after timeouts.
TIMEOUT_S = {  # Per-operation reply timeouts in wall-clock seconds; step adds hold time.
    cp.OP_HELLO: 10.0,
    cp.OP_RESET: 60.0,  # Spawn settling and warm-up.
    cp.OP_SET_POSE: 60.0,
    cp.OP_CAPTURE: 20.0,
    cp.OP_TRUTH: 10.0,
    cp.OP_STATUS: 10.0,
    cp.OP_SHUTDOWN: 10.0,
}
STEP_TIMEOUT_BASE_S = 20.0  # Fixed step overhead allowance.
STEP_TIMEOUT_PER_HOLD = 10.0  # Allow ten wall seconds per hold second.


class SimUnavailable(InfraError):
    """The service is unreachable or repeatedly fails to reply."""


class SimFault(InfraError):
    """The service entered a poisoned state and must restart."""


class CtrlClient:
    def __init__(
        self,
        endpoint: str,
        ctx: Optional[zmq.Context] = None,
        session: Optional[str] = None,
        timeout_scale: float = 1.0,  # Tests use smaller values to shorten timeouts.
    ) -> None:
        self.endpoint = endpoint
        self._ctx = ctx or zmq.Context.instance()
        self.session = session or uuid.uuid4().hex[:12]
        self.timeout_scale = timeout_scale
        self._req = 0
        self._sock: Optional[zmq.Socket] = None
        self.retries_used = 0  # Cumulative retry count for run.json infrastructure records.
        self.last_server_ms: Optional[float] = None  # Previous server execution time.

    def _open(self) -> zmq.Socket:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.endpoint)
        return sock

    def _drop(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None

    def call(self, op: str, args: Optional[Dict[str, Any]] = None, timeout_s: float = 0.0) -> Any:
        """Send one request and return its result, raising protocol/transport errors."""
        self._req += 1
        req = self._req
        raw = cp.encode_request(self.session, req, op, args)
        timeout_ms = int(1000 * (timeout_s or TIMEOUT_S.get(op, 10.0)) * self.timeout_scale)
        for attempt in range(RETRIES + 1):
            if self._sock is None:
                self._sock = self._open()
            self._sock.send(raw)
            if self._sock.poll(timeout_ms, zmq.POLLIN):
                payload = cp.decode_reply(self._sock.recv())
                if (payload.get("session"), payload.get("req")) != (self.session, req) and (
                    payload.get("req") != -1
                ):
                    self._drop()
                    raise SimUnavailable(f"回复对不上请求: {payload.get('req')} != {req}")
                self.last_server_ms = payload.get("server_ms")
                if payload.get("ok"):
                    return payload.get("result", {})
                err = payload.get("error") or {}
                code, msg = str(err.get("code", "")), str(err.get("message", ""))
                if code == cp.E_SIM_FAULT:
                    raise SimFault(msg)
                raise cp.CtrlError(code, msg)
            self._drop()  # A timed-out REQ socket must be discarded and recreated.
            if attempt < RETRIES:
                self.retries_used += 1
        raise SimUnavailable(f"{op} 重试 {RETRIES} 次仍无回复({self.endpoint})")

    # -- Convenience methods --
    def hello(self) -> Dict[str, Any]:
        return self.call(cp.OP_HELLO)

    def reset(self, x: float, y: float, yaw: float) -> cp.WireObs:
        return cp.obs_from_wire(self.call(cp.OP_RESET, {"x": x, "y": y, "yaw": yaw})["obs"])

    def step(
        self,
        vx: float,
        vy: float,
        wz: float,
        hold_s: float,
        frames: bool = True,
        third: bool = False,
    ) -> cp.StepReply:
        timeout = STEP_TIMEOUT_BASE_S + STEP_TIMEOUT_PER_HOLD * hold_s
        args = {"cmd": [vx, vy, wz], "hold_s": hold_s, "want": {"frames": frames, "third": third}}
        return cp.step_from_wire(self.call(cp.OP_STEP, args, timeout))

    def set_pose(self, x: float, y: float, yaw: float, settle_s: float = 1.5) -> cp.WireObs:
        args = {"x": x, "y": y, "yaw": yaw, "settle_s": settle_s}
        return cp.obs_from_wire(self.call(cp.OP_SET_POSE, args)["obs"])

    def capture(self, warm_renders: int = 3) -> cp.WireObs:
        return cp.obs_from_wire(self.call(cp.OP_CAPTURE, {"warm_renders": warm_renders})["obs"])

    def truth(self) -> cp.WirePose:
        return cp.pose_from_wire(self.call(cp.OP_TRUTH)["pose"])

    def status(self) -> Dict[str, Any]:
        return self.call(cp.OP_STATUS)

    def shutdown(self) -> None:
        try:
            self.call(cp.OP_SHUTDOWN)
        finally:
            self.close()

    def close(self) -> None:
        self._drop()


class InProcessClient(CtrlClient):
    """Pass encoded requests directly to an in-process CtrlServer.handle.

    Used by tests and kinematic campaigns. Encoding, idempotency and error codes
    match the network client, without sockets or retries.
    """

    def __init__(self, server: Any, session: Optional[str] = None) -> None:
        super().__init__("inproc://local", ctx=None, session=session)
        self.server = server

    def call(self, op: str, args: Optional[Dict[str, Any]] = None, timeout_s: float = 0.0) -> Any:
        self._req += 1
        reply, _ = self.server.handle(cp.encode_request(self.session, self._req, op, args))
        payload = cp.decode_reply(reply)
        self.last_server_ms = payload.get("server_ms")
        if payload.get("ok"):
            return payload.get("result", {})
        err = payload.get("error") or {}
        code, msg = str(err.get("code", "")), str(err.get("message", ""))
        if code == cp.E_SIM_FAULT:
            raise SimFault(msg)
        raise cp.CtrlError(code, msg)

    def close(self) -> None:
        pass
