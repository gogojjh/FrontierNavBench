"""Single-threaded lockstep control server with idempotency and idle exit.

hello creates or replaces a session; requests from another session return
session_mismatch. Cache the last (session, req) response; duplicate requests reuse
it without execution, while older requests return stale_request.
Unexpected backend exceptions poison the server: only status and shutdown remain
available, and other operations return sim_fault until restart. Exit after
IDLE_EXIT_S without requests, polling every POLL_MS without rendering or stepping.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import zmq

from frontier_nav_bench import fairness
from frontier_nav_bench.sim import ctrl_protocol as cp
from frontier_nav_bench.sim.backend import SimBackend

POLL_MS = 100  # Idle polling interval.
IDLE_EXIT_S = 15 * 60.0  # Exit after fifteen minutes without requests.
READY_TOKEN = "VNAV_SERVER_READY"  # Print after binding; the orchestrator waits for this marker.
EXIT_TOKEN = "VNAV_SERVER_EXIT"  # Print before exiting, followed by shutdown/idle/error reason.
SET_POSE_SETTLE_S = 1.5  # Default set_pose settling duration, matching go2_rig.SETTLE_S.
CAPTURE_WARM_RENDERS = 3  # Default capture warm-up render count.

LOG = logging.getLogger("vnav.ctrl_server")


def _check_hold(hold_s: Any) -> float:
    try:
        hold = float(hold_s)
    except (TypeError, ValueError) as exc:
        raise cp.CtrlError(cp.E_HOLD_INVALID, f"hold_s 不是数: {hold_s!r}") from exc
    steps = hold / cp.HOLD_STEP_S
    if not math.isfinite(hold) or abs(steps - round(steps)) > 1e-6:
        raise cp.CtrlError(cp.E_HOLD_INVALID, f"hold_s={hold_s} 不是 0.1 s 整数倍")
    if not (fairness.HOLD_MIN_S - 1e-9 <= hold <= fairness.HOLD_MAX_S + 1e-9):
        raise cp.CtrlError(cp.E_HOLD_INVALID, f"hold_s={hold_s} 超出 [0.1, 2.0]")
    return round(round(steps) * cp.HOLD_STEP_S, 6)


def _clamp_cmd(cmd: Any) -> Tuple[Tuple[float, float, float], bool]:
    """Clamp again server-side as a counted safeguard after interface checks."""
    try:
        vx, vy, wz = (float(v) for v in cmd)
    except (TypeError, ValueError) as exc:
        raise cp.CtrlError(cp.E_BAD_REQUEST, f"cmd 应为 [vx, vy, wz]: {cmd!r}") from exc
    vals = [v if math.isfinite(v) else 0.0 for v in (vx, vy, wz)]
    out = (
        min(max(vals[0], fairness.VX_MIN), fairness.VX_MAX),
        min(max(vals[1], -fairness.VY_MAX), fairness.VY_MAX),
        min(max(vals[2], -fairness.WZ_MAX), fairness.WZ_MAX),
    )
    return out, out != (vx, vy, wz)


def _xyyaw(args: Dict[str, Any]) -> Tuple[float, float, float]:
    try:
        x, y, yaw = float(args["x"]), float(args["y"]), float(args["yaw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise cp.CtrlError(cp.E_BAD_REQUEST, f"要 x / y / yaw: {args!r}") from exc
    if not all(math.isfinite(v) for v in (x, y, yaw)):
        raise cp.CtrlError(cp.E_POSE_INVALID, "位姿有 NaN / inf")
    return x, y, yaw


class CtrlServer:
    def __init__(
        self,
        backend: SimBackend,
        endpoint: str,
        info_extra: Optional[Dict[str, Any]] = None,
        idle_exit_s: float = IDLE_EXIT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.endpoint = endpoint
        self.info_extra = dict(info_extra or {})
        self.idle_exit_s = idle_exit_s
        self._clock = clock
        self.session: Optional[str] = None
        self._last: Optional[Tuple[str, int, bytes]] = None  # (session, req, response bytes)
        self.poisoned: Optional[str] = None
        self.in_episode = False
        self.counters: Dict[str, int] = {"requests": 0, "replays": 0, "errors": 0, "clamped": 0}
        self.exit_reason = ""

    # -- Handle one request; pure logic callable directly in tests. --
    def handle(self, raw: bytes) -> Tuple[bytes, bool]:
        """Return (response bytes, should exit)."""
        self.counters["requests"] += 1
        try:
            request = cp.decode_request(raw)
        except cp.CtrlError as err:
            self.counters["errors"] += 1
            return cp.encode_reply("", -1, error=err), False
        if self._last is not None and self._last[:2] == (request.session, request.req):
            self.counters["replays"] += 1
            return self._last[2], False
        if request.op == cp.OP_HELLO:
            if request.session != self.session:  # Clear cache and restart request numbering.
                self.session, self._last, self.in_episode = request.session, None, False
        elif request.session != self.session:
            return self._error(request, cp.CtrlError(cp.E_SESSION_MISMATCH, "先 hello")), False
        if self._last is not None and request.req < self._last[1]:
            return self._error(request, cp.CtrlError(cp.E_STALE_REQUEST, "req 比上一条旧")), False
        should_exit = request.op == cp.OP_SHUTDOWN
        if self.poisoned and request.op not in (cp.OP_STATUS, cp.OP_SHUTDOWN):
            reply = cp.encode_reply(
                request.session, request.req, error=cp.CtrlError(cp.E_SIM_FAULT, self.poisoned)
            )
        else:
            try:
                t0 = time.perf_counter()
                result = self._dispatch(request)
                server_ms = (time.perf_counter() - t0) * 1000.0  # Backend execution time.
                reply = cp.encode_reply(request.session, request.req, result=result,
                                        server_ms=server_ms)  # fmt: skip
            except cp.CtrlError as err:
                self.counters["errors"] += 1
                reply = cp.encode_reply(request.session, request.req, error=err)
            except Exception as exc:  # Unexpected backend failure poisons the server.
                LOG.exception("后端异常,进入中毒态")
                self.poisoned = f"{type(exc).__name__}: {exc}"
                self.counters["errors"] += 1
                reply = cp.encode_reply(
                    request.session, request.req, error=cp.CtrlError(cp.E_SIM_FAULT, self.poisoned)
                )
        self._last = (request.session, request.req, reply)
        return reply, should_exit

    def _error(self, request: cp.CtrlRequest, err: cp.CtrlError) -> bytes:
        self.counters["errors"] += 1
        return cp.encode_reply(request.session, request.req, error=err)

    def _dispatch(self, request: cp.CtrlRequest) -> Dict[str, Any]:
        op, args = request.op, request.args
        if op == cp.OP_HELLO:
            return {**self.backend.info(), **self.info_extra, "schema": cp.SCHEMA}
        if op == cp.OP_STATUS:
            return {
                "poisoned": self.poisoned,
                "in_episode": self.in_episode,
                "counters": dict(self.counters),
            }
        if op == cp.OP_SHUTDOWN:
            return {"bye": True}
        if op == cp.OP_RESET:
            obs = self.backend.reset(*_xyyaw(args))
            self.in_episode = True
            return {"obs": cp.obs_to_wire(obs)}
        if op == cp.OP_STEP:
            if not self.in_episode:
                raise cp.CtrlError(cp.E_NO_EPISODE, "先 reset")
            hold = _check_hold(args.get("hold_s"))
            cmd, clamped = _clamp_cmd(args.get("cmd"))
            want = args.get("want") or {}
            reply = self.backend.step(
                cmd, hold, bool(want.get("frames", False)), bool(want.get("third", False))
            )
            if clamped:
                self.counters["clamped"] += 1
            out = cp.step_to_wire(reply)
            out["clamped"] = bool(clamped or reply.clamped)
            return out
        if op == cp.OP_SET_POSE:
            settle = float(args.get("settle_s", SET_POSE_SETTLE_S))
            obs = self.backend.set_pose(*_xyyaw(args), settle_s=settle)
            self.in_episode = False  # Require reset after teleporting before scored steps.
            return {"obs": cp.obs_to_wire(obs)}
        if op == cp.OP_CAPTURE:
            warm = int(args.get("warm_renders", CAPTURE_WARM_RENDERS))
            return {"obs": cp.obs_to_wire(self.backend.capture(warm))}
        if op == cp.OP_TRUTH:
            return {"pose": cp.pose_to_wire(self.backend.truth())}
        raise cp.CtrlError(cp.E_BAD_REQUEST, f"没实现的 op {op}")

    # -- Main loop --
    def serve_forever(self, ctx: Optional[zmq.Context] = None, announce: bool = True) -> str:
        ctx = ctx or zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(self.endpoint)
        if announce:
            print(f"{READY_TOKEN} {self.endpoint}", flush=True)
        last_activity = self._clock()
        try:
            while True:
                if sock.poll(POLL_MS, zmq.POLLIN):
                    reply, should_exit = self.handle(sock.recv())
                    sock.send(reply)
                    last_activity = self._clock()
                    if should_exit:
                        self.exit_reason = "shutdown"
                        break
                elif self._clock() - last_activity > self.idle_exit_s:
                    self.exit_reason = "idle"
                    break
        finally:
            sock.close(0)
            if announce:  # Isaac close() exits the process; print the outcome first.
                print(f"{EXIT_TOKEN} {self.exit_reason or 'error'}", flush=True)
            self.backend.close()
        return self.exit_reason
