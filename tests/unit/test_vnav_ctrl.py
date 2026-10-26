"""Control protocol tests: idempotence, retries, frame deduplication, fault states and idle exit."""

import math
import threading
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pytest
import zmq

from vnav_benchmark.sim import ctrl_protocol as cp
from vnav_benchmark.sim.backend import KinematicBackend, KinematicFaults, yaw_from_quat
from vnav_benchmark.sim.ctrl_client import CtrlClient, SimFault, SimUnavailable
from vnav_benchmark.sim.ctrl_server import CtrlServer


def _req(server: CtrlServer, session: str, req: int, op: str, args: dict = None) -> dict:
    reply, _ = server.handle(cp.encode_request(session, req, op, args))
    return cp.decode_reply(reply)


def _start(server: CtrlServer) -> None:
    assert _req(server, "s1", 1, cp.OP_HELLO)["ok"]
    assert _req(server, "s1", 2, cp.OP_RESET, {"x": 1.0, "y": 2.0, "yaw": 0.0})["ok"]


def test_step_trace_and_motion() -> None:
    server = CtrlServer(KinematicBackend(), "inproc://x")
    _start(server)
    out = _req(server, "s1", 3, cp.OP_STEP, {"cmd": [0.5, 0, 0], "hold_s": 1.0})
    assert out["ok"]
    step = cp.step_from_wire(out["result"])
    assert [round(s.t_sim, 3) for s in step.trace] == [round(0.1 * k, 3) for k in range(1, 11)]
    assert math.isclose(step.obs.pose.position_xyz[0], 1.5, abs_tol=1e-6)
    assert step.obs.t_sim == pytest.approx(1.0)


def test_replay_is_idempotent() -> None:
    backend = KinematicBackend()
    server = CtrlServer(backend, "inproc://x")
    _start(server)
    raw = cp.encode_request("s1", 3, cp.OP_STEP, {"cmd": [0.5, 0, 0], "hold_s": 0.5})
    first, _ = server.handle(raw)
    again, _ = server.handle(raw)
    assert first == again and server.counters["replays"] == 1
    assert backend._steps == 1 and backend._t == pytest.approx(0.5)


def test_errors() -> None:
    server = CtrlServer(KinematicBackend(), "inproc://x")
    assert _req(server, "s1", 1, cp.OP_HELLO)["ok"]
    no_ep = _req(server, "s1", 2, cp.OP_STEP, {"cmd": [0, 0, 0], "hold_s": 0.1})
    assert no_ep["error"]["code"] == cp.E_NO_EPISODE
    assert _req(server, "s2", 3, cp.OP_TRUTH)["error"]["code"] == cp.E_SESSION_MISMATCH
    assert _req(server, "s1", 4, cp.OP_RESET, {"x": 0, "y": 0, "yaw": 0})["ok"]
    for k, hold in enumerate([0.15, 2.5, 0.0, "x"]):
        bad = _req(server, "s1", 5 + k, cp.OP_STEP, {"cmd": [0, 0, 0], "hold_s": hold})
        assert bad["error"]["code"] == cp.E_HOLD_INVALID
    assert _req(server, "s1", 2, cp.OP_TRUTH)["error"]["code"] == cp.E_STALE_REQUEST
    reply, _ = server.handle(b"garbage")
    assert cp.decode_reply(reply)["error"]["code"] == cp.E_BAD_REQUEST
    bad_pose = _req(server, "s1", 20, cp.OP_RESET, {"x": float("nan"), "y": 0, "yaw": 0})
    assert bad_pose["error"]["code"] == cp.E_POSE_INVALID


def test_server_clamps_and_counts() -> None:
    server = CtrlServer(KinematicBackend(), "inproc://x")
    _start(server)
    out = _req(server, "s1", 3, cp.OP_STEP, {"cmd": [2.0, -1.0, 3.0], "hold_s": 0.1})
    step = cp.step_from_wire(out["result"])
    assert step.applied_cmd == (0.8, -0.3, 0.7) and step.clamped
    assert server.counters["clamped"] == 1
    back = cp.step_from_wire(
        _req(server, "s1", 4, cp.OP_STEP, {"cmd": [-2.0, 0, 0], "hold_s": 0.1})["result"]
    )
    assert back.applied_cmd == (0.0, 0.0, 0.0) and back.clamped  # Reverse motion is disabled


def test_poisoned_state() -> None:
    server = CtrlServer(KinematicBackend(faults=KinematicFaults(raise_on_step=1)), "inproc://x")
    _start(server)
    out = _req(server, "s1", 3, cp.OP_STEP, {"cmd": [0, 0, 0], "hold_s": 0.1})
    assert out["error"]["code"] == cp.E_SIM_FAULT
    assert _req(server, "s1", 4, cp.OP_TRUTH)["error"]["code"] == cp.E_SIM_FAULT
    status = _req(server, "s1", 5, cp.OP_STATUS)
    assert status["ok"] and status["result"]["poisoned"]


def test_set_pose_and_capture_stable() -> None:
    server = CtrlServer(KinematicBackend(), "inproc://x")
    _start(server)
    a = _req(server, "s1", 3, cp.OP_SET_POSE, {"x": 3.0, "y": 1.0, "yaw": 1.2})["result"]["obs"]
    b = _req(server, "s1", 4, cp.OP_CAPTURE)["result"]["obs"]
    c = _req(server, "s1", 5, cp.OP_CAPTURE)["result"]["obs"]
    assert b["rgb_jpeg"] == c["rgb_jpeg"] == a["rgb_jpeg"] and b["t_sim"] == c["t_sim"]
    assert yaw_from_quat(tuple(b["pose"]["quat_xyzw"])) == pytest.approx(1.2)
    # After teleporting, reset again before stepping
    step = _req(server, "s1", 6, cp.OP_STEP, {"cmd": [0, 0, 0], "hold_s": 0.1})
    assert step["error"]["code"] == cp.E_NO_EPISODE


def test_walls_block_motion() -> None:
    occ = np.zeros((40, 40), dtype=bool)
    occ[:, 25] = True  # Wall at x = 2.5 m
    backend = KinematicBackend(occupancy=occ)
    backend.reset(1.0, 2.0, 0.0)
    reply = backend.step((0.8, 0.0, 0.0), 2.0, False, False)
    assert reply.obs.pose.position_xyz[0] < 2.5
    with pytest.raises(cp.CtrlError):
        backend.reset(2.55, 2.0, 0.0)


# ---- Real ZMQ transport ----


@pytest.fixture()
def endpoint(tmp_path: Path) -> str:
    return f"ipc://{tmp_path}/ctrl.sock"


def _serve(server: CtrlServer, ctx: zmq.Context) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, args=(ctx, False), daemon=True)
    thread.start()
    return thread


@pytest.fixture()
def live(endpoint: str) -> Iterator[Tuple[CtrlServer, CtrlClient]]:
    ctx = zmq.Context()
    server = CtrlServer(KinematicBackend(), endpoint)
    thread = _serve(server, ctx)
    client = CtrlClient(endpoint, ctx=ctx)
    yield server, client
    client.shutdown()
    thread.join(5)
    assert server.exit_reason == "shutdown" and server.backend.closed
    ctx.term()


def test_round_trip_and_frame_dedup(live: Tuple[CtrlServer, CtrlClient]) -> None:
    _, client = live
    assert client.hello()["schema"] == cp.SCHEMA
    obs0 = client.reset(1.0, 1.0, 0.0)
    times = [obs0.t_sim]
    for _ in range(3):
        step = client.step(0.3, 0.0, 0.2, 0.5, frames=True)
        assert [t for t, _ in step.frames] == [s.t_sim for s in step.trace]
        assert step.frames[-1][0] == step.obs.t_sim
        times += [t for t, _ in step.frames]
    assert all(b > a for a, b in zip(times, times[1:]))  # Strictly increasing frames without
    # duplicates across steps
    assert client.truth().position_xyz[0] > 1.0


def test_lazy_pirate_retry_no_double_step(endpoint: str) -> None:
    """Reconnect/retry after a lost reply must return cached output without advancing simulation
    twice.
    """
    ctx = zmq.Context()
    backend = KinematicBackend()
    server = CtrlServer(backend, endpoint)
    _start(server)  # Session s1 is ready
    dropped = threading.Event()
    finished = threading.Event()

    def flaky() -> None:
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(endpoint)
        server.handle(sock.recv())  # Execute without replying
        sock.close(0)
        dropped.set()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(endpoint)
        while not finished.is_set():  # If a busy client retries again, return the same idempotently
        # cached response
            if sock.poll(100):
                sock.send(server.handle(sock.recv())[0])
        sock.close(0)

    thread = threading.Thread(target=flaky, daemon=True)
    thread.start()
    client = CtrlClient(endpoint, ctx=ctx, session="s1", timeout_scale=0.05)
    client._req = 2
    step = client.step(0.5, 0.0, 0.0, 1.0)
    finished.set()
    thread.join(5)
    assert dropped.is_set() and client.retries_used >= 1
    assert backend._steps == 1 and step.obs.t_sim == pytest.approx(1.0)
    assert server.counters["replays"] >= 1
    client.close()
    ctx.term()


def test_unavailable_after_retries(endpoint: str) -> None:
    ctx = zmq.Context()
    client = CtrlClient(endpoint, ctx=ctx, timeout_scale=0.005)
    with pytest.raises(SimUnavailable):
        client.hello()
    assert client.retries_used == 3
    client.close()
    ctx.term()


def test_sim_fault_raises_infra(endpoint: str) -> None:
    ctx = zmq.Context()
    server = CtrlServer(KinematicBackend(faults=KinematicFaults(raise_on_step=1)), endpoint)
    thread = _serve(server, ctx)
    client = CtrlClient(endpoint, ctx=ctx)
    client.hello()
    client.reset(0.0, 0.0, 0.0)
    with pytest.raises(SimFault):
        client.step(0.0, 0.0, 0.0, 0.1)
    with pytest.raises(SimFault):
        client.reset(0.0, 0.0, 0.0)  # A faulted server also returns sim_fault for reset
    client.shutdown()
    thread.join(5)
    ctx.term()


def test_idle_exit(endpoint: str) -> None:
    ctx = zmq.Context()
    server = CtrlServer(KinematicBackend(), endpoint, idle_exit_s=0.3)
    assert server.serve_forever(ctx, announce=False) == "idle"
    ctx.term()


def test_reply_carries_server_time() -> None:
    """Subtract reported server execution time from client wall time to measure protocol overhead.
    """
    from vnav_benchmark.sim.ctrl_client import InProcessClient

    client = InProcessClient(CtrlServer(KinematicBackend(), "inproc://x"))
    client.hello()
    client.reset(0.0, 0.0, 0.0)
    client.step(0.3, 0.0, 0.0, 0.5)
    assert client.last_server_ms is not None and client.last_server_ms > 0.0
