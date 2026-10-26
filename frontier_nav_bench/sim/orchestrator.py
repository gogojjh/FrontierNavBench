"""Manage isolated Isaac server process groups on localhost or a remote SSH host."""

import math
import os
import re
import secrets
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import yaml

from frontier_nav_bench.interfaces import InfraError
from frontier_nav_bench.paths import data_root, repository_root, setting

REPO_ROOT = repository_root()
PORTS_CONFIG = REPO_ROOT / "configs" / "ports.yaml"
START_SCRIPT_REL = "scripts/isaac_server_start.sh"
REMOTE_LOG_ROOT = "/tmp/vnav-bench"  # One log subdirectory per server launch.
READY_TAG = "VNAV_SERVER_READY"
EXIT_TAG = "VNAV_SERVER_EXIT"
REFUSE_TAGS = ("PIN_MISMATCH", "PORT_BUSY")  # Server refusal markers (exit codes 4 and 5).
READY_TIMEOUT_S = 300.0  # Wall-clock limit for Isaac startup, scene loading and spawn settling.
READY_POLL_S = 3.0  # Simulator log polling interval.
STOP_WAIT_S = 30.0  # Grace period after the shutdown request.
KILL_STEPS: Tuple[Tuple[str, float], ...] = (("INT", 15.0), ("TERM", 8.0), ("KILL", 3.0))
SSH_TIMEOUT_S = 60.0
SSH_RETRIES = 4  # Retry transient SSH handshake failures under concurrent connections.
SSH_RETRY_WAIT_S = 15.0
SSH_TRANSIENT = ("kex_exchange_identification", "Connection reset", "Connection timed out")
LOG_TAIL_LINES = 200  # Number of log tail lines fetched after a failure.
STALE_KILL_STEPS: Tuple[Tuple[str, float], ...] = (("TERM", 10.0), ("KILL", 3.0))  # Stale cleanup.
# Report listeners and visible PID holders; other users may hide PID data.
PORT_CHECK_SH = r"""
q='( sport = :%(port)d )'
echo "__REPO__ $(readlink -f -- %(repo)s 2>/dev/null)"
echo "__LISTEN__ $(ss -ltnH "$q" 2>/dev/null | wc -l)"
for p in $(ss -ltnpH "$q" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
  g=$(ps -o pgid= -p "$p" | tr -d ' ')
  echo "__CWD__ $p $(readlink -f -- "/proc/$p/cwd" 2>/dev/null)"
  echo "__HOLDER__ $p $g $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)"
done
"""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str  # Combined stdout and stderr.


class CommandRunner(Protocol):
    def run(
        self, argv: Sequence[str], input_text: Optional[str] = None, timeout_s: float = 60.0
    ) -> CommandResult: ...


class SubprocessRunner:
    """Execute commands with subprocess.run, merging stdout and stderr."""

    def run(
        self, argv: Sequence[str], input_text: Optional[str] = None, timeout_s: float = 60.0
    ) -> CommandResult:
        try:
            proc = subprocess.run(
                list(argv), input=input_text, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=timeout_s,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            return CommandResult(124, f"超时 {timeout_s:.0f} s: {exc}")
        return CommandResult(proc.returncode, proc.stdout or "")


class OrchestratorError(InfraError):
    """Retryable infrastructure failure: unreachable, unready or refused server."""


@dataclass(frozen=True)
class PortsConfig:
    desktop_hosts: Tuple[str, ...]  # Hosts in preference order.
    desktop_ws: str
    slots: Dict[str, int]
    data_root: Path
    min_free_gb: float

    @property
    def desktop_host(self) -> str:
        """Return the preferred endpoint; choose_desktop_host handles fallback."""
        return self.desktop_hosts[0]


def host_ip(host: str) -> str:
    return host.split("@", 1)[-1]


PING_TIMEOUT_S = 2  # Seconds to wait for a ping reply.


def choose_desktop_host(config: PortsConfig, runner: CommandRunner) -> Tuple[str, str]:
    """Return the first reachable host and a note, or raise OrchestratorError."""
    tried = []
    for index, host in enumerate(config.desktop_hosts):
        if is_local_host(host):
            return host, "localhost"
        ping = runner.run(
            ["ping", "-c", "1", "-W", str(PING_TIMEOUT_S), host_ip(host)],
            timeout_s=PING_TIMEOUT_S + 3,
        )
        if ping.returncode == 0:
            note = "首选地址" if index == 0 else f"首选不通,退到备用(不通的:{', '.join(tried)})"
            return host, note
        tried.append(host)
    raise OrchestratorError(f"台式机所有地址都 ping 不通: {', '.join(tried)}")


def is_local_host(host: str) -> bool:
    """Recognize explicit loopback endpoints without network probes."""
    return host in ("localhost", "127.0.0.1", "::1", "[::1]")


def load_ports_config(path: Path = PORTS_CONFIG) -> PortsConfig:
    """Read slot settings and apply local .env/environment endpoint overrides."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    host = setting("VNAV_DESKTOP_HOST")
    hosts = (host,) if host else tuple(str(h) for h in data.get("desktop_hosts", ["localhost"]))
    return PortsConfig(
        desktop_hosts=hosts,
        desktop_ws=setting("VNAV_DESKTOP_REPO") or str(
            data.get("desktop_ws") or repository_root()
        ),
        slots={str(k): int(v) for k, v in data["slots"].items()},
        data_root=data_root(),
        min_free_gb=float(data["min_free_gb"]),
    )


def remote_path(desktop_ws: str, rel: str) -> str:
    """Quote a simulator path, expanding a leading tilde through remote HOME."""
    base = desktop_ws.rstrip("/")
    if base.startswith("~"):
        return '"$HOME"' + shlex.quote(base[1:] + "/" + rel)
    return shlex.quote(base + "/" + rel)


class Ssh:
    """Run bash scripts locally or over SSH, retrying transient SSH failures."""

    def __init__(
        self, host: str, runner: CommandRunner, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        self.host = host
        self.runner = runner
        self.sleep = sleep

    def argv(self) -> List[str]:
        if is_local_host(self.host):
            return ["bash", "-s"]
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.host, "bash", "-s"]

    def run(self, script: str, timeout_s: float = SSH_TIMEOUT_S, check: bool = True) -> str:
        for attempt in range(SSH_RETRIES):
            result = self.runner.run(self.argv(), input_text=script, timeout_s=timeout_s)
            if result.returncode == 0 or not check:
                return result.stdout
            if not any(tag in result.stdout for tag in SSH_TRANSIENT):
                raise OrchestratorError(
                    f"台式机命令失败(码 {result.returncode}): {result.stdout.strip()[-500:]}"
                )
            if attempt < SSH_RETRIES - 1:
                self.sleep(SSH_RETRY_WAIT_S)
        raise OrchestratorError(f"台式机 ssh 连续 {SSH_RETRIES} 次被拒")


ClientFactory = Callable[[str], Any]  # Client with hello/shutdown/close.


def _default_client(endpoint: str) -> Any:
    from frontier_nav_bench.sim.ctrl_client import CtrlClient

    return CtrlClient(endpoint)


class IsaacOrchestrator:
    """Manage one server lifecycle per slot (s1: 6559; s2: 7559)."""

    def __init__(
        self,
        slot: str,
        config: Optional[PortsConfig] = None,
        runner: Optional[CommandRunner] = None,
        client_factory: ClientFactory = _default_client,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or load_ports_config()
        if slot not in self.config.slots:
            raise ValueError(f"未知槽位 {slot!r},可选 {sorted(self.config.slots)}")
        self.slot = slot
        self.port = self.config.slots[slot]
        self._runner = runner or SubprocessRunner()
        self.ssh = Ssh(self.config.desktop_host, self._runner, sleep)
        self.host_note = ""  # Endpoint selection note from the most recent start.
        self._client_factory = client_factory
        self._sleep = sleep
        self._clock = clock
        self._owner_token = ""
        self._pgid: Optional[int] = None
        self._remote_log: Optional[str] = None
        self._local_log: Optional[Path] = None
        self._identity: Dict[str, Any] = {}
        self._starts = 0

    @property
    def endpoint(self) -> str:
        host = host_ip(self.ssh.host)
        if ":" in host and not host.startswith("["):
            host = "[" + host + "]"
        return f"tcp://{host}:{self.port}"

    def start(
        self,
        scene: str,
        robot_xy: Tuple[float, float],
        robot_yaw: float,
        log_path: Path,
        third: bool = False,
    ) -> str:
        """Start a server, wait until ready and return its control endpoint.

        robot_yaw is in radians; convert it to degrees for the launcher.
        """
        if self._pgid is not None:
            raise OrchestratorError(f"槽位 {self.slot} 上已有本编排器起的服务,先 stop")
        self._starts += 1
        self._owner_token = secrets.token_hex(16)
        self._local_log = Path(log_path)
        self.ssh.host, self.host_note = choose_desktop_host(self.config, self._runner)
        self._clear_port()
        log_dir = f"{REMOTE_LOG_ROOT}/{self.slot}-{os.getpid()}-{self._starts}"
        args = [log_dir, scene, f"{robot_xy[0]:.4f}", f"{robot_xy[1]:.4f}"]
        args += [f"{math.degrees(robot_yaw):.4f}", str(self.port)]
        if third:
            args.append("--third")
        script = "set -eo pipefail\n{}bash {} {}\n".format(
            f"export VNAV_LAUNCH_TOKEN={shlex.quote(self._owner_token)}\n" + "".join(
                f"export {key}={shlex.quote(setting(key))}\n"
                for key in ("VNAV_SIM_PYTHON", "VNAV_SIM_ENV", "VNAV_CONDA",
                            "VNAV_DATA_DIR", "VNAV_ASSET_DIR")
                if is_local_host(self.ssh.host) and setting(key)
            ),
            remote_path(self.config.desktop_ws, START_SCRIPT_REL),
            " ".join(shlex.quote(a) for a in args),
        )
        out = self.ssh.run(script)
        match = re.search(r"ISAAC_PID (\d+) LOG (\S+)", out)
        if match is None:
            raise OrchestratorError(f"启动脚本没打印 ISAAC_PID: {out.strip()[-300:]}")
        self._pgid, self._remote_log = int(match.group(1)), match.group(2)
        try:
            self._wait_ready()
            client = self._client_factory(self.endpoint)
            try:
                self._identity = dict(client.hello())
            finally:
                client.close()
        except Exception as exc:
            try:
                self.stop()
            except OrchestratorError as stop_exc:  # Preserve startup and cleanup failures.
                raise OrchestratorError(f"{exc};收尾时又{stop_exc}") from exc
            raise
        return self.endpoint

    def _port_state(self) -> Tuple[int, List[Tuple[int, str, bool]]]:
        """Return listeners and (process group, command, verified repo cwd) holders."""
        out = self.ssh.run(PORT_CHECK_SH % {
            "port": self.port, "repo": remote_path(self.config.desktop_ws, ""),
        })
        listen = re.search(r"^__LISTEN__ (\d+)$", out, re.MULTILINE)
        repo = re.search(r"^__REPO__ (.*)$", out, re.MULTILINE)
        expected = repo.group(1) if repo else ""
        cwds = dict(re.findall(r"^__CWD__ (\d+) (.*)$", out, re.MULTILINE))
        holders = [
            (int(match.group(2)), match.group(3).strip(),
             bool(expected and cwds.get(match.group(1)) == expected))
            for match in re.finditer(r"^__HOLDER__ (\d+) (\d+) ?(.*)$", out, re.MULTILINE)
        ]
        return (int(listen.group(1)) if listen else 0), holders

    def _clear_port(self) -> None:
        """Refuse occupied slots, including servers started by another campaign."""
        listeners, _holders = self._port_state()
        if listeners:
            raise OrchestratorError(
                f"Port {self.port} is occupied; leaving the existing process untouched"
            )


    def _probe(self) -> Tuple[str, bool]:
        """Return matching log markers and whether the process group is alive."""
        tags = "|".join((READY_TAG, EXIT_TAG) + REFUSE_TAGS)
        out = self.ssh.run(
            f"grep -m3 -E {shlex.quote(tags)} {shlex.quote(str(self._remote_log))} || true\n"
            f"kill -0 -- -{self._pgid} 2>/dev/null && echo __ALIVE__ || echo __DEAD__\n"
        )
        return out, "__ALIVE__" in out

    def _wait_ready(self) -> None:
        deadline = self._clock() + READY_TIMEOUT_S
        while True:
            out, alive = self._probe()
            if READY_TAG in out:
                return
            for tag in REFUSE_TAGS + (EXIT_TAG,):
                if tag in out:
                    raise OrchestratorError(f"服务拒绝启动 / 已退出({tag}),见 {self._local_log}")
            if not alive:
                raise OrchestratorError(f"服务进程没了也没就绪,见 {self._local_log}")
            if self._clock() > deadline:
                raise OrchestratorError(f"服务 {READY_TIMEOUT_S:.0f} s 没就绪,见 {self._local_log}")
            self._sleep(READY_POLL_S)

    def identity(self) -> Dict[str, Any]:
        """Return the latest hello identity for the runner's run.json record."""
        return dict(self._identity)

    def alive(self) -> bool:
        """Recognize only the process group carrying this launch's random receipt."""
        if self._pgid is None or not self._owner_token:
            return False
        marker = shlex.quote("VNAV_LAUNCH_TOKEN=" + self._owner_token)
        script = (
            f"for pid in $(ps -eo pid=,pgid= | awk '$2 == {self._pgid} {{print $1}}'); do\n"
            f"  if tr '\\0' '\\n' < /proc/$pid/environ 2>/dev/null | grep -Fxq -- {marker}; then\n"
            "    echo __OWNED__; break\n"
            "  fi\n"
            "done\n"
        )
        return "__OWNED__" in self.ssh.run(script)

    def _wait_gone(self, seconds: float) -> bool:
        deadline = self._clock() + seconds
        while self.alive():
            if self._clock() > deadline:
                return False
            self._sleep(1.0)
        return True

    def stop(self) -> None:
        """Request shutdown, then escalate INT/TERM/KILL for the owned process group.

        Fetch logs on exit. Repeated calls are safe.

        Raises:
            OrchestratorError: The process group survives all shutdown signals.
        """
        if self._pgid is None:
            return
        try:
            try:
                listeners, holders = self._port_state()
                if listeners and holders and all(group == self._pgid for group, _, _ in holders):
                    client = self._client_factory(self.endpoint)
                    try:
                        if client.hello().get("launch_token") == self._owner_token:
                            client.shutdown()
                    finally:
                        client.close()
            except Exception:  # Clean up even if dead or unready.
                pass
            if self._wait_gone(STOP_WAIT_S):
                return
            unsent = []
            for sig, wait_s in KILL_STEPS:
                if not self.alive():
                    return
                try:  # Retry SSH failures; an already-exited process group is harmless.
                    self.ssh.run(f"kill -{sig} -- -{self._pgid} 2>/dev/null || true\n")
                except OrchestratorError as exc:
                    unsent.append(f"{sig}: {exc}")
                if self._wait_gone(wait_s):
                    return
            note = f"(没发出去的信号:{'; '.join(unsent)})" if unsent else ""
            raise OrchestratorError(
                f"服务关不掉:槽位 {self.slot} 进程组 {self._pgid} "
                f"发完 INT / TERM / KILL 还活着{note}"
            )
        finally:
            self._fetch_log()
            self._pgid = None

    def _fetch_log(self) -> None:
        if self._remote_log is None or self._local_log is None:
            return
        try:
            text = self.ssh.run(
                f"tail -n {LOG_TAIL_LINES} {shlex.quote(self._remote_log)} 2>/dev/null || true\n",
                check=False,
            )
        except OrchestratorError as exc:
            text = f"(拷日志失败: {exc})\n"
        self._local_log.parent.mkdir(parents=True, exist_ok=True)
        with self._local_log.open("a", encoding="utf-8") as fh:
            source = f"{self.ssh.host}:{self._remote_log}"
            fh.write(f"# 台式机 {source}(末 {LOG_TAIL_LINES} 行)\n")
            fh.write(text)
