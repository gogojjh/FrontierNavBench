"""Check local and simulator repository versions, pins, disk and scene identity."""

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from frontier_nav_bench import mapview, pins
from frontier_nav_bench.paths import repository_root
from frontier_nav_bench.scene_fingerprint import scene_fingerprint
from frontier_nav_bench.sim.orchestrator import (
    CommandRunner,
    OrchestratorError,
    PortsConfig,
    Ssh,
    choose_desktop_host,
    remote_path,
)

REPOS = {"frontier_nav_bench": "."}
SCENE_FINGERPRINT_KEYS = ("build_blender_md5", "assets_sha256")  # Layout and scene assets.
GB = 1e9


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def check_disk(config: PortsConfig) -> Check:
    probe = config.data_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(str(probe)).free / GB
    return Check(
        "disk_free", free_gb >= config.min_free_gb,
        f"{probe} 空闲 {free_gb:.0f} GB(要求 ≥ {config.min_free_gb:.0f} GB)",
    )  # fmt: skip


def check_local_pins() -> Check:
    bad = pins.check_pins()
    detail = "全部一致" if not bad else "; ".join(f"{m.path} 变了" for m in bad)
    return Check("local_pins", not bad, detail)


def git_sha(path: Path) -> Dict[str, Any]:
    """Return a checkout's HEAD and tracked-file dirty status.

    Uses the same convention as sim/isaac_server.git_sha. Keeping this helper
    separate avoids importing the simulator backend and cv2 during preflight;
    loading cv2 first can trigger a libgomp TLS error on Jetson.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()  # fmt: skip
        return {"head": head, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"head": "", "dirty": None, "error": str(exc)}


def local_git(root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    root = root or repository_root()
    return {name: git_sha(root / rel) for name, rel in REPOS.items()}


def remote_git(config: PortsConfig, ssh: Ssh) -> Dict[str, Dict[str, Any]]:
    """Read one standalone checkout, or explicit metadata from a committed archive."""
    metadata = remote_path(config.desktop_ws, ".vnav-stage.json")
    out = ssh.run(f"if test -f {metadata}; then cat {metadata}; fi\n")
    if out.strip():
        try:
            head = json.loads(out)["head"]
            if not isinstance(head, str) or len(head) != 40:
                raise ValueError("Stage head must be a full Git SHA")
            int(head, 16)
        except (KeyError, TypeError, ValueError) as exc:
            raise OrchestratorError("Invalid .vnav-stage.json metadata") from exc
        return {"frontier_nav_bench": {"head": head, "dirty": False, "stage": True}}
    lines = []
    for name, rel in REPOS.items():
        path = remote_path(config.desktop_ws, rel)
        lines.append(
            f'echo "{name} $(git -C {path} rev-parse HEAD 2>/dev/null || echo none) '
            f'$(git -C {path} status --porcelain --untracked-files=no 2>/dev/null | wc -l)"'
        )
    out = ssh.run("\n".join(lines) + "\n")
    result: Dict[str, Dict[str, Any]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in REPOS:
            head = "" if parts[1] == "none" else parts[1]
            result[parts[0]] = {"head": head, "dirty": int(parts[2]) > 0}
    return result


def compare_git(
    local: Mapping[str, Mapping[str, Any]], remote: Mapping[str, Mapping[str, Any]], tag: str
) -> List[Check]:
    checks = []
    for name in REPOS:
        mine, theirs = local.get(name, {}), remote.get(name, {})
        if theirs.get("stage"):  # Compare the archived revision with local HEAD.
            same = bool(theirs.get("head")) and str(mine.get("head", "")).startswith(
                str(theirs["head"])
            )
            detail = f"Jetson {str(mine.get('head'))[:10]} / 台式机暂存 {theirs['head']}"
            checks.append(Check(f"{tag}_git_{name}", same, detail))
            continue
        same = bool(mine.get("head")) and mine.get("head") == theirs.get("head")
        clean = mine.get("dirty") is False and theirs.get("dirty") is False
        detail = f"Jetson {str(mine.get('head'))[:10]}(未提交改动 {mine.get('dirty')})"
        if tag != "local":
            detail += f" / {tag} {str(theirs.get('head'))[:10]}(未提交改动 {theirs.get('dirty')})"
        checks.append(Check(f"{tag}_git_{name}", same and clean, detail))
    return checks


def check_desktop(config: PortsConfig, runner: CommandRunner) -> List[Check]:
    """Select an endpoint, verify shell access and compare repository revisions."""
    try:
        host, note = choose_desktop_host(config, runner)
    except OrchestratorError as exc:
        return [Check("desktop_route", False, str(exc))]
    checks = [Check("desktop_route", True, f"用 {host}({note})")]
    ssh = Ssh(host, runner)
    try:
        ssh.run("true\n")
        checks.append(Check("desktop_ssh", True, host))
    except OrchestratorError as exc:
        return checks + [Check("desktop_ssh", False, str(exc))]
    try:
        remote = remote_git(config, ssh)
    except OrchestratorError as exc:
        return checks + [Check("desktop_git_frontier_nav_bench", False, str(exc))]
    return checks + compare_git(local_git(), remote, "desktop")


def _route_set_scene_check(episodes_json: Path, remote: Mapping[str, Any]) -> Check:
    """Check recorded route layout MD5 values against the simulator scene."""
    frozen = json.loads(episodes_json.read_text(encoding="utf-8"))
    now = str(remote.get("build_blender_md5", ""))
    bad = []
    registry = mapview.load_maps_config()
    for route in frozen.get("routes") or []:
        session = registry.route_info(route["route"]).session_dir
        truth = json.loads((session / "sim_truth.json").read_text(encoding="utf-8"))
        if truth.get("layout_md5", {}).get("build_blender") != now:
            bad.append(route["map_id"])
    ok = bool(now) and bool(frozen.get("routes")) and not bad
    detail = f"{len(frozen.get('routes') or [])} 条路线录制时场景 = 台式机 {now[:12]}"
    return Check("episodes_scene", ok, detail if not bad else f"场景变了: {', '.join(bad)}")


def check_identity(
    hello: Mapping[str, Any], scene: str, episodes_json: Optional[Path] = None
) -> List[Check]:
    """Compare the server hello response against local scene and code versions.

    Recorded route sets have no server section: their goal images come from
    recordings rather than server renders. Compare the layout MD5 in each
    session's sim_truth.json instead of the USD hash for these sets.
    """
    checks = []
    local = scene_fingerprint(scene)
    remote = hello.get("scene_fingerprint") or {}
    for key in SCENE_FINGERPRINT_KEYS:  # Ignore other scenes and their assets.
        checks.append(
            Check(
                f"scene_{key}", bool(remote) and remote.get(key) == local[key],
                f"台式机 {str(remote.get(key, '(hello 没报 scene_fingerprint)'))[:16]}"
                f" / Jetson {local[key][:16]}",
            )  # fmt: skip
        )
    if episodes_json is not None and "server" not in json.loads(
        Path(episodes_json).read_text(encoding="utf-8")
    ):
        checks.append(_route_set_scene_check(Path(episodes_json), remote))
    elif episodes_json is not None:
        frozen = json.loads(Path(episodes_json).read_text(encoding="utf-8"))
        server = frozen.get("server") or {}
        expected = server.get("usd_sha256")
        frozen_scene = server.get("scene_fingerprint") or {}
        same_scene = all(frozen_scene.get(k) == remote.get(k) for k in SCENE_FINGERPRINT_KEYS)
        checks.append(  # Frozen goal images were rendered with this scene version.
            Check(
                "episodes_scene", bool(frozen_scene) and bool(remote) and same_scene,
                f"冻结集 {str(frozen_scene.get('assets_sha256'))[:12]}"
                f" / 台式机 {str(remote.get('assets_sha256'))[:12]}",
            )  # fmt: skip
        )
        checks.append(
            Check(
                "usd_sha256", bool(expected) and hello.get("usd_sha256") == expected,
                f"台式机 {str(hello.get('usd_sha256'))[:12]} / 冻结集 {str(expected)[:12]}",
            )  # fmt: skip
        )
    server_pins = dict(hello.get("pins") or {})
    diff = sorted(set(server_pins.items()) ^ set(pins.PINS.items()))
    checks.append(
        Check("server_pins", not diff, "一致" if not diff else f"{len(diff)} 项不同")
    )
    server_git = hello.get("git") or {}
    if all(not (g or {}).get("head") for g in server_git.values()):
        # Archives without Git identity rely on desktop_git_* stage checks.
        checks.append(Check("server_git", True, "服务跑在暂存目录,版本见 desktop_git_*"))
    else:
        checks += compare_git(local_git(), server_git, "server")
    return checks


@dataclass(frozen=True)
class PreflightReport:
    checks: List[Check]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_json(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}

    def markdown(self) -> str:
        rows = ["| 检查 | 结果 | 说明 |", "|---|---|---|"]
        rows += [f"| {c.name} | {'✓' if c.ok else '✗'} | {c.detail} |" for c in self.checks]
        return "\n".join(rows)


def run_preflight(
    config: PortsConfig,
    runner: CommandRunner,
    scene: str,
    hello: Optional[Mapping[str, Any]] = None,
    episodes_json: Optional[Path] = None,
) -> PreflightReport:
    checks = [check_disk(config), check_local_pins()]
    checks += compare_git(local_git(), local_git(), "local")  # Local tracked-file cleanliness.
    checks += check_desktop(config, runner)
    if hello is not None:
        checks += check_identity(hello, scene, episodes_json)
    return PreflightReport(checks)
