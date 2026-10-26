"""Isaac benchmark service: spawn the robot, then answer lockstep control requests.

Usually invoked by scripts/isaac_server_start.sh or the orchestrator. Example:
    OMNI_KIT_ACCEPT_EULA=YES python -m frontier_nav_bench.sim.isaac_server \\
        --scene env0_office_single_floor --start 12.5 2.6 90 --port 6559
Add --third for a third-person tracking camera when recording a video.
Prints VNAV_SERVER_READY tcp://*:<port> when ready and VNAV_SERVER_EXIT <reason>
before exit. Refuses startup when source hashes differ from pins.py; inspect
the source differences before updating the pins."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from frontier_nav_bench import pins
from frontier_nav_bench.sim.ctrl_server import CtrlServer
from frontier_nav_bench.sim.stage_check import sim_dir_sha256

EXIT_BAD_USAGE = 2
EXIT_PIN_MISMATCH = 4
EXIT_PORT_BUSY = 5


def git_sha(path: Path) -> Dict[str, Any]:
    """Return checkout HEAD and dirty status for hello and cross-host preflight."""
    metadata = path / ".vnav-stage.json"
    if metadata.is_file():
        staged = json.loads(metadata.read_text(encoding="utf-8"))
        return {"head": str(staged["head"]), "dirty": False, "stage": True}
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {"head": head, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"head": "", "dirty": None, "error": str(exc)}


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def server_identity(scene: str, scene_usd: Optional[Path]) -> Dict[str, Any]:
    """Return the service code and scene identity included in hello."""
    out: Dict[str, Any] = {
        "sim_dir_sha256": sim_dir_sha256(Path(__file__).resolve().parent),
        "pins": pins.pins_digest(),
        "launch_token": os.environ.get("VNAV_LAUNCH_TOKEN", ""),
        "runtime_sha256": pins.PINS.get("frontier_nav_bench/sim/runtime.py"),
        "git": {
            "frontier_nav_bench": git_sha(Path(__file__).resolve().parents[2]),
        },
    }
    try:  # Compare only this scene layout and assets, ignoring unrelated additions.
        from frontier_nav_bench.scene_fingerprint import scene_fingerprint

        out["scene_fingerprint"] = scene_fingerprint(scene)
    except Exception as exc:  # Missing values allow startup but cause preflight to refuse the run.
        out["scene_fingerprint_error"] = str(exc)
    if scene_usd is not None and scene_usd.is_file():
        out["usd_sha256"] = pins.file_sha256(scene_usd)
    return out


def bench_robot() -> str:
    """Read run.robot directly from configs/bench.yaml without runner/jsonschema."""
    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / "bench.yaml"
    return str(yaml.safe_load(path.read_text(encoding="utf-8"))["run"]["robot"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", required=True, help="场景短名,如 env0_office_single_floor")
    parser.add_argument(
        "--start", nargs=3, type=float, required=True, metavar=("X", "Y", "YAW_DEG"),
        help="出发位姿:场景坐标米 + 朝向度",
    )  # fmt: skip
    parser.add_argument("--port", type=int, required=True, help="ctrl 端口(S1 6559 / S2 7559)")
    parser.add_argument("--third", action="store_true", help="开第三人称跟拍相机(出片回合)")
    parser.add_argument(
        "--robot", default=None, help="机器人;不给就读 configs/bench.yaml 的 run.robot(跑批口径)"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    bad = pins.check_pins(pins.SIM_SERVER_PINS)
    if bad:
        for m in bad:
            print(f"PIN_MISMATCH {m.path} 钉 {m.expected[:12]} 现 {m.actual}", flush=True)
        return EXIT_PIN_MISMATCH
    if not port_free(args.port):
        print(f"PORT_BUSY {args.port}", flush=True)
        return EXIT_PORT_BUSY
    from frontier_nav_bench.sim.isaac_backend import IsaacBackend

    x, y, yaw_deg = args.start
    robot = args.robot or bench_robot()
    backend = IsaacBackend(
        args.scene, (x, y), math.radians(yaw_deg), third_person=args.third, robot=robot
    )
    server = CtrlServer(
        backend,
        f"tcp://*:{args.port}",
        info_extra=server_identity(args.scene, backend.scene_path),
    )
    server.serve_forever()  # backend.close() terminates the process during normal shutdown.
    return 0


if __name__ == "__main__":
    sys.exit(main())
