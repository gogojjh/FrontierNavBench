#!/usr/bin/env python3
"""Provision the official Go2 and build its frozen Odin payload without starting Isaac.

The NVIDIA robot files stay local under their upstream license. The payload
parameters originate in nav_system's MIT-licensed go2_payload.py/usd_payload.py;
this helper has no imports from that workspace. Run with the simulation Python
or another interpreter with the OpenUSD Python bindings installed.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "assets/robots/go2"
OFFICIAL_ROOT = (
    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/Unitree/Go2/"
)
OFFICIAL_FILES = {
    "go2.usd": "1378aa1134646a026dacf388fee1101a269780cf5ea29c5384142b113fb4100c",
    "go2/configuration.usd": "54f7be1fe3128a78c12c4535768cc69901924d057cc23cabe16f009ab7ddeaec",
}
FROZEN_ODIN_SHA256 = "03fed2e83fad447448e88b308ae81e6dcf0a68fd90d5b657ed44b2906c85076b"
ROOT = "/go2_description"
CHUNK_BYTES = 1 << 20  # Stream large official assets in one-megabyte chunks.
DOWNLOAD_TIMEOUT_S = 60
MASSLESS_LINK_KG = 0.001  # Prevent PhysX's implicit one-kilogram empty-body fallback.


def sha256(path: Path) -> str:
    """Return the content digest without loading the complete asset into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlinks(path: Path) -> None:
    for part in (path,) + tuple(path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlink destinations are refused: {part}")


def _require_hash(path: Path, expected: str) -> None:
    _reject_symlinks(path)
    if not path.is_file():
        raise ValueError(f"Missing robot asset: {path}")
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch for existing asset: {path}")


def _install(temp: Path, destination: Path) -> None:
    """Atomically install complete bytes, refusing even concurrent replacement."""
    _reject_symlinks(destination)
    try:
        os.link(str(temp), str(destination))
    except FileExistsError:
        _require_hash(destination, sha256(temp))


def download(url: str, destination: Path, expected_sha256: str) -> None:
    """Download a verified HTTPS asset or reuse identical existing bytes.

    Args:
        url: An HTTPS source; redirects must also remain HTTPS.
        destination: Final file inside the selected robot directory.
        expected_sha256: Frozen digest from the original asset inventory.

    Raises:
        ValueError: Unsafe destination, changed existing file or wrong digest.
        OSError: Network or filesystem failure; existing files remain untouched.
    """
    if urlparse(url).scheme != "https":
        raise ValueError("Robot downloads require HTTPS")
    _reject_symlinks(destination)
    if destination.exists():
        _require_hash(destination, expected_sha256)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".robot-download-", dir=str(destination.parent))
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
                if urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Robot download redirected away from HTTPS")
                for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256(temp) != expected_sha256:
            raise ValueError(f"SHA256 mismatch for downloaded asset: {destination}")
        temp.chmod(0o644)
        _install(temp, destination)
    finally:
        temp.unlink()


def payloads() -> Tuple[Dict[str, Any], ...]:
    """Return the original measured front payload and assumed rear payload in meters.

    The IMU transform (-0.151975, -0.04, -0.626543), pitch -0.009947 rad
    and trunk height 0.302 m reproduce the frozen real-robot frame convention.
    Front camera/bracket and rear battery/Jetson each weigh 1.5 kg.
    """
    pitch = -0.009947
    x, y, z = -0.151975, -0.04, -0.626543
    origin = (
        -math.cos(pitch) * x + math.sin(pitch) * z,
        -y,
        -math.sin(pitch) * x - math.cos(pitch) * z - 0.302,
    )
    bracket_length = origin[2] - 0.04 - 0.089
    bracket_center = -(0.04 + bracket_length / 2)
    return (
        {"name": "odin_payload", "origin": origin, "mass": 1.5,
         "com": (0.0, 0.0, bracket_center / 1.5),
         "boxes": (((0.12, 0.10, 0.08), (0.0, 0.0, 0.0)),
                   ((0.04, 0.04, bracket_length), (0.0, 0.0, bracket_center)))},
        {"name": "rear_payload", "origin": (-0.10, 0.0, 0.089 + 0.08 / 2),
         "mass": 1.5, "com": (0.0, 0.0, 0.0),
         "boxes": (((0.20, 0.14, 0.08), (0.0, 0.0, 0.0)),)},
    )


def _usd_environment() -> Optional[Dict[str, str]]:
    """Find installed Isaac OpenUSD bindings without importing or starting Isaac."""
    try:
        from pxr import Usd  # noqa: F401
        return None
    except ImportError:
        if os.environ.get("VNAV_ROBOT_USD_CHILD"):
            raise RuntimeError("OpenUSD bindings could not load; install usd-core")
    for entry in sys.path:
        isaac = Path(entry) / "isaacsim"
        for library in (isaac / "extscache").glob("omni.usd.libs-*"):
            if not (library / "pxr/Usd").is_dir():
                continue
            env = dict(os.environ)
            env["VNAV_ROBOT_USD_CHILD"] = "1"
            env["PYTHONPATH"] = str(library) + os.pathsep + env.get("PYTHONPATH", "")
            locations = (Path(sys.prefix) / "lib", library / "bin", isaac / "kit")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, locations)) + os.pathsep + env.get(
                "LD_LIBRARY_PATH", ""
            )
            return env
    raise RuntimeError("OpenUSD is required: use the simulation Python or install usd-core")


def _author_payloads(stage: Any) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    base = stage.GetPrimAtPath(ROOT + "/base")
    if not base:
        raise ValueError("Official Go2 is missing its expected base rigid body")
    translation = Gf.Vec3d(0.0)
    for op in UsdGeom.Xformable(base).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translation = Gf.Vec3d(*op.Get())
            break
    for payload in payloads():
        path = ROOT + "/" + payload["name"]
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(translation + Gf.Vec3d(*payload["origin"]))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(payload["mass"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*payload["com"]))
        for index, (size, center) in enumerate(payload["boxes"]):
            cube = UsdGeom.Cube.Define(stage, path + f"/collision_{index}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        joint = UsdPhysics.FixedJoint.Define(stage, path + "_joint")
        joint.CreateBody0Rel().SetTargets([ROOT + "/base"])
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*payload["origin"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    for name in ("imu", "radar"):
        prim = stage.GetPrimAtPath(ROOT + "/" + name)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not prim.HasAPI(
            UsdPhysics.MassAPI
        ):
            raise ValueError("Official Go2 is missing a mass-correction body: " + name)
        UsdPhysics.MassAPI(prim).GetMassAttr().Set(MASSLESS_LINK_KG)


def verify_semantics(path: Path) -> None:
    """Compare the authored root layer against the original physical payload contract.

    Rebuild an anonymous reference layer from the verified official base asset.
    Comparing its full authored text checks shapes, visibility, masses, transforms,
    joint relationships and all other authored opinions, independent of USDC bytes.
    """
    from pxr import Sdf, Usd

    actual = Sdf.Layer.FindOrOpen(str(path))
    if actual is None:
        raise ValueError(f"Cannot open Odin robot layer: {path}")
    reference = Sdf.Layer.CreateAnonymous("expected-odin.usda")
    # The anonymous layer cannot anchor a relative sublayer: compose an absolute
    # base temporarily, then restore the exact portable authored sublayer path.
    reference.subLayerPaths = [str(path.parent / "go2.usd")]
    stage = Usd.Stage.Open(reference)
    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT))
    _author_payloads(stage)
    stage = None  # Release composition before restoring its portable sublayer path.
    reference.subLayerPaths = ["./go2.usd"]
    if actual.ExportToString() != reference.ExportToString():
        raise ValueError("Odin robot does not match the frozen physical payload contract")


def build_robot(directory: Path) -> Dict[str, Any]:
    """Build and semantically verify an atomic, portable Odin root layer."""
    from pxr import Usd

    for relative, expected in OFFICIAL_FILES.items():
        _require_hash(directory / relative, expected)
    destination = directory / "go2_odin.usd"
    _reject_symlinks(destination)
    if destination.exists():
        return check_robot(directory)
    descriptor, name = tempfile.mkstemp(prefix=".odin-build-", suffix=".usd", dir=str(directory))
    os.close(descriptor)
    temp = Path(name)
    temp.unlink()  # OpenUSD CreateNew requires a fresh file identifier.
    try:
        stage = Usd.Stage.CreateNew(str(temp))
        stage.GetRootLayer().subLayerPaths = ["./go2.usd"]
        root = stage.GetPrimAtPath(ROOT)
        if not root:
            raise ValueError("Official Go2 is missing its expected root prim")
        stage.SetDefaultPrim(root)
        _author_payloads(stage)
        stage.GetRootLayer().Save()
        verify_semantics(temp)
        temp.chmod(0o644)
        _install(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()
    return check_robot(directory)


def check_robot(directory: Path) -> Dict[str, Any]:
    """Check installed files read-only, accepting verified equivalent USD serialization."""
    for relative, expected in OFFICIAL_FILES.items():
        _require_hash(directory / relative, expected)
    destination = directory / "go2_odin.usd"
    _reject_symlinks(destination)
    if not destination.is_file():
        raise ValueError(f"Missing robot asset: {destination}")
    actual = sha256(destination)
    exact = actual == FROZEN_ODIN_SHA256
    if not exact:
        verify_semantics(destination)
    return {"robot": str(destination), "sha256": actual, "exact_frozen_bytes": exact,
            "semantic_match": True, "payload_mass_kg": 3.0, "simulation_started": False}


def main(argv: Optional[List[str]] = None) -> int:
    """Run dry-run, read-only verification or explicitly requested provisioning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Show files; do not write or connect")
    modes.add_argument("--check", action="store_true", help="Read-only installed-file verification")
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().absolute()
    try:
        _reject_symlinks(directory)
        if args.dry_run:
            report = {"mode": "dry_run", "directory": str(directory),
                      "downloads": [{"url": OFFICIAL_ROOT + name, "sha256": value}
                                    for name, value in OFFICIAL_FILES.items()],
                      "generated": "go2_odin.usd", "simulation_started": False}
        else:
            if args.check:
                # Exact frozen files can be checked with the standard library alone.
                target = directory / "go2_odin.usd"
                _reject_symlinks(target)
                needs_usd = target.is_file() and sha256(target) != FROZEN_ODIN_SHA256
            else:
                needs_usd = True
            if needs_usd:
                env = _usd_environment()
                if env is not None:
                    arguments = list(sys.argv[1:] if argv is None else argv)
                    return subprocess.call([sys.executable, str(Path(__file__).resolve())]
                                           + arguments, env=env)
            if args.check:
                report = check_robot(directory)
            else:
                for name, expected in OFFICIAL_FILES.items():
                    download(OFFICIAL_ROOT + name, directory / name, expected)
                report = build_robot(directory)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Robot setup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
