"""Verify the distributed runtime and learned model checksums."""

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from vnav_benchmark.paths import asset_root, repository_root

PINS: Dict[str, str] = json.loads(
    (repository_root() / "configs/runtime_manifest.json").read_text(encoding="utf-8")
)["files"]
SIM_SERVER_PINS = tuple(path for path in PINS if path.startswith("vnav_benchmark/sim/"))
SRU_PINS = tuple(path for path in PINS if path.startswith("third_party/SRU-Odin/"))
ASSET_PINS = {
    "weights/sru/vae_encoder.onnx":
        "de70c217c91ef916b877d81c7a134e00d861d6b282f7de2489984b1818596cd0",
    "weights/sru/nav_policy.onnx":
        "f1410cb77ee7897874d69ee7ba4c4814c2388810367b1ae4627bb3c12b523f1b",
    "weights/go2/go2_moe_cts_176k_0.6984.pt":
        "c602e749ac292921e3d6f5b2ab1749c4e4eaa6dbe51f751cce5b669b102d10a2",
    "weights/go2/go2_moe_cts_176k_0.6984.onnx":
        "93ed62bcca5ddd969da1f35960619005f3d3ae692ff41cde79f20697e9b06641",
}

class PinError(RuntimeError):
    """A required source or downloaded asset differs from its recorded checksum."""


@dataclass(frozen=True)
class PinMismatch:
    path: str
    expected: str
    actual: Optional[str]


def file_sha256(path: Path) -> str:
    """Stream a file's checksum without loading large assets into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_pins(
    paths: Optional[Iterable[str]] = None, root: Optional[Path] = None,
) -> List[PinMismatch]:
    """Compare repository source files with the published runtime registry."""
    root = root or repository_root()
    out = []
    for rel in paths if paths is not None else PINS:
        full = root / rel
        actual = file_sha256(full) if full.is_file() else None
        if actual != PINS[rel]:
            out.append(PinMismatch(rel, PINS[rel], actual))
    return out


def require_pins(paths: Optional[Iterable[str]] = None, root: Optional[Path] = None) -> None:
    """Refuse to run with altered or missing runtime sources."""
    bad = check_pins(paths, root)
    if bad:
        raise PinError("Runtime checksum mismatch: " + ", ".join(m.path for m in bad))


def pins_digest() -> Dict[str, str]:
    """Return the exact source registry recorded in simulator and run identities."""
    return dict(PINS)


def require_assets(names: Iterable[str], root: Optional[Path] = None) -> None:
    """Reject missing or changed model files before starting inference."""
    base = root or asset_root()
    for name in names:
        path = base / name
        if not path.is_file() or file_sha256(path) != ASSET_PINS[name]:
            raise PinError("Missing or mismatched model asset: " + name)


def ensure_sru_importable() -> None:
    """Expose the retained SRU package only when its inference sidecar starts."""
    source = str(repository_root() / "third_party/SRU-Odin/Deployment/src")
    if source not in sys.path:
        sys.path.insert(0, source)
