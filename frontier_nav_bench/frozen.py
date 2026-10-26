"""Freeze canonical bytes and SHA256 manifests through exclusive creation.

Used by episode directories (configs/episodes/<name>/) and interface/evaluation
manifests (configs/frozen_manifest_v<N>.json, H8). The approach follows the
canonical_bytes/freeze/load_frozen helpers in tools/retrieval_eval/qa_cases.py
(lines 262/849/836), without importing that module. No overwrite option exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_SCHEMA = "vnav.frozen_dir/1"


class FrozenError(ValueError):
    """Frozen content differs from its hash, or an overwrite was attempted."""


def canonical_bytes(value: Any) -> bytes:
    """Encode deterministically with fixed key order, indentation and no NaN."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_exclusive(path: Path, data: bytes) -> None:
    """Write only if absent using exclusive binary creation; otherwise fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def freeze_dir(
    target: Path,
    files: Mapping[str, bytes],
    meta: Optional[Dict[str, Any]] = None,
    parent_sha256: Optional[str] = None,
) -> str:
    """Freeze files into a new directory and return the manifest SHA256.

    MANIFEST.json stores file hashes, meta and the parent fingerprint (or None).
    Build in a sibling temporary directory, persist all files, then rename to
    prevent partially written frozen directories.
    """
    target = Path(target)
    if target.exists():
        raise FrozenError(f"{target} 已存在:冻结目录只能新建,改动请升版本")
    tmp = target.with_name(f".{target.name}.partial")
    if tmp.exists():
        raise FrozenError(f"{tmp} 是上次没写完的残留,先人工看一眼再删")
    tmp.mkdir(parents=True)
    for name, data in sorted(files.items()):
        if name == MANIFEST_NAME or "/" in name or name.startswith("."):
            raise FrozenError(f"文件名不合法: {name!r}")
        write_exclusive(tmp / name, data)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "files": {name: sha256_bytes(data) for name, data in sorted(files.items())},
        "meta": meta or {},
        "parent_sha256": parent_sha256,
    }
    manifest_bytes = canonical_bytes(manifest)
    write_exclusive(tmp / MANIFEST_NAME, manifest_bytes)
    os.rename(tmp, target)
    return sha256_bytes(manifest_bytes)


def verify_dir(target: Path) -> Dict[str, Any]:
    """Verify registered hashes and reject extra files; return manifest with SHA."""
    target = Path(target)
    raw = (target / MANIFEST_NAME).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise FrozenError(f"{target}: MANIFEST schema 不对")
    if canonical_bytes(manifest) != raw:
        raise FrozenError(f"{target}: MANIFEST 不是固定编码(被手改过?)")
    listed = manifest["files"]
    present = {p.name for p in target.iterdir() if p.is_file() and p.name != MANIFEST_NAME}
    extra = sorted(present - set(listed))
    if extra:
        raise FrozenError(f"{target}: 多出未登记的文件 {extra}")
    for name, digest in listed.items():
        path = target / name
        if not path.is_file():
            raise FrozenError(f"{target}: 缺文件 {name}")
        if sha256_bytes(path.read_bytes()) != digest:
            raise FrozenError(f"{target}: {name} 内容和冻结时不一样")
    return {**manifest, "sha256": sha256_bytes(raw)}


# ---- Interface and evaluation manifests (H8; v1 at T-I0). ----

LAYER_SCHEMA = "vnav.frozen_layer/1"
LAYER_MANIFEST_GLOB = "frozen_manifest_v*.json"
LAYER_PATTERNS = (
    "frontier_nav_bench/__init__.py",
    "frontier_nav_bench/paths.py",
    "third_party/**/*",
    # Interface layer.
    "frontier_nav_bench/interfaces.py",
    "frontier_nav_bench/fairness.py",
    "frontier_nav_bench/sim_env.py",
    "frontier_nav_bench/localizer.py",
    "frontier_nav_bench/mapview.py",
    "frontier_nav_bench/topo.py",  # Repository-local map topology loading and planning.
    "frontier_nav_bench/methods/__init__.py",
    "frontier_nav_bench/sim/*.py",
    # Evaluation layer.
    "frontier_nav_bench/oracle.py",
    "frontier_nav_bench/episodes.py",
    "frontier_nav_bench/metrics.py",
    "frontier_nav_bench/runner.py",
    "frontier_nav_bench/rundir.py",
    "frontier_nav_bench/budget.py",
    "frontier_nav_bench/frozen.py",
    "frontier_nav_bench/pins.py",
    "frontier_nav_bench/preflight.py",
    "frontier_nav_bench/scene_fingerprint.py",
    # Frozen configuration.
    "configs/bench.yaml",
    "configs/maps.yaml",
    "configs/runtime_manifest.json",
    "assets/manifest.json",
    "data/episode_provenance.json",
    "data/maps_renamed.json",
    "data/active_maps/selections.json",
    "scripts/*.py",
    "scripts/*.sh",
    "setup/**/*",
    "pyproject.toml",
    "frontier_nav_bench/scenes.py",
    "data/scenes/**/*",
    "configs/episodes/**/*",
)
"""Frozen interfaces, evaluation, runtime vendor sources and episode configuration."""


def layer_files(repo: Path, patterns: Sequence[str] = LAYER_PATTERNS) -> List[str]:
    out = set()
    for pattern in patterns:
        for path in Path(repo).glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                out.add(path.relative_to(repo).as_posix())
    return sorted(out)


def manifest_version(path: Path) -> int:
    match = re.fullmatch(r"frozen_manifest_v(\d+)\.json", Path(path).name)
    if match is None:
        raise FrozenError(f"清单文件名不对: {path}")
    return int(match.group(1))


def layer_manifests(config_dir: Path) -> List[Path]:
    """Return all manifests sorted by version number."""
    return sorted(Path(config_dir).glob(LAYER_MANIFEST_GLOB), key=manifest_version)


def build_layer_manifest(
    repo: Path, version: int, parent_sha256: Optional[str], note: str = ""
) -> Dict[str, Any]:
    files = layer_files(repo)
    return {
        "schema": LAYER_SCHEMA,
        "version": version,
        "parent_sha256": parent_sha256,
        "note": note,
        "files": {rel: sha256_bytes((Path(repo) / rel).read_bytes()) for rel in files},
    }


def freeze_layer(repo: Path, config_dir: Path, note: str = "") -> Path:
    """Create the next manifest exclusively, linking to the previous file SHA."""
    existing = layer_manifests(config_dir)
    verify_chain(existing)
    version = manifest_version(existing[-1]) + 1 if existing else 1
    parent = sha256_bytes(existing[-1].read_bytes()) if existing else None
    target = Path(config_dir) / f"frozen_manifest_v{version}.json"
    write_exclusive(target, canonical_bytes(build_layer_manifest(repo, version, parent, note)))
    return target


def verify_chain(manifests: Sequence[Path]) -> None:
    """Check contiguous versions from one and parent hashes linking each version."""
    previous: Optional[Path] = None
    for k, path in enumerate(manifests, start=1):
        data = json.loads(path.read_bytes())
        if manifest_version(path) != k or data.get("version") != k:
            raise FrozenError(f"清单版本不连续: {path.name}")
        expect = sha256_bytes(previous.read_bytes()) if previous else None
        if data.get("parent_sha256") != expect:
            raise FrozenError(f"{path.name} 的父指纹和上一版对不上")
        previous = path


def check_layer(repo: Path, config_dir: Path) -> List[str]:
    """List changed, missing or newly frozen files; return [] before freezing."""
    manifests = layer_manifests(config_dir)
    if not manifests:
        return []
    verify_chain(manifests)
    data = json.loads(manifests[-1].read_bytes())
    frozen_files: Dict[str, str] = data["files"]
    problems = []
    now = set(layer_files(repo))
    for rel, digest in sorted(frozen_files.items()):
        path = Path(repo) / rel
        if not path.is_file():
            problems.append(f"缺文件 {rel}")
        elif sha256_bytes(path.read_bytes()) != digest:
            problems.append(f"改过 {rel}")
    problems += [f"新增未登记 {rel}" for rel in sorted(now - set(frozen_files))]
    return problems
