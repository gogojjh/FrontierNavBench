"""Content identity shared by the desktop simulator and the batch client."""

import hashlib
from pathlib import Path
from typing import Optional

from frontier_nav_bench.pins import file_sha256


def sim_dir_sha256(root: Optional[Path] = None) -> str:
    """Return a deterministic digest of all Python sources under the sim directory.

    Args:
        root: Sim directory to hash; defaults to this module's own directory.

    Returns:
        SHA256 of sorted relative POSIX paths and their file SHA256 values, with
        NUL/newline separators. Absolute paths, Git metadata, and bytecode are excluded.
    """
    directory = Path(root) if root is not None else Path(__file__).resolve().parent
    files = sorted(
        (path for path in directory.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
