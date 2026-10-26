"""Resolve local settings without sourcing a shell or an external workspace."""

import os
from pathlib import Path


def repository_root() -> Path:
    """Return this checkout, independent of the caller's working directory."""
    return Path(__file__).resolve().parents[1]


def setting(name: str, default: str = "") -> str:
    """Read a non-secret setting, preferring the environment over the local .env."""
    value = os.environ.get(name)
    if value:
        return value
    from frontier_nav_bench.vlm.keys import read_dotenv

    filename = os.environ.get("VNAV_ENV_FILE")
    dotenv = Path(filename).expanduser() if filename else repository_root() / ".env"
    return read_dotenv(dotenv).get(name) or default


def data_root() -> Path:
    """Return the configured data root; relative values are relative to the repo."""
    path = Path(setting("VNAV_DATA_DIR", "data")).expanduser()
    return path if path.is_absolute() else repository_root() / path


def expand_data_path(value: str) -> Path:
    """Expand the explicit data-root token without rewriting arbitrary fixture paths."""
    return Path(str(value).replace("${VNAV_DATA_DIR}", str(data_root()))).expanduser()


def asset_root() -> Path:
    """Return bundled simulator and model assets, with an optional local override."""
    path = Path(setting("VNAV_ASSET_DIR", "assets")).expanduser()
    return path if path.is_absolute() else repository_root() / path
