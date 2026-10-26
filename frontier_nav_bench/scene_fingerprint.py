"""Original per-scene layout and referenced-asset identities.

The frozen export records the MD5 of the original build_blender.py and SHA256
of the canonically encoded asset entries used by that scene. These identify
released USD assets and frozen episodes without importing Blender or rebuilding
layouts. Source provenance and notices live alongside each scene export.
"""

from typing import Any, Dict

from frontier_nav_bench.scenes import scene_data


def scene_fingerprint(scene: str) -> Dict[str, Any]:
    """Return a fresh copy of the original layout/asset fingerprint."""
    fingerprint = scene_data(scene)["fingerprint"]
    return {
        "scene": fingerprint["scene"],
        "build_blender_md5": fingerprint["build_blender_md5"],
        "assets_sha256": fingerprint["assets_sha256"],
        "asset_keys": list(fingerprint["asset_keys"]),
    }
