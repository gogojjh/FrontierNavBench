"""Integrity verification must inspect the same relocated inputs used at runtime."""

import hashlib
import json

import pytest

from scripts import check_install


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    assets = tmp_path / "assets"
    data = tmp_path / "data"
    assets.mkdir()
    data.mkdir()
    rows = []
    for path, relative in ((assets / "example.bin", "assets/example.bin"),
                           (data / "example.bin", "data/example.bin")):
        path.write_bytes(b"published input")
        rows.append({"path": relative, "size_bytes": path.stat().st_size,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (assets / "manifest.json").write_text(json.dumps({"files": rows}))
    monkeypatch.setattr(check_install, "REPO", repo)
    monkeypatch.setattr(check_install, "asset_root", lambda: assets)
    monkeypatch.setattr(check_install, "data_root", lambda: data, raising=False)
    return assets, data


def test_verifies_relocated_runtime_inputs(relocated):
    assert check_install.verify_assets() == 2


def test_relocated_tampering_is_rejected(relocated):
    _, data = relocated
    (data / "example.bin").write_bytes(b"altered content")
    with pytest.raises(ValueError, match="resized asset|Changed asset"):
        check_install.verify_assets()


def test_relocated_symlink_is_rejected(relocated):
    assets, data = relocated
    path = data / "example.bin"
    path.unlink()
    path.symlink_to(assets / "example.bin")
    with pytest.raises(ValueError, match="symlinks"):
        check_install.verify_assets()
