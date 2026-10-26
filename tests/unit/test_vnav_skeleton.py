import socket

import pytest

import vnav_benchmark


def test_package_imports() -> None:
    assert vnav_benchmark.__version__


def test_network_blocked() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(RuntimeError):
        s.connect(("8.8.8.8", 53))
    s.close()
