"""Shared unit-test setup: prohibit external network connections and credentials."""

import socket
from pathlib import Path
from typing import Any, Iterator

import pytest

_SECRET_VARS = ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "FEISHU_WEBHOOK_ID")
# Clear proxy variables too: allowing loopback otherwise lets HTTP clients reach the local proxy.
_PROXY_VARS = ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY")


class _NoNetwork(socket.socket):
    def connect(self, address: Any) -> None:  # type: ignore[override]
        # Allow only loopback and Unix sockets; ZMQ uses libzmq sockets outside this wrapper.
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "localhost", "::1"):
            raise RuntimeError(f"单测不许联网: {address!r}")
        super().connect(address)


@pytest.fixture(scope="session")
def _empty_dotenv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide an empty credential file shared by offline tests."""
    path = tmp_path_factory.mktemp("offline-env") / ".env"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _no_network_no_keys(
    monkeypatch: pytest.MonkeyPatch, _empty_dotenv: Path,
) -> Iterator[None]:
    for name in _SECRET_VARS + _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("VNAV_DATA_DIR", str(repo / "data"))
    monkeypatch.setenv("VNAV_ASSET_DIR", str(repo / "assets"))
    monkeypatch.setenv("VNAV_ENV_FILE", str(_empty_dotenv))
    monkeypatch.setattr(socket, "socket", _NoNetwork)
    yield
