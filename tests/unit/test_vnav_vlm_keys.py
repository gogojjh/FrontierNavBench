"""Test dotenv parsing, variable precedence, missing-key messages, redaction and proxy checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from frontier_nav_bench.vlm import keys


def test_read_dotenv_parses_comments_whitespace_quotes_and_blank_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n"
        "# 这是注释行\n"
        "   DASHSCOPE_API_KEY=abc123\n"  # Leading whitespace.
        '  OPENAI_API_KEY="sk-quoted-value"  \n'  # Quotes and trailing whitespace.
        "EMPTY_VALUE=\n"  # Skip empty values.
        "HASH_VALUE=foo#bar\n"  # Preserve a hash character inside the value.
        "not a valid line\n"  # Skip malformed lines.
        "=novalue\n"  # Skip invalid keys.
        "DASHSCOPE_BASE_URL=https://example.invalid/v1\n",
        encoding="utf-8",
    )
    result = keys.read_dotenv(env_file)
    assert result == {
        "DASHSCOPE_API_KEY": "abc123",
        "OPENAI_API_KEY": "sk-quoted-value",
        "HASH_VALUE": "foo#bar",
        "DASHSCOPE_BASE_URL": "https://example.invalid/v1",
    }


def test_read_dotenv_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert keys.read_dotenv(tmp_path / "nope.env") == {}


def test_resolve_env_api_key_prefers_dotenv_over_env() -> None:
    # .env takes priority: stale terminal credentials cannot replace newer local-file values.
    env = {"DASHSCOPE_API_KEY": "old-stale-key-xxxx"}
    dotenv = {"DASHSCOPE_API_KEY": "new-good-key-yyyy"}
    resolved = keys.resolve_env(["DASHSCOPE_API_KEY"], env, dotenv)
    assert resolved["DASHSCOPE_API_KEY"] == "new-good-key-yyyy"


def test_resolve_env_non_key_var_prefers_env_over_dotenv() -> None:
    env = {"DASHSCOPE_BASE_URL": "https://env.example/v1"}
    dotenv = {"DASHSCOPE_BASE_URL": "https://dotenv.example/v1"}
    resolved = keys.resolve_env(["DASHSCOPE_BASE_URL"], env, dotenv)
    assert resolved["DASHSCOPE_BASE_URL"] == "https://env.example/v1"


def test_resolve_env_non_key_var_falls_back_to_dotenv_when_env_missing() -> None:
    dotenv = {"DASHSCOPE_BASE_URL": "https://dotenv.example/v1"}
    resolved = keys.resolve_env(["DASHSCOPE_BASE_URL"], {}, dotenv)
    assert resolved["DASHSCOPE_BASE_URL"] == "https://dotenv.example/v1"


def test_resolve_env_missing_everywhere_not_in_result() -> None:
    resolved = keys.resolve_env(["NO_SUCH_VAR"], {}, {})
    assert "NO_SUCH_VAR" not in resolved


def test_require_key_missing_openai_message_has_hint_no_secret() -> None:
    with pytest.raises(keys.MissingKeyError) as exc_info:
        keys.require_key("OPENAI_API_KEY", {})
    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert "sk-" not in message


def test_require_key_missing_dashscope_message_points_to_env_file() -> None:
    with pytest.raises(keys.MissingKeyError) as exc_info:
        keys.require_key("DASHSCOPE_API_KEY", {})
    assert "repository .env" in str(exc_info.value)


def test_require_key_present_returns_value() -> None:
    assert keys.require_key("OPENAI_API_KEY", {"OPENAI_API_KEY": "sk-abcdef123456"}) == (
        "sk-abcdef123456"
    )


def test_redact_masks_sk_key_bearer_and_explicit_secrets() -> None:
    text = "key=sk-abc123def456 header=Bearer xyz secret=plain-secret-999"
    out = keys.redact(text, secrets=["plain-secret-999"])
    assert "sk-abc123def456" not in out
    assert "sk-***" in out
    assert "Bearer xyz" not in out
    assert "Bearer ***" in out
    assert "plain-secret-999" not in out


def test_check_proxy_http_https_proxy_with_broken_socks_all_proxy_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Local setup: https_proxy is HTTP; ALL_PROXY is SOCKS5 without PySocks.
    # HTTPS URLs choose the HTTP proxy, avoid SOCKS and return None.
    monkeypatch.setitem(sys.modules, "socks", None)
    env = {"https_proxy": "http://127.0.0.1:7890", "ALL_PROXY": "socks5://127.0.0.1:7890"}
    assert keys.check_proxy("https://api.openai.com/v1/responses", env) is None


def test_check_proxy_only_socks_all_proxy_without_pysocks_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "socks", None)
    env = {"ALL_PROXY": "socks5://127.0.0.1:7890"}
    result = keys.check_proxy("https://api.openai.com/v1/responses", env)
    assert result is not None
    assert "PySocks" in result


def test_check_proxy_no_proxy_overrides_host_even_with_broken_socks_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Check no_proxy before choosing a proxy, bypassing invalid ALL_PROXY settings when matched.
    monkeypatch.setitem(sys.modules, "socks", None)
    env = {
        "ALL_PROXY": "socks5://127.0.0.1:7890",
        "no_proxy": "localhost,127.0.0.1,api.dashscope.invalid",
    }
    assert keys.check_proxy("https://api.dashscope.invalid/compatible-mode/v1", env) is None
    assert keys.check_proxy("https://127.0.0.1/v1", env) is None
    assert keys.check_proxy("https://localhost/v1", env) is None
