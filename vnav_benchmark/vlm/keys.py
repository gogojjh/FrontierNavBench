"""Load and redact VLM credentials.

Match the environment-loader precedence: .env wins for names ending in _API_KEY
because stale shell credentials can cause authentication failures. Parse values
without printing or persisting any raw credentials.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse

# Default credential file is .env in the benchmark repository root.
# Resolve the default relative to this module rather than the current working directory.
# Callers may select ROBOHIKE_ENV_FILE; this module provides only the default path.
DOTENV_PATH_DEFAULT = Path(__file__).resolve().parents[2] / ".env"

_DOTENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_SK_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")
_BEARER_RE = re.compile(r"Bearer\s+\S+")


def read_dotenv(path: Path) -> Dict[str, str]:
    """Parse .env line-by-line using the shell-loader rules; return an empty mapping if absent."""
    if not path.is_file():
        return {}
    result: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.lstrip()  # Strip leading whitespace
        match = _DOTENV_LINE_RE.match(line)
        if not match:
            continue  # Skip comment, blank and malformed lines
        key, val = match.group(1), match.group(2)
        val = val.rstrip()  # Strip trailing whitespace; preserve # inside values
        if val.startswith('"'):
            val = val[1:]
        if val.endswith('"'):
            val = val[:-1]
        if not val:
            continue  # Skip empty values
        result[key] = val
    return result


def resolve_env(
    names: Sequence[str], env: Mapping[str, str], dotenv: Mapping[str, str]
) -> Dict[str, str]:
    """Resolve names with .env priority for _API_KEY values and environment priority for all others.
    """
    result: Dict[str, str] = {}
    for name in names:
        if name.endswith("_API_KEY") and dotenv.get(name):
            result[name] = dotenv[name]
            continue
        env_val = env.get(name)
        if env_val:
            result[name] = env_val
            continue
        dotenv_val = dotenv.get(name)
        if dotenv_val:
            result[name] = dotenv_val
    return result


class MissingKeyError(RuntimeError):
    """Missing required credential/configuration, without exposing credential values."""


def require_key(name: str, resolved: Mapping[str, str]) -> str:
    """Return resolved[name], or raise with troubleshooting guidance that never exposes its value.
    """
    value = resolved.get(name)
    if value:
        return value
    if name == "OPENAI_API_KEY":
        hint = "check OPENAI_API_KEY in the repository .env"
    elif name.startswith("DASHSCOPE_"):
        hint = "check the repository .env"
    else:
        hint = "检查环境变量或 .env 文件"
    raise MissingKeyError(f"缺少 {name},{hint}")


def redact(text: str, secrets: Sequence[str] = ()) -> str:
    """Replace common credential patterns and explicitly supplied secrets with *** in logs/errors.
    """
    out = _SK_KEY_RE.sub("sk-***", text)
    out = _BEARER_RE.sub("Bearer ***", out)
    for secret in secrets:
        if len(secret) >= 8 and secret in out:
            out = out.replace(secret, "***")
    return out


def _no_proxy_hosts(env: Mapping[str, str]) -> Sequence[str]:
    raw = env.get("no_proxy") or env.get("NO_PROXY") or ""
    return [h.strip() for h in raw.split(",") if h.strip()]


def _host_matches_no_proxy(host: str, patterns: Sequence[str]) -> bool:
    host = host.lower()
    for pattern in patterns:
        pattern = pattern.lower()
        if pattern == "*":
            return True
        stripped = pattern.lstrip(".")  # Support common no_proxy domain suffixes such as
        # ".example.com"
        if host == stripped or host.endswith("." + stripped):
            return True
    return False


def check_proxy(url: str, env: Mapping[str, str]) -> Optional[str]:
    """Predict requests proxy selection and detect SOCKS proxies without PySocks.

    no_proxy/NO_PROXY bypasses matching hosts. Otherwise select https_proxy or
    HTTPS_PROXY for HTTPS URLs, then all_proxy/ALL_PROXY. Return the existing
    diagnostic if a selected SOCKS proxy cannot import socks.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if _host_matches_no_proxy(host, _no_proxy_hosts(env)):
        return None

    proxy = None
    if parsed.scheme == "https":
        proxy = env.get("https_proxy") or env.get("HTTPS_PROXY")
    if not proxy:
        proxy = env.get("all_proxy") or env.get("ALL_PROXY")
    if not proxy:
        return None

    if proxy.lower().startswith("socks"):
        try:
            import socks  # noqa: F401  # Probe import availability without using the module
        except ImportError:
            return "ALL_PROXY 是 socks 代理但没装 PySocks;设 https_proxy=http://… 或去掉 ALL_PROXY"
    return None
