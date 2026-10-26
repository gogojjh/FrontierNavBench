"""VLM request/response types, endpoints, API adapters, retries and error classification.

OpenAI /responses serves astra_flex, astra and luna_dev; the OpenAI-compatible
DashScope /chat/completions serves qwen. Pure build_*_body functions construct
dictionaries for golden comparisons. VlmClient handles credentials, proxy checks,
HTTP transport and retries.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, Union

import requests

from vnav_benchmark.interfaces import UsageRecord

from . import keys, pricing
from .pricing import TokenCounts

# ---------------------------------------------------------------------------
# 5.1 Request / response data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextPart:
    text: str
    cache_breakpoint: bool = False  # Explicit astra cache breakpoint; ignored by qwen


@dataclass(frozen=True)
class ImagePart:
    jpeg: bytes
    width: int
    height: int
    detail: str = "high"


Part = Union[TextPart, ImagePart]


@dataclass(frozen=True)
class VlmRequest:
    system: str  # Full system prompt, byte-identical for both providers
    parts: Tuple[Part, ...]  # Ordered user-message content blocks
    schema_name: str  # Structured-output name, such as "vnav_action_v1"
    schema: Mapping[str, Any]  # JSON schema(dict)


@dataclass(frozen=True)
class VlmResult:
    text: str  # Raw model text, expected to contain a JSON string
    usage: UsageRecord  # Includes list_usd, billed_usd and request_id
    counts: TokenCounts  # Includes cache_write_tokens
    endpoint: str  # Endpoint name
    model_reported: str  # Provider-reported model field
    response_id: str  # Response-body id
    service_tier: str  # Response-body service_tier, or empty if absent
    finish_reason: str  # OpenAI normalized status/incomplete_details.reason; qwen finish_reason
    server_ms: Optional[float]  # OpenAI openai-processing-ms header; None when absent for qwen
    wall_s: float  # Total call wall time, including retries and waiting
    retries: int  # Number of retries
    reasoning_text: str = ""  # Store qwen reasoning_content when present, without separate billing


class VlmClientLike(Protocol):
    """Structural protocol shared by VlmClient and FakeVlmClient."""

    @property
    def endpoint(self) -> Endpoint: ...

    def complete(self, req: VlmRequest) -> VlmResult: ...


# ---------------------------------------------------------------------------
# 5.2 Endpoint registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    name: str
    wire: str  # "responses" | "chat"
    provider: str  # "openai" | "dashscope"
    model: str
    key_env: str  # Credential environment-variable name
    base_url: str  # Fixed base URL; read base_url_env when empty
    base_url_env: str  # For example, "DASHSCOPE_BASE_URL"
    flex: bool  # Whether to send service_tier="flex"
    table_ok: bool  # Whether results are eligible for the main table
    structured: str  # "json_schema" | "json_object", relevant only for Chat


ENDPOINTS: Dict[str, Endpoint] = {
    "astra_flex": Endpoint(
        "astra_flex",
        "responses",
        "openai",
        "gpt-6-astra",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1",
        "",
        True,
        True,
        "json_schema",
    ),
    "astra": Endpoint(
        "astra",
        "responses",
        "openai",
        "gpt-6-astra",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1",
        "",
        False,
        True,
        "json_schema",
    ),
    "luna_dev": Endpoint(
        "luna_dev",
        "responses",
        "openai",
        "gpt-6-luna",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1",
        "",
        False,
        False,
        "json_schema",
    ),
    "qwen": Endpoint(
        "qwen",
        "chat",
        "dashscope",
        "qwen3.8-max-0902",
        "DASHSCOPE_API_KEY",
        "",
        "DASHSCOPE_BASE_URL",
        False,
        True,
        "json_schema",
    ),
}

MAX_OUTPUT_TOKENS = 3000  # Identical for both providers
OPENAI_REASONING_EFFORT = "low"  # astra does not support temperature; use fixed effort instead
OPENAI_VERBOSITY = "low"
PROMPT_CACHE_KEY = "vnav_benchmark_v1"  # Constant cache key
PROMPT_CACHE_TTL = "30m"
# Use the lowest reasoning setting for both providers, as approved for this benchmark
QWEN_ENABLE_THINKING = False  # Disable Qwen reasoning and omit thinking_budget
THINKING_NOTE = "astra cannot disable thinking, lowest effort 'low' used; Qwen thinking disabled"
QWEN_TEMPERATURE = 0.0
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 180.0  # Flex queueing can take longer
MAX_RETRIES = 6  # Maximum retries for 429 / 5xx / timeout / connection errors
BACKOFF_BASE_S = 2.0  # Exponential backoff: 2, 4, 8, ... seconds
BACKOFF_MAX_S = 60.0
RETRY_AFTER_MAX_S = 120.0  # Maximum honored Retry-After delay
# Quota/billing error codes or types always stop the campaign
# Smoke-tested credit_balance_exhausted: insufficient_quota type with HTTP 429
FATAL_ERROR_CODES = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "account_deactivated",
    "Arrearage",
    "AllocationQuota.FreeTierOnly",
    "AccessDenied.Unpurchased",
)
MAX_CACHE_BREAKPOINTS = 4  # OpenAI maximum cache-write breakpoints per request


def endpoint_info(ep: Endpoint) -> Dict[str, Any]:
    """Describe API, schema, reasoning, caching, temperature and tier differences for run.json."""
    if ep.wire == "responses":
        return {
            "endpoint": ep.name,
            "model": ep.model,
            "wire": "responses",
            "structured": "json_schema_strict",
            "thinking": {
                "reasoning_effort": OPENAI_REASONING_EFFORT,
                "verbosity": OPENAI_VERBOSITY,
            },
            "thinking_note": THINKING_NOTE,
            "temperature": None,
            "cache": {"mode": "explicit", "ttl": PROMPT_CACHE_TTL, "key": PROMPT_CACHE_KEY},
            "service_tier": "flex" if ep.flex else "default",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
    return {
        "endpoint": ep.name,
        "model": ep.model,
        "wire": "chat",
        "structured": ep.structured,
        "thinking": {"enable_thinking": QWEN_ENABLE_THINKING},
        "thinking_note": THINKING_NOTE,
        "temperature": QWEN_TEMPERATURE,
        "cache": {"mode": "implicit"},
        "service_tier": "flex" if ep.flex else "default",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


# ---------------------------------------------------------------------------
# 5.3 Pure API request builders for golden comparisons
# ---------------------------------------------------------------------------


def _image_data_url(part: ImagePart) -> str:
    b64 = base64.b64encode(part.jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_responses_body(ep: Endpoint, req: VlmRequest) -> Dict[str, Any]:
    """Build an OpenAI /responses request, always caching the system block.

    Raise ValueError if breakpoints, including the system block, exceed
    MAX_CACHE_BREAKPOINTS (four cache writes per request). Field/key ordering is
    fixed and compared byte-for-byte by golden tests.
    """
    breakpoint_count = 1  # System block
    developer_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": req.system,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]

    user_content: List[Dict[str, Any]] = []
    for part in req.parts:
        if isinstance(part, TextPart):
            block: Dict[str, Any] = {"type": "input_text", "text": part.text}
            if part.cache_breakpoint:
                block["prompt_cache_breakpoint"] = {"mode": "explicit"}
                breakpoint_count += 1
            user_content.append(block)
        elif isinstance(part, ImagePart):
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_url(part),
                    "detail": part.detail,
                }
            )
        else:
            raise TypeError(f"未知的内容块类型: {type(part)!r}")

    if breakpoint_count > MAX_CACHE_BREAKPOINTS:
        raise ValueError(
            f"缓存断点数 {breakpoint_count} 超过 OpenAI 单请求上限 {MAX_CACHE_BREAKPOINTS}"
        )

    body: Dict[str, Any] = {
        "model": ep.model,
        "input": [
            {"role": "developer", "content": developer_content},
            {"role": "user", "content": user_content},
        ],
        "reasoning": {"effort": OPENAI_REASONING_EFFORT},
        "text": {
            "format": {
                "type": "json_schema",
                "name": req.schema_name,
                "schema": dict(req.schema),
                "strict": True,
            },
            "verbosity": OPENAI_VERBOSITY,
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "prompt_cache_options": {"mode": "explicit", "ttl": PROMPT_CACHE_TTL},
        "prompt_cache_key": PROMPT_CACHE_KEY,
    }
    if ep.flex:
        body["service_tier"] = "flex"
    return body


def build_chat_body(ep: Endpoint, req: VlmRequest) -> Dict[str, Any]:
    """Build a DashScope OpenAI-compatible /chat/completions request, ignoring cache_breakpoint."""
    user_content: List[Dict[str, Any]] = []
    for part in req.parts:
        if isinstance(part, TextPart):
            user_content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            user_content.append({"type": "image_url", "image_url": {"url": _image_data_url(part)}})
        else:
            raise TypeError(f"未知的内容块类型: {type(part)!r}")

    if ep.structured == "json_object":
        response_format: Dict[str, Any] = {"type": "json_object"}
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": req.schema_name,
                "schema": dict(req.schema),
                "strict": True,
            },
        }

    return {
        "model": ep.model,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": user_content},
        ],
        "response_format": response_format,
        "enable_thinking": QWEN_ENABLE_THINKING,
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
    }


def _dig(mapping: Any, *path: str, default: Any = None) -> Any:
    """Read a nested dictionary path, returning default for missing keys or non-dictionary nodes."""
    cur = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Read a response header case-insensitively."""
    name_lower = name.lower()
    for k, v in headers.items():
        if k.lower() == name_lower:
            return v
    return None


def parse_responses(
    payload: Mapping[str, Any], headers: Mapping[str, str], ep: Endpoint
) -> Dict[str, Any]:
    """Parse OpenAI /responses text, usage, status and metadata."""
    texts: List[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for block in item.get("content", []) or []:
            if isinstance(block, Mapping) and block.get("type") == "output_text":
                texts.append(block.get("text", ""))
    text = "".join(texts)

    usage = payload.get("usage") or {}
    cache_write = _dig(usage, "input_tokens_details", "cache_write_tokens", default=0) or 0
    reasoning = _dig(usage, "output_tokens_details", "reasoning_tokens", default=0) or 0
    counts = TokenCounts(
        input_tokens=usage.get("input_tokens", 0) or 0,
        cached_tokens=_dig(usage, "input_tokens_details", "cached_tokens", default=0) or 0,
        cache_write_tokens=cache_write,
        output_tokens=usage.get("output_tokens", 0) or 0,
        reasoning_tokens=reasoning,
    )

    status = payload.get("status", "")
    if status == "completed":
        finish_reason = "stop"
    elif status == "incomplete":
        finish_reason = _dig(payload, "incomplete_details", "reason", default="incomplete")
    else:
        finish_reason = status

    server_ms_raw = _header(headers, "openai-processing-ms")
    server_ms = float(server_ms_raw) if server_ms_raw is not None else None
    request_id = _header(headers, "x-request-id") or payload.get("id", "")

    return {
        "text": text,
        "counts": counts,
        "model_reported": payload.get("model", ""),
        "response_id": payload.get("id", ""),
        "service_tier": payload.get("service_tier") or "",
        "finish_reason": finish_reason,
        "server_ms": server_ms,
        "request_id": request_id,
        "reasoning_text": "",
    }


def parse_chat(
    payload: Mapping[str, Any], headers: Mapping[str, str], ep: Endpoint
) -> Dict[str, Any]:
    """Parse a DashScope OpenAI-compatible /chat/completions response."""
    choices = payload.get("choices") or [{}]
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    text = message.get("content") or ""
    reasoning_text = message.get("reasoning_content") or ""

    usage = payload.get("usage") or {}
    reasoning = _dig(usage, "completion_tokens_details", "reasoning_tokens", default=0) or 0
    counts = TokenCounts(
        input_tokens=usage.get("prompt_tokens", 0) or 0,
        cached_tokens=_dig(usage, "prompt_tokens_details", "cached_tokens", default=0) or 0,
        cache_write_tokens=0,
        output_tokens=usage.get("completion_tokens", 0) or 0,
        reasoning_tokens=reasoning,
    )

    finish_reason = choice.get("finish_reason") or ""
    request_id = _header(headers, "x-request-id") or payload.get("id", "")

    return {
        "text": text,
        "counts": counts,
        "model_reported": payload.get("model", ""),
        "response_id": payload.get("id", ""),
        "service_tier": payload.get("service_tier") or "",
        "finish_reason": finish_reason,
        "server_ms": None,
        "request_id": request_id,
        "reasoning_text": reasoning_text,
    }


# ---------------------------------------------------------------------------
# 5.4 Error classification and retry handling
# ---------------------------------------------------------------------------


class VlmError(Exception):
    """Base VLM error; redact messages with keys.redact first."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        code: str = "",
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id


class VlmFatalError(VlmError):
    """Authentication/quota/billing errors: stop the campaign on 401, 403 or FATAL_ERROR_CODES."""


class VlmBadRequestError(VlmError):
    """Invalid requests, such as 400 / 404 / 422, are not retried."""


class VlmTransientError(VlmError):
    """429 / 5xx / timeout / connection failure after retry exhaustion."""


def _extract_error_code(payload: Optional[Any]) -> str:
    """Read OpenAI error.code or error.type, including DashScope-compatible error.code.

    Also try top-level code used by some responses; return empty if absent.
    """
    if not isinstance(payload, Mapping):
        return ""
    error = payload.get("error")
    if isinstance(error, Mapping):
        code = error.get("code") or error.get("type")
        if code:
            return str(code)
    code = payload.get("code")
    if code:
        return str(code)
    return ""


def _error_type(payload: Optional[Any]) -> str:
    """Read OpenAI error.type, the broad category such as insufficient_quota, distinct from code."""
    error = payload.get("error") if isinstance(payload, Mapping) else None
    return str(error.get("type") or "") if isinstance(error, Mapping) else ""


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse Retry-After as numeric seconds or the remaining delay to an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (dt - now).total_seconds())


def _backoff_wait(attempt: int) -> float:
    """Return exponential backoff seconds: 2, 4, 8, ... capped at BACKOFF_MAX_S."""
    return min(BACKOFF_BASE_S * (2**attempt), BACKOFF_MAX_S)


class VlmClient:
    """HTTP client choosing an API adapter by endpoint, with retries and error classification."""

    def __init__(
        self,
        endpoint: str,
        *,
        session: Optional[Any] = None,
        env: Optional[Mapping[str, str]] = None,
        dotenv_path: Optional[Path] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if endpoint not in ENDPOINTS:
            raise KeyError(f"未知端点: {endpoint}")
        self._ep = ENDPOINTS[endpoint]
        self._sleep = sleep
        self._clock = clock

        env_map: Mapping[str, str] = env if env is not None else os.environ
        dotenv_file = dotenv_path or Path(env_map.get("VNAV_ENV_FILE") or keys.DOTENV_PATH_DEFAULT)
        dotenv_map = keys.read_dotenv(dotenv_file)
        names = [self._ep.key_env]
        if self._ep.base_url_env:
            names.append(self._ep.base_url_env)
        resolved = keys.resolve_env(names, env_map, dotenv_map)

        self._api_key = keys.require_key(self._ep.key_env, resolved)
        if self._ep.base_url:
            base_url = self._ep.base_url
        else:
            base_url = keys.require_key(self._ep.base_url_env, resolved)

        if self._ep.wire == "responses":
            self._url = base_url.rstrip("/") + "/responses"
        else:
            self._url = base_url.rstrip("/") + "/chat/completions"

        proxy_error = keys.check_proxy(self._url, env_map)
        if proxy_error is not None:
            raise VlmFatalError(proxy_error)

        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            self._session.trust_env = True

    @property
    def endpoint(self) -> Endpoint:
        return self._ep

    def complete(self, req: VlmRequest) -> VlmResult:
        ep = self._ep
        body = build_responses_body(ep, req) if ep.wire == "responses" else build_chat_body(ep, req)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        start = self._clock()
        attempt = 0
        retries = 0

        while True:
            try:
                resp = self._session.post(
                    self._url,
                    headers=headers,
                    json=body,
                    timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt >= MAX_RETRIES:
                    raise VlmTransientError(
                        keys.redact(
                            f"{type(exc).__name__}: 重试 {attempt} 次后仍失败",
                            secrets=[self._api_key],
                        )
                    ) from exc
                self._sleep(_backoff_wait(attempt))
                retries += 1
                attempt += 1
                continue

            status = resp.status_code

            if status == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    # Treat JSON decoding failures as 5xx errors
                    if attempt >= MAX_RETRIES:
                        raise VlmTransientError(
                            keys.redact("响应不是合法 JSON,重试用完仍失败", secrets=[self._api_key])
                        )
                    self._sleep(_backoff_wait(attempt))
                    retries += 1
                    attempt += 1
                    continue
                wall_s = self._clock() - start
                return self._build_result(ep, payload, resp.headers, wall_s, retries)

            try:
                err_payload = resp.json()
            except ValueError:
                err_payload = None
            code = _extract_error_code(err_payload)
            request_id = _header(resp.headers, "x-request-id") or (
                err_payload.get("id", "") if isinstance(err_payload, Mapping) else ""
            )
            body_text = keys.redact(resp.text[:300], secrets=[self._api_key])
            message = f"status={status} code={code} request_id={request_id} body={body_text}"

            if status in (401, 403):
                raise VlmFatalError(message, status=status, code=code, request_id=request_id)

            if status == 429:
                if code in FATAL_ERROR_CODES or _error_type(err_payload) in FATAL_ERROR_CODES:
                    raise VlmFatalError(message, status=status, code=code, request_id=request_id)
                if attempt >= MAX_RETRIES:
                    raise VlmTransientError(
                        message, status=status, code=code, request_id=request_id
                    )
                retry_after = _parse_retry_after(_header(resp.headers, "Retry-After"))
                if retry_after is not None:
                    wait = min(retry_after, RETRY_AFTER_MAX_S)
                else:
                    wait = _backoff_wait(attempt)
                self._sleep(wait)
                retries += 1
                attempt += 1
                continue

            if 500 <= status < 600:
                if attempt >= MAX_RETRIES:
                    raise VlmTransientError(
                        message, status=status, code=code, request_id=request_id
                    )
                self._sleep(_backoff_wait(attempt))
                retries += 1
                attempt += 1
                continue

            raise VlmBadRequestError(message, status=status, code=code, request_id=request_id)

    def _build_result(
        self,
        ep: Endpoint,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        wall_s: float,
        retries: int,
    ) -> VlmResult:
        parsed = (
            parse_responses(payload, headers, ep)
            if ep.wire == "responses"
            else parse_chat(payload, headers, ep)
        )
        counts: TokenCounts = parsed["counts"]
        list_usd, billed_usd = pricing.cost_usd(ep.provider, ep.model, counts, ep.flex)
        usage = UsageRecord(
            provider=ep.provider,
            model=ep.model,
            input_tokens=counts.input_tokens,
            cached_tokens=counts.cached_tokens,
            output_tokens=counts.output_tokens,
            reasoning_tokens=counts.reasoning_tokens,
            list_usd=list_usd,
            billed_usd=billed_usd,
            request_id=parsed["request_id"],
        )
        return VlmResult(
            text=parsed["text"],
            usage=usage,
            counts=counts,
            endpoint=ep.name,
            model_reported=parsed["model_reported"],
            response_id=parsed["response_id"],
            service_tier=parsed["service_tier"],
            finish_reason=parsed["finish_reason"],
            server_ms=parsed["server_ms"],
            wall_s=wall_s,
            retries=retries,
            reasoning_text=parsed["reasoning_text"],
        )
