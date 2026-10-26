"""VLM client tests for golden requests, parsing, retries/errors, redaction and credentials.

FakeSession/FakeResponse replay scripted HTTP outcomes; inject fake sleeps/clocks.
No real network requests are sent.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests
from vnav_fake_client import FakeVlmClient

from vnav_benchmark.vlm.client import (
    ENDPOINTS,
    ImagePart,
    TextPart,
    VlmBadRequestError,
    VlmClient,
    VlmFatalError,
    VlmRequest,
    VlmTransientError,
    build_chat_body,
    build_responses_body,
    endpoint_info,
)
from vnav_benchmark.vlm.keys import MissingKeyError

GOLDEN_DIR = Path(__file__).parent / "golden"

_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"vx": {"type": "number"}},
    "required": ["vx"],
}
_SYSTEM_TEXT = "You are a navigation policy. Output JSON only."


def _fixed_request() -> VlmRequest:
    """Fixed request shared by golden-body comparisons and parser tests."""
    return VlmRequest(
        system=_SYSTEM_TEXT,
        parts=(
            TextPart("GOAL IMAGE:"),
            ImagePart(b"\xff\xd8\xff\xe0fakejpeg", 640, 480),
            TextPart("END OF GOAL IMAGE", cache_breakpoint=True),
            TextPart("DECISION 0"),
        ),
        schema_name="vnav_action_v1",
        schema=_SCHEMA,
    )


def _dump(body: Dict[str, Any]) -> str:
    return json.dumps(body, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


class FakeResponse:
    """Fake HTTP response exposing only status_code, json(), headers and text needed by tests."""

    def __init__(
        self,
        status_code: int,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        text: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.headers: Dict[str, str] = headers if headers is not None else {}
        if text is not None:
            self.text = text
        elif json_data is not None:
            self.text = json.dumps(json_data)
        else:
            self.text = ""

    def json(self) -> Dict[str, Any]:
        if self._json_data is None:
            raise ValueError("假响应没有 JSON 体")
        return self._json_data


class FakeSession:
    """Replay FakeResponse/exception script entries and record each HTTP call argument."""

    def __init__(self, script: List[Any]) -> None:
        self._script = list(script)
        self.calls: List[Dict[str, Any]] = []

    def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[Any] = None,
    ) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if not self._script:
            raise AssertionError("FakeSession 脚本用完")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(
    endpoint: str,
    session: Any,
    env: Dict[str, str],
    tmp_path: Path,
    sleep_calls: Optional[List[float]] = None,
    dotenv_path: Optional[Path] = None,
) -> VlmClient:
    sleep_calls = sleep_calls if sleep_calls is not None else []
    return VlmClient(
        endpoint,
        session=session,
        env=env,
        dotenv_path=dotenv_path if dotenv_path is not None else tmp_path / "nope.env",
        sleep=lambda s: sleep_calls.append(s),
        clock=lambda: 0.0,
    )


# ---------------------------------------------------------------------------
# 9-10: Golden request bodies and variants
# ---------------------------------------------------------------------------


def test_golden_responses_body_matches_astra_flex() -> None:
    body = build_responses_body(ENDPOINTS["astra_flex"], _fixed_request())
    rendered = _dump(body)
    golden_path = GOLDEN_DIR / "vlm_responses_body.json"
    if os.environ.get("VNAV_REGEN_GOLDEN") == "1":
        golden_path.write_text(rendered, encoding="utf-8")
    assert rendered == golden_path.read_text(encoding="utf-8")


def test_golden_chat_body_matches_qwen() -> None:
    body = build_chat_body(ENDPOINTS["qwen"], _fixed_request())
    rendered = _dump(body)
    golden_path = GOLDEN_DIR / "vlm_chat_body.json"
    if os.environ.get("VNAV_REGEN_GOLDEN") == "1":
        golden_path.write_text(rendered, encoding="utf-8")
    assert rendered == golden_path.read_text(encoding="utf-8")


def test_astra_non_flex_body_has_no_service_tier_key() -> None:
    body = build_responses_body(ENDPOINTS["astra"], _fixed_request())
    assert "service_tier" not in body


def test_qwen_json_object_variant_sends_json_object_format() -> None:
    ep = replace(ENDPOINTS["qwen"], structured="json_object")
    body = build_chat_body(ep, _fixed_request())
    assert body["response_format"] == {"type": "json_object"}


def test_too_many_cache_breakpoints_raises_value_error() -> None:
    req = VlmRequest(
        system=_SYSTEM_TEXT,
        parts=(
            TextPart("a", cache_breakpoint=True),
            TextPart("b", cache_breakpoint=True),
            TextPart("c", cache_breakpoint=True),
            TextPart("d", cache_breakpoint=True),
        ),
        schema_name="x",
        schema=_SCHEMA,
    )
    # One system breakpoint plus four text breakpoints exceeds the limit of four
    with pytest.raises(ValueError):
        build_responses_body(ENDPOINTS["astra_flex"], req)


# ---------------------------------------------------------------------------
# 11-12: Successful response parsing
# ---------------------------------------------------------------------------


def test_responses_success_parses_text_usage_and_metadata(tmp_path: Path) -> None:
    payload = {
        "id": "resp_1",
        "model": "gpt-6-astra",
        "status": "completed",
        "service_tier": "flex",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": '{"vx": 0.1}'}]},
        ],
        "usage": {
            "input_tokens": 500,
            "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 50},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 10},
        },
    }
    headers = {"openai-processing-ms": "1234", "x-request-id": "req_1"}
    session = FakeSession([FakeResponse(200, json_data=payload, headers=headers)])
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("astra_flex", session, env, tmp_path)

    result = client.complete(_fixed_request())

    assert result.text == '{"vx": 0.1}'
    assert result.counts.input_tokens == 500
    assert result.counts.cached_tokens == 100
    assert result.counts.cache_write_tokens == 50
    assert result.counts.output_tokens == 40
    assert result.counts.reasoning_tokens == 10
    assert result.server_ms == 1234.0
    assert result.usage.request_id == "req_1"
    assert result.finish_reason == "stop"
    assert result.service_tier == "flex"

    uncached = 500 - 100 - 50
    expected_list = (uncached * 10.0 + 100 * 1.0 + 50 * 10.0 * 1.25 + 40 * 50.0) / 1e6
    assert result.usage.list_usd == pytest.approx(expected_list)
    assert result.usage.billed_usd == pytest.approx(expected_list * 0.5)  # astra_flex

    call = session.calls[0]
    assert call["url"] == "https://api.openai.com/v1/responses"
    assert call["headers"]["Authorization"] == "Bearer sk-test-key-000000"
    assert call["timeout"] == (10.0, 180.0)


def test_chat_success_parses_content_reasoning_and_usage(tmp_path: Path) -> None:
    payload = {
        "id": "chatcmpl_1",
        "model": "qwen3.8-max-0902",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"vx": 0.2}', "reasoning_content": "thinking..."},
            }
        ],
        "usage": {
            "prompt_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 50},
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }
    session = FakeSession([FakeResponse(200, json_data=payload, headers={})])
    env = {
        "DASHSCOPE_API_KEY": "sk-qwen-test-000000",
        "DASHSCOPE_BASE_URL": "https://dashscope.example/compatible-mode/v1",
    }
    client = _client("qwen", session, env, tmp_path)

    result = client.complete(_fixed_request())

    assert result.text == '{"vx": 0.2}'
    assert result.reasoning_text == "thinking..."
    assert result.counts.input_tokens == 300
    assert result.counts.cached_tokens == 50
    assert result.counts.output_tokens == 20
    assert result.counts.reasoning_tokens == 5
    assert result.finish_reason == "stop"
    assert result.server_ms is None


# ---------------------------------------------------------------------------
# 13-14: Retries
# ---------------------------------------------------------------------------


def test_retry_503_twice_then_success(tmp_path: Path) -> None:
    sleep_calls: List[float] = []
    payload = {"id": "r1", "status": "completed", "output": [], "usage": {}}
    session = FakeSession(
        [
            FakeResponse(503, json_data={"error": {"code": "server_error"}}),
            FakeResponse(503, json_data={"error": {"code": "server_error"}}),
            FakeResponse(200, json_data=payload, headers={}),
        ]
    )
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)

    result = client.complete(_fixed_request())

    assert result.retries == 2
    assert sleep_calls == [2.0, 4.0]


def test_retry_429_with_retry_after_header_waits_that_many_seconds(tmp_path: Path) -> None:
    sleep_calls: List[float] = []
    payload = {"id": "r1", "status": "completed", "output": [], "usage": {}}
    session = FakeSession(
        [
            FakeResponse(
                429, json_data={"error": {"code": "rate_limit"}}, headers={"Retry-After": "7"}
            ),
            FakeResponse(200, json_data=payload, headers={}),
        ]
    )
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)

    client.complete(_fixed_request())

    assert sleep_calls == [7.0]


def test_retry_429_without_retry_after_uses_exponential_backoff(tmp_path: Path) -> None:
    sleep_calls: List[float] = []
    payload = {"id": "r1", "status": "completed", "output": [], "usage": {}}
    session = FakeSession(
        [
            FakeResponse(429, json_data={"error": {"code": "rate_limit"}}, headers={}),
            FakeResponse(200, json_data=payload, headers={}),
        ]
    )
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)

    client.complete(_fixed_request())

    assert sleep_calls == [2.0]


def test_timeout_exception_is_retried(tmp_path: Path) -> None:
    sleep_calls: List[float] = []
    payload = {"id": "r1", "status": "completed", "output": [], "usage": {}}
    session = FakeSession(
        [requests.Timeout("timed out"), FakeResponse(200, json_data=payload, headers={})]
    )
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)

    result = client.complete(_fixed_request())

    assert result.retries == 1
    assert sleep_calls == [2.0]


def test_retries_exhausted_raises_transient_error(tmp_path: Path) -> None:
    sleep_calls: List[float] = []
    responses = [FakeResponse(500, json_data={"error": {"code": "server_error"}}) for _ in range(7)]
    session = FakeSession(responses)
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)

    with pytest.raises(VlmTransientError):
        client.complete(_fixed_request())

    assert len(session.calls) == 7
    assert len(sleep_calls) == 6
    assert sleep_calls[-1] <= 60.0


# ---------------------------------------------------------------------------
# 15: Fatal errors / nonretryable 4xx
# ---------------------------------------------------------------------------


def test_401_raises_fatal_without_retry(tmp_path: Path) -> None:
    sleep_calls: List[float] = []
    session = FakeSession([FakeResponse(401, json_data={"error": {"code": "invalid_api_key"}})])
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)

    with pytest.raises(VlmFatalError):
        client.complete(_fixed_request())

    assert sleep_calls == []
    assert len(session.calls) == 1


def test_429_insufficient_quota_is_fatal_not_retried(tmp_path: Path) -> None:
    sleep_calls: List[float] = []
    session = FakeSession([FakeResponse(429, json_data={"error": {"code": "insufficient_quota"}})])
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)

    with pytest.raises(VlmFatalError):
        client.complete(_fixed_request())

    assert sleep_calls == []


def test_429_credit_exhausted_seen_in_v2_smoke_is_fatal(tmp_path: Path) -> None:
    """Smoke-observed error: code=credit_balance_exhausted and type=insufficient_quota."""
    sleep_calls: List[float] = []
    for err in (
        {"code": "credit_balance_exhausted", "type": "insufficient_quota"},
        {"code": "some_new_code", "type": "insufficient_quota"},
    ):
        session = FakeSession([FakeResponse(429, json_data={"error": err})])
        env = {"OPENAI_API_KEY": "sk-test-key-000000"}
        client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)
        with pytest.raises(VlmFatalError):
            client.complete(_fixed_request())
    assert sleep_calls == []


def test_403_dashscope_arrearage_is_fatal(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(403, json_data={"code": "Arrearage"})])
    env = {
        "DASHSCOPE_API_KEY": "sk-qwen-test-000000",
        "DASHSCOPE_BASE_URL": "https://dashscope.example/v1",
    }
    client = _client("qwen", session, env, tmp_path)

    with pytest.raises(VlmFatalError):
        client.complete(_fixed_request())


def test_400_raises_bad_request_error_not_retried(tmp_path: Path) -> None:
    sleep_calls: List[float] = []
    session = FakeSession([FakeResponse(400, json_data={"error": {"code": "invalid_request"}})])
    env = {"OPENAI_API_KEY": "sk-test-key-000000"}
    client = _client("luna_dev", session, env, tmp_path, sleep_calls=sleep_calls)

    with pytest.raises(VlmBadRequestError):
        client.complete(_fixed_request())

    assert sleep_calls == []


# ---------------------------------------------------------------------------
# 16: Redaction
# ---------------------------------------------------------------------------


def test_error_message_never_contains_real_key(tmp_path: Path) -> None:
    secret = "sk-test-SECRET-123456"
    responses = [
        FakeResponse(
            500, json_data={"error": {"code": "server_error", "message": f"bad key {secret}"}}
        )
        for _ in range(7)
    ]
    session = FakeSession(responses)
    env = {"OPENAI_API_KEY": secret}
    client = _client("luna_dev", session, env, tmp_path)

    with pytest.raises(VlmTransientError) as exc_info:
        client.complete(_fixed_request())

    assert secret not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 17: .env precedence
# ---------------------------------------------------------------------------


def test_env_priority_prefers_dotenv_key_and_resolves_base_url_from_env_file(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=new-good-key-yyyy\n"
        "DASHSCOPE_BASE_URL=https://example.invalid/compatible-mode/v1\n",
        encoding="utf-8",
    )
    env = {"DASHSCOPE_API_KEY": "old-stale-key-xxxx"}
    payload = {
        "id": "r1",
        "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        "usage": {},
    }
    session = FakeSession([FakeResponse(200, json_data=payload, headers={})])
    client = _client("qwen", session, env, tmp_path, dotenv_path=env_file)

    client.complete(_fixed_request())

    call = session.calls[0]
    assert call["headers"]["Authorization"] == "Bearer new-good-key-yyyy"
    assert call["url"] == "https://example.invalid/compatible-mode/v1/chat/completions"


# ---------------------------------------------------------------------------
# 18: Missing credentials
# ---------------------------------------------------------------------------


def test_missing_openai_api_key_raises_missing_key_error(tmp_path: Path) -> None:
    with pytest.raises(MissingKeyError):
        _client("luna_dev", FakeSession([]), {}, tmp_path)


# ---------------------------------------------------------------------------
# 19: FakeVlmClient
# ---------------------------------------------------------------------------


def test_fake_vlm_client_replays_script_records_requests_and_estimates_usage() -> None:
    fake = FakeVlmClient(['{"vx": 0.1}', ValueError("boom"), lambda req: "{}"], endpoint="luna_dev")

    result1 = fake.complete(_fixed_request())
    assert result1.text == '{"vx": 0.1}'
    assert result1.usage.billed_usd > 0
    assert len(fake.requests) == 1

    with pytest.raises(ValueError):
        fake.complete(_fixed_request())
    assert len(fake.requests) == 2

    result3 = fake.complete(_fixed_request())
    assert result3.text == "{}"
    assert len(fake.requests) == 3

    with pytest.raises(AssertionError):
        fake.complete(_fixed_request())


# ---------------------------------------------------------------------------
# 20: endpoint_info
# ---------------------------------------------------------------------------


def test_endpoint_info_differences_for_astra_flex_and_qwen() -> None:
    astra_info = endpoint_info(ENDPOINTS["astra_flex"])
    assert astra_info["wire"] == "responses"
    assert astra_info["structured"] == "json_schema_strict"
    assert astra_info["thinking"] == {"reasoning_effort": "low", "verbosity": "low"}
    assert astra_info["temperature"] is None
    assert astra_info["cache"] == {"mode": "explicit", "ttl": "30m", "key": "vnav_benchmark_v1"}
    assert astra_info["service_tier"] == "flex"
    assert astra_info["max_output_tokens"] == 3000

    qwen_info = endpoint_info(ENDPOINTS["qwen"])
    assert qwen_info["wire"] == "chat"
    assert qwen_info["structured"] == "json_schema"
    assert qwen_info["thinking"] == {"enable_thinking": False}
    assert qwen_info["thinking_note"] == astra_info["thinking_note"]
    assert "cannot disable" in qwen_info["thinking_note"]
    assert qwen_info["temperature"] == 0.0
    assert qwen_info["cache"] == {"mode": "implicit"}
    assert qwen_info["service_tier"] == "default"
