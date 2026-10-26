"""Scripted fake client for unit/smoke tests, without real requests."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

from frontier_nav_bench.interfaces import UsageRecord
from frontier_nav_bench.vlm.client import (
    ENDPOINTS,
    Endpoint,
    ImagePart,
    TextPart,
    VlmRequest,
    VlmResult,
)
from frontier_nav_bench.vlm.pricing import (
    TokenCounts,
    cost_usd,
    estimate_text_tokens,
    openai_image_tokens,
    qwen_image_tokens,
)

ScriptItem = Union[str, Exception, Callable[[VlmRequest], str]]


class FakeVlmClient:
    """Replay scripted reply strings, VlmError instances or callables f(req) -> str.
    """

    def __init__(
        self,
        script: Sequence[ScriptItem],
        endpoint: str = "luna_dev",
        *,
        counts: Optional[TokenCounts] = None,
        wall_s: float = 0.01,
    ) -> None:
        self._script = list(script)
        self._ep = ENDPOINTS[endpoint]
        self._counts = counts
        self._wall_s = wall_s
        self._n = 0
        self.requests: List[VlmRequest] = []

    @property
    def endpoint(self) -> Endpoint:
        return self._ep

    def complete(self, req: VlmRequest) -> VlmResult:
        self.requests.append(req)
        if self._n >= len(self._script):
            raise AssertionError("脚本用完")
        item = self._script[self._n]
        self._n += 1

        if isinstance(item, Exception):
            raise item

        text = item(req) if callable(item) else item

        counts = self._counts if self._counts is not None else self._estimate_counts(req, text)
        list_usd, billed_usd = cost_usd(self._ep.provider, self._ep.model, counts, self._ep.flex)
        request_id = f"fake-{self._n}"
        usage = UsageRecord(
            provider=self._ep.provider,
            model=self._ep.model,
            input_tokens=counts.input_tokens,
            cached_tokens=counts.cached_tokens,
            output_tokens=counts.output_tokens,
            reasoning_tokens=counts.reasoning_tokens,
            list_usd=list_usd,
            billed_usd=billed_usd,
            request_id=request_id,
        )
        return VlmResult(
            text=text,
            usage=usage,
            counts=counts,
            endpoint=self._ep.name,
            model_reported=self._ep.model,
            response_id=request_id,
            service_tier="flex" if self._ep.flex else "default",
            finish_reason="stop",
            server_ms=None,
            wall_s=self._wall_s,
            retries=0,
            reasoning_text="",
        )

    def _estimate_counts(self, req: VlmRequest, text: str) -> TokenCounts:
        """Estimate billing TokenCounts from request content when no explicit counts are supplied.
        """
        input_tokens = estimate_text_tokens(req.system)
        for part in req.parts:
            if isinstance(part, TextPart):
                input_tokens += estimate_text_tokens(part.text)
            elif isinstance(part, ImagePart):
                if self._ep.provider == "dashscope":
                    input_tokens += qwen_image_tokens(part.width, part.height)
                else:
                    input_tokens += openai_image_tokens(part.width, part.height)
        return TokenCounts(
            input_tokens=input_tokens,
            cached_tokens=0,
            cache_write_tokens=0,
            output_tokens=estimate_text_tokens(text),
            reasoning_tokens=0,
        )
