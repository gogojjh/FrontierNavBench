"""VLM price tables, billing formulas and image/text token estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Price:
    input: float  # List USD per million input tokens
    cached_input: float  # USD per million cached-input tokens
    output: float  # USD per million output tokens, including reasoning


@dataclass(frozen=True)
class TokenCounts:
    input_tokens: int  # Provider total input tokens, including cache reads and writes
    cached_tokens: int  # Cached-input tokens read from existing cache, excluding new writes
    cache_write_tokens: int  # Newly cached input tokens for OpenAI; always zero for qwen
    output_tokens: int  # Total output tokens, including reasoning
    reasoning_tokens: int  # Reasoning subset of output tokens; recorded without separate billing


# Recorded OpenAI pricing (2026-10-20): Flex = standard * 0.5; cache writes = input * 1.25
OPENAI_PRICES: Dict[str, Price] = {
    "gpt-6-astra": Price(input=10.0, cached_input=1.0, output=50.0),
    "gpt-6-luna": Price(input=0.10, cached_input=0.01, output=0.50),
}
OPENAI_FLEX_FACTOR = 0.5
OPENAI_CACHE_WRITE_FACTOR = 1.25

# qwen3.8-max-0902 starts from international qwen-max estimates, calibrated to account billing
# Implicit cache-hit price = input price * 0.2
DASHSCOPE_PRICES: Dict[str, Price] = {
    "qwen3.8-max-0902": Price(input=1.2, cached_input=0.24, output=6.0),
}

# Image-token formula constants
OPENAI_PATCH_PX = 32  # gpt-6 family detail=high patch edge length, pixels
OPENAI_PATCH_MULT = 1.2
QWEN_PATCH_PX = 28  # Qwen-VL patch edge length, pixels
QWEN_IMAGE_EXTRA = 2  # Fixed additive term in the Qwen-VL image-token formula

BYTES_PER_TOKEN = 4  # Replay-only text estimate: bytes / 4; never used for actual billing


def cost_usd(provider: str, model: str, counts: TokenCounts, flex: bool) -> Tuple[float, float]:
    """Compute (list_usd, billed_usd) for one provider/model call.

    For OpenAI, uncached = input - cached - cache_write; negative counts raise
    ValueError. Sum the four standard-price components and multiply by 0.5 for
    Flex billing, otherwise 1.0. For DashScope, charge uncached input, cached
    input and output separately; billed equals list price and flex=True raises.
    Unknown provider/model combinations raise KeyError.
    """
    if provider == "openai":
        price = OPENAI_PRICES[model]
        uncached = counts.input_tokens - counts.cached_tokens - counts.cache_write_tokens
        if uncached < 0:
            raise ValueError("uncached < 0:cached_tokens + cache_write_tokens 超过了 input_tokens")
        list_usd = (
            uncached * price.input
            + counts.cached_tokens * price.cached_input
            + counts.cache_write_tokens * price.input * OPENAI_CACHE_WRITE_FACTOR
            + counts.output_tokens * price.output
        ) / 1e6
        billed_usd = list_usd * (OPENAI_FLEX_FACTOR if flex else 1.0)
        return list_usd, billed_usd

    if provider == "dashscope":
        if flex:
            raise ValueError("dashscope 不支持 flex 档")
        price = DASHSCOPE_PRICES[model]
        uncached = counts.input_tokens - counts.cached_tokens
        if uncached < 0:
            raise ValueError("uncached < 0:cached_tokens 超过了 input_tokens")
        list_usd = (
            uncached * price.input
            + counts.cached_tokens * price.cached_input
            + counts.output_tokens * price.output
        ) / 1e6
        return list_usd, list_usd

    raise KeyError(f"未知服务商: {provider}")


def openai_image_tokens(w: int, h: int) -> int:
    """Estimate gpt-6 detail="high" image tokens as rounded 32-pixel patch count times 1.2."""
    patches_w = math.ceil(w / OPENAI_PATCH_PX)
    patches_h = math.ceil(h / OPENAI_PATCH_PX)
    return math.ceil(patches_w * patches_h * OPENAI_PATCH_MULT)


# Smoke-measured prompt-token differences for one extra image; estimate unlisted sizes by formula
QWEN_MEASURED_IMAGE_TOKENS = {(640, 480): 469, (320, 240): 82}


def qwen_image_tokens(w: int, h: int) -> int:
    """Use measured Qwen image-token counts for known sizes, otherwise the Qwen-VL formula."""
    if (w, h) in QWEN_MEASURED_IMAGE_TOKENS:
        return QWEN_MEASURED_IMAGE_TOKENS[(w, h)]
    return math.ceil(w / QWEN_PATCH_PX) * math.ceil(h / QWEN_PATCH_PX) + QWEN_IMAGE_EXTRA


def estimate_text_tokens(text: str) -> int:
    """Estimate text tokens as bytes / 4 for replay pricing only, never actual billing."""
    return math.ceil(len(text.encode("utf-8")) / BYTES_PER_TOKEN)
