"""Pricing tests for image-token formulas, OpenAI/DashScope charges and invalid inputs."""

from __future__ import annotations

import math

import pytest

from frontier_nav_bench.vlm.pricing import (
    TokenCounts,
    cost_usd,
    openai_image_tokens,
    qwen_image_tokens,
)


def test_openai_image_tokens_640x480_and_320x240_are_pinned() -> None:
    assert openai_image_tokens(640, 480) == 360
    assert openai_image_tokens(320, 240) == 96


def test_qwen_image_tokens_measured_sizes_then_formula() -> None:
    # Use smoke-measured counts for 640x480 / 320x240; estimate other sizes by formula
    assert qwen_image_tokens(640, 480) == 469 and qwen_image_tokens(320, 240) == 82
    assert qwen_image_tokens(100, 100) == math.ceil(100 / 28) ** 2 + 2


def test_openai_cost_usd_hand_computed_with_cache_and_flex() -> None:
    # gpt-6-astra: input=10, cached input=1, output=50 USD per million tokens
    counts = TokenCounts(
        input_tokens=1000, cached_tokens=200, cache_write_tokens=100,
        output_tokens=50, reasoning_tokens=10,
    )
    uncached = 1000 - 200 - 100
    expected_list = (
        uncached * 10.0 + 200 * 1.0 + 100 * 10.0 * 1.25 + 50 * 50.0
    ) / 1e6

    list_usd, billed_usd = cost_usd("openai", "gpt-6-astra", counts, flex=False)
    assert list_usd == pytest.approx(expected_list)
    assert billed_usd == pytest.approx(expected_list)

    list_usd_flex, billed_usd_flex = cost_usd("openai", "gpt-6-astra", counts, flex=True)
    assert list_usd_flex == pytest.approx(expected_list)
    assert billed_usd_flex == pytest.approx(expected_list * 0.5)


def test_openai_cost_usd_negative_uncached_raises_value_error() -> None:
    counts = TokenCounts(
        input_tokens=100, cached_tokens=80, cache_write_tokens=50,
        output_tokens=0, reasoning_tokens=0,
    )
    with pytest.raises(ValueError):
        cost_usd("openai", "gpt-6-astra", counts, flex=False)


def test_dashscope_cost_usd_hand_computed() -> None:
    # qwen3.8-max-0902: input=1.2, cached_input=0.24, output=6.0
    counts = TokenCounts(
        input_tokens=1000, cached_tokens=200, cache_write_tokens=0,
        output_tokens=50, reasoning_tokens=0,
    )
    expected_list = ((1000 - 200) * 1.2 + 200 * 0.24 + 50 * 6.0) / 1e6
    list_usd, billed_usd = cost_usd("dashscope", "qwen3.8-max-0902", counts, flex=False)
    assert list_usd == pytest.approx(expected_list)
    assert billed_usd == pytest.approx(expected_list)  # DashScope does not support Flex; billed
    # equals list price


def test_dashscope_cost_usd_flex_true_raises_value_error() -> None:
    counts = TokenCounts(
        input_tokens=1000, cached_tokens=200, cache_write_tokens=0,
        output_tokens=50, reasoning_tokens=0,
    )
    with pytest.raises(ValueError):
        cost_usd("dashscope", "qwen3.8-max-0902", counts, flex=True)


def test_dashscope_cost_usd_cached_greater_than_input_raises_value_error() -> None:
    counts = TokenCounts(
        input_tokens=100, cached_tokens=150, cache_write_tokens=0,
        output_tokens=0, reasoning_tokens=0,
    )
    with pytest.raises(ValueError):
        cost_usd("dashscope", "qwen3.8-max-0902", counts, flex=False)


def test_cost_usd_unknown_model_raises_key_error() -> None:
    counts = TokenCounts(
        input_tokens=100, cached_tokens=0, cache_write_tokens=0,
        output_tokens=10, reasoning_tokens=0,
    )
    with pytest.raises(KeyError):
        cost_usd("openai", "no-such-model", counts, flex=False)
    with pytest.raises(KeyError):
        cost_usd("dashscope", "no-such-model", counts, flex=False)
