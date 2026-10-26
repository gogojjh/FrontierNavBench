"""BudgetGuard for API costs (evaluation layer, frozen after T-I0; plan sections 6.7 and 8).

The durable ``<campaign>/ledger.jsonl`` ledger records actual ``billed_usd`` for
paid calls. Each entry is fsynced so interrupted campaigns can resume accounting.
Before an episode, reject admission if spending plus 90 calls at the per-call
p95 cost exceeds the cap (90 seconds at 1 Hz). With fewer than 20 samples, use
the larger of the prior estimate and observed p95. Episode costs above $5 trigger
an unscored ``cost_guard`` termination. OpenAI and DashScope have separate ledgers
and caps; the approved OpenAI total is $100 and Qwen is accounted separately.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frontier_nav_bench.interfaces import UsageRecord

PROVIDER_CAPS_USD = {"openai": 100.0, "dashscope": 30.0}
"""Provider caps in billed USD: OpenAI $100; DashScope uses the $30 upper estimate.

The DashScope estimate follows plan section 8.
"""
EPISODE_CAP_USD = 5.0  # Per-episode cap; exceeding it triggers cost_guard
CALLS_PER_EPISODE_MAX = 90  # 90-second budget at a 1 Hz decision rate
PRIOR_CALL_USD = {"openai": 0.042, "dashscope": 0.007}  # Plan section 8 per-call priors for
# standard astra / qwen pricing
MIN_SAMPLES_FOR_P95 = 20
VLM_PROVIDER = {"astra": "openai", "qwen": "dashscope"}


class BudgetGuard:
    def __init__(
        self,
        ledger_path: Path,
        caps: Optional[Dict[str, float]] = None,
        episode_cap_usd: float = EPISODE_CAP_USD,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.caps = dict(caps or PROVIDER_CAPS_USD)
        self.episode_cap_usd = episode_cap_usd
        self._rows: List[Dict[str, Any]] = []
        if self.ledger_path.is_file():
            with open(self.ledger_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self._rows.append(json.loads(line))

    def spent(self, provider: str) -> float:
        return float(sum(r["billed_usd"] for r in self._rows if r["provider"] == provider))

    def p95_call(self, provider: str) -> float:
        costs = [r["billed_usd"] for r in self._rows if r["provider"] == provider]
        prior = PRIOR_CALL_USD.get(provider, 0.0)
        if len(costs) < MIN_SAMPLES_FOR_P95:
            return max([prior] + costs)
        return float(np.percentile(costs, 95))

    def can_start(self, provider: Optional[str]) -> Tuple[bool, str]:
        """Check admission before an episode; free methods with provider=None are always allowed."""
        if provider is None:
            return True, ""
        cap = self.caps.get(provider)
        if cap is None:
            return False, f"{provider} 没设上限"
        projected = self.spent(provider) + CALLS_PER_EPISODE_MAX * self.p95_call(provider)
        if projected > cap:
            return False, (
                f"{provider} 已花 ${self.spent(provider):.2f}"
                f" + 90 × p95 ${self.p95_call(provider):.4f}"
                f" = ${projected:.2f} > 上限 ${cap:.0f}"
            )
        return True, ""

    def record(self, row: Dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._rows.append(row)

    def sink(self, method_id: str, token: str, repeat: int) -> EpisodeUsage:
        return EpisodeUsage(self, method_id, token, repeat)


class EpisodeUsage:
    """Episode UsageSink that writes each record immediately and accumulates episode usage."""

    def __init__(
        self, guard: Optional[BudgetGuard], method_id: str, token: str, repeat: int
    ) -> None:
        self.guard = guard
        self.method_id, self.token, self.repeat = method_id, token, repeat
        self.records: List[UsageRecord] = []

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)
        if self.guard is not None:
            row = {**asdict(record), "t": time.time(), "method_id": self.method_id,
                   "token": self.token, "repeat": self.repeat}  # fmt: skip
            self.guard.record(row)

    def episode_billed_usd(self) -> float:
        return float(sum(r.billed_usd for r in self.records))

    def over_cap(self) -> bool:
        cap = self.guard.episode_cap_usd if self.guard is not None else EPISODE_CAP_USD
        return self.episode_billed_usd() > cap

    def summary(self) -> Dict[str, Any]:
        fields = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
        out: Dict[str, Any] = {"calls": len(self.records)}
        out.update({f: int(sum(getattr(r, f) for r in self.records)) for f in fields})
        out["list_usd"] = float(sum(r.list_usd for r in self.records))
        out["billed_usd"] = self.episode_billed_usd()
        by: Dict[str, Dict[str, float]] = {}
        for r in self.records:
            slot = by.setdefault(r.provider, {"calls": 0, "billed_usd": 0.0, "list_usd": 0.0})
            slot["calls"] += 1
            slot["billed_usd"] += r.billed_usd
            slot["list_usd"] += r.list_usd
        out["by_provider"] = by
        out["models"] = sorted({r.model for r in self.records})
        return out
