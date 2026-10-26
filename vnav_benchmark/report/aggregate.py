"""Aggregate RunRow values by method, difficulty and paired endpoint differences.

The main table uses scored first runs (repeat 0). SR and SPL average all scored episodes; PLR
and arrival time average successful episodes only. Decision latency is the median of episode
medians, with the median episode p95 in brackets. Paired confidence intervals bootstrap endpoint
pairs.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from vnav_benchmark.metrics import SCORED_TERMINATIONS, bootstrap_ci
from vnav_benchmark.report.results_io import REPORT_METHOD_IDS, RunRow
from vnav_benchmark.vlm.client import ENDPOINTS

MAIN_REPEAT = 0  # Repeat index used by the main table.

# Paired comparisons: (A, B, question); compare VLM methods separately for Astra and Qwen.
CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ("B0", "M1", "map density (point-goal planner)"),
    ("M2", "M3", "map density (VLM, pose-tracked)"),
    ("M4", "M5", "map density (VLM, initial localization only)"),
    ("B0", "M2", "planner: SRU vs VLM (dense map)"),
    ("M1", "M3", "planner: SRU vs VLM (sparse map)"),
    ("M2", "M4", "pose tracking during navigation (dense)"),
    ("M3", "M5", "pose tracking during navigation (sparse)"),
    ("M4", "M6", "prior map vs no map (dense)"),
    ("M5", "M6", "prior map vs no map (sparse)"),
)
PTGOAL_CODES = ("B0", "M1")  # Methods without a model suffix.


class ReportInputError(ValueError):
    """Invalid table inputs, such as mixed prompt fingerprints or development endpoints."""


def scored(rows: Iterable[RunRow], repeat: Optional[int] = MAIN_REPEAT) -> List[RunRow]:
    """Return scored episodes; repeat=None includes every repeat."""
    return [
        r
        for r in rows
        if r.scored
        and r.termination in SCORED_TERMINATIONS
        and (repeat is None or r.repeat == repeat)
    ]


def validate(rows: Sequence[RunRow]) -> None:
    """Require one VLM prompt fingerprint and reportable endpoints; reject development endpoints."""
    vlm_rows = [r for r in rows if r.vlm is not None]
    prints = sorted({r.fingerprint for r in vlm_rows})
    if len(prints) > 1:
        raise ReportInputError(f"看图方法混了 {len(prints)} 个提示词指纹: {prints}")
    for r in vlm_rows:
        ep = ENDPOINTS.get(r.endpoint)
        if ep is None or not ep.table_ok:
            raise ReportInputError(f"{r.method_id}/{r.episode_id} 用的端点 {r.endpoint!r} 不能进表")


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


@dataclass(frozen=True)
class MethodStats:
    method_id: str
    n: int
    sr: Optional[float]  # 0–1
    sr_loose: Optional[float]
    spl: Optional[float]
    plr: Optional[float]  # Successful episodes only.
    time_s: Optional[float]  # Arrival time for successful episodes.
    decisions: Optional[float]
    calls: Optional[float]
    act_med_s: Optional[float]
    act_p95_s: Optional[float]
    act_max_s: Optional[float]
    compute_per_sim: Optional[float]
    list_usd: Optional[float]  # Mean list-price cost per episode.
    billed_usd_total: float


def method_stats(rows: Sequence[RunRow]) -> MethodStats:
    ok = [r for r in rows if r.success]
    maxes = [r.act_max_s for r in rows if r.act_max_s is not None]
    return MethodStats(
        method_id=rows[0].method_id if rows else "",
        n=len(rows),
        sr=_mean([float(r.success) for r in rows]),
        sr_loose=_mean([float(r.success_loose) for r in rows]),
        spl=_mean([r.spl for r in rows]),
        plr=_mean([r.plr for r in ok if r.plr is not None]),
        time_s=_mean([r.sim_time_s for r in ok]),
        decisions=_mean([float(r.decisions) for r in rows]),
        calls=_mean([float(r.calls) for r in rows]),
        act_med_s=_median([r.act_med_s for r in rows]),
        act_p95_s=_median([r.act_p95_s for r in rows]),
        act_max_s=max(maxes) if maxes else None,
        compute_per_sim=_mean([r.compute_per_sim for r in rows if r.compute_per_sim is not None]),
        list_usd=_mean([r.list_usd for r in rows]),
        billed_usd_total=sum(r.billed_usd for r in rows),
    )


def group(rows: Iterable[RunRow], key: Callable[[RunRow], str]) -> Dict[str, List[RunRow]]:
    out: Dict[str, List[RunRow]] = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def common_episodes(rows: Sequence[RunRow]) -> Set[str]:
    """Return endpoint pairs with scored results for every configuration present.

    The main table uses common pairs because the original Astra batch covered only four pairs.
    """
    per = [set(r.episode_id for r in rs) for rs in group(rows, lambda r: r.method_id).values()]
    return set.intersection(*per) if per else set()


def methods_beyond(rows: Sequence[RunRow], common: Set[str]) -> List[str]:
    """Return configurations with extra pairs for the supplement, in table order."""
    extra = {r.method_id for r in rows if r.episode_id not in common}
    return [m for m in ordered_methods(rows) if m in extra]


def ordered_bands(rows: Iterable[RunRow]) -> List[str]:
    """Order difficulty tiers by mean shortest-path length; tier names come from the episode set."""
    groups = group(rows, lambda r: r.band)
    return sorted(groups, key=lambda b: statistics.fmean(r.shortest_m for r in groups[b]))


def ordered_methods(rows: Iterable[RunRow]) -> List[str]:
    """Order rows as B0, M1, M2 through M6 (Astra first), followed by other methods."""
    present = {r.method_id for r in rows}
    extra = sorted(present - set(REPORT_METHOD_IDS))
    return [m for m in REPORT_METHOD_IDS if m in present] + extra


def by_method(rows: Sequence[RunRow]) -> Dict[str, MethodStats]:
    groups = group(rows, lambda r: r.method_id)
    return {m: method_stats(groups[m]) for m in ordered_methods(rows)}


def repeat_spread(rows: Sequence[RunRow], method_id: str) -> Optional[Tuple[float, float, int]]:
    """Return mean SR, standard deviation and repeat count; return None for a single repeat."""
    per_rep = group(
        scored([r for r in rows if r.method_id == method_id], None), lambda r: str(r.repeat)
    )
    if len(per_rep) < 2:
        return None
    srs = [statistics.fmean(float(r.success) for r in rs) for rs in per_rep.values()]
    return statistics.fmean(srs), statistics.stdev(srs), len(srs)


@dataclass(frozen=True)
class PairedDiff:
    a: str
    b: str
    n_pairs: int
    d_sr: Optional[float]  # A minus B, in the range 0-1.
    d_sr_ci: Tuple[float, float]
    d_spl: Optional[float]
    d_spl_ci: Tuple[float, float]
    # Difference under the loose success criterion (2 m / 90 deg).
    d_sr_loose: Optional[float] = None
    d_sr_loose_ci: Tuple[float, float] = (float("nan"), float("nan"))


def paired_diff(rows_a: Sequence[RunRow], rows_b: Sequence[RunRow], a: str, b: str) -> PairedDiff:
    """Compare scored common endpoint pairs with a paired-bootstrap 95% confidence interval."""
    ia = {r.episode_id: r for r in rows_a}
    ib = {r.episode_id: r for r in rows_b}
    common = sorted(set(ia) & set(ib))
    dsr = [float(ia[e].success) - float(ib[e].success) for e in common]
    dspl = [ia[e].spl - ib[e].spl for e in common]
    dloose = [float(ia[e].success_loose) - float(ib[e].success_loose) for e in common]
    nan = (float("nan"), float("nan"))
    return PairedDiff(
        a,
        b,
        len(common),
        _mean(dsr),
        bootstrap_ci(dsr) if dsr else nan,
        _mean(dspl),
        bootstrap_ci(dspl) if dspl else nan,
        _mean(dloose),
        bootstrap_ci(dloose) if dloose else nan,
    )


def contrast_pairs(rows: Sequence[RunRow]) -> List[Tuple[str, str, str]]:
    """Expand comparisons into method IDs and descriptions, skipping unavailable pairs."""
    by_code: Dict[Tuple[str, Optional[str]], str] = {(r.code, r.vlm): r.method_id for r in rows}
    out: List[Tuple[str, str, str]] = []
    for ca, cb, what in CONTRASTS:
        for model in ("astra", "qwen"):
            ma = by_code.get((ca, None if ca in PTGOAL_CODES else model))
            mb = by_code.get((cb, None if cb in PTGOAL_CODES else model))
            if ma and mb and (ma, mb) not in [(x, y) for x, y, _ in out]:
                out.append((ma, mb, what))
    # Check whether the conclusion changes with the model.
    for code in ("M2", "M3", "M4", "M5", "M6"):
        ma, mb = by_code.get((code, "astra")), by_code.get((code, "qwen"))
        if ma and mb:
            out.append((ma, mb, f"astra vs qwen ({code})"))
    return out
