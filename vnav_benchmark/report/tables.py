"""Build main, tier, latency, VLM diagnostic, paired-difference and repeat tables.

Each table first produces shared string cells (Table), then renders Markdown, LaTeX booktabs and
CSV with identical numbers. All table labels are English for paper use.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from vnav_benchmark import metrics
from vnav_benchmark.fairness import parse_method_id
from vnav_benchmark.methods import base_topo
from vnav_benchmark.report import aggregate as agg
from vnav_benchmark.report.results_io import METHOD_CODES, RunRow
from vnav_benchmark.vlm.client import THINKING_NOTE

DASH = "–"  # No data.
MAP_LABEL = {"dense": "dense", "sparse": "sparse", None: "none"}
LOC_LABEL = {"contloc": "continuous", "initloc": "initial only", "noloc": "none"}
PLANNER_LABEL = {"ptgoal": "SRU (point goal)", "imggoal": "VLM (image goal)"}
VLM_LABEL = {"astra": "gpt-6-astra", "qwen": "qwen3.8-max", None: DASH}


@dataclass
class Table:
    key: str  # Output filename stem.
    title: str
    columns: List[str]
    rows: List[List[str]]
    notes: List[str] = field(default_factory=list)


# Reporting criteria text.


def loose_label() -> str:
    """Return the loose-success column label using evaluator constants (2 m / 90 deg)."""
    dist = metrics.SUCCESS_DIST_LOOSE_M
    yaw = getattr(metrics, "SUCCESS_YAW_LOOSE_DEG", None)
    return f"SR@{dist:g}m" + ("" if yaw is None else f"/{yaw:g}°")


def _arrival_note() -> str:
    """Return pose-based arrival thresholds for B0/M1/M2/M3 from base_topo constants."""
    dist = base_topo.FINAL_POS_TOL_M
    yaw = math.degrees(base_topo.FINAL_YAW_TOL_RAD)
    return (
        f"Arrival judged by pose (B0, M1, M2, M3): within {dist:g} m and {yaw:g}° of the goal; "
        "M4–M6 stop when the VLM says done."
    )


def _budget_note(rows: Sequence[RunRow]) -> str:
    budgets = sorted({r.budget_s for r in rows if r.budget_s})
    shown = " / ".join(f"{b:g}" for b in budgets) if budgets else "?"
    return f"Budget: {shown} s of simulated time per episode (time spent computing is free)."


# Number formatting.


def _pct(x: Optional[float]) -> str:
    return DASH if x is None else f"{100 * x:.1f}"


def _f(x: Optional[float], nd: int) -> str:
    return DASH if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _signed(x: Optional[float], nd: int) -> str:
    return DASH if x is None else f"{x:+.{nd}f}"


def _signed_pct(x: Optional[float]) -> str:
    return DASH if x is None else f"{100 * x:+.1f}"


def _ci_pct(ci: Sequence[float]) -> str:
    if any(math.isnan(v) for v in ci):
        return DASH
    return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def _ci(ci: Sequence[float], nd: int) -> str:
    if any(math.isnan(v) for v in ci):
        return DASH
    return f"[{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]"


def _lat(med: Optional[float], p95: Optional[float]) -> str:
    return DASH if med is None else f"{med:.1f} [{_f(p95, 1)}]"


def _label(method_id: str) -> str:
    code = METHOD_CODES["-".join(method_id.split("-")[:3])]
    vlm = parse_method_id(method_id).vlm
    return code if vlm is None else f"{code}-{vlm}"


# Individual tables.


def main_table(
    rows: Sequence[RunRow],
    key: str = "main",
    title: str = "Main results (episode pairs run by every configuration, first run)",
    extra_notes: Sequence[str] = (),
) -> Table:
    """Build the main table from caller-selected pairs; build_all uses pairs common to all
    configurations.
    """
    stats = agg.by_method(rows)
    n_pairs = len({r.episode_id for r in rows})
    out: List[List[str]] = []
    for mid, s in stats.items():
        p = parse_method_id(mid)
        out.append(
            [
                _label(mid),
                mid,
                MAP_LABEL[p.map_variant],
                LOC_LABEL[p.loc_mode],
                PLANNER_LABEL[p.planner],
                VLM_LABEL[p.vlm],
                str(s.n),
                _pct(s.sr),
                _pct(s.sr_loose),
                _f(s.spl, 3),
                _f(s.plr, 2),
                _f(s.time_s, 1),
                _f(s.calls, 1),
                _lat(s.act_med_s, s.act_p95_s),
                _f(s.list_usd, 3),
            ]
        )
    return Table(
        key,
        title,
        [
            "ID",
            "Method",
            "Map",
            "Self-loc",
            "Planner",
            "VLM",
            "n",
            "SR@0.5m/45° (%)",
            f"{loose_label()} (%)",
            "SPL",
            "PLR",
            "Time (s)",
            "VLM calls",
            "Decision latency med [p95] (s)",
            "List $ / ep",
        ],
        out,
        [
            "SR: success rate (declared done within 0.5 m and 45° of the goal pose); "
            f"{loose_label()} is the looser criterion. PLR and Time are averaged over successful "
            "episodes only.",
            _budget_note(rows),
            "Decision latency: wall-clock per act(), median over episodes of the per-episode "
            "median [median of per-episode p95]. The simulator is paused while methods compute.",
            f"{n_pairs} episode pairs × 1 run; the SR granularity is "
            f"{100 / max(n_pairs, 1):g} percentage points.",
            "VLM reasoning: " + THINKING_NOTE + ".",
            "List $: list price per episode (standard tier). Qwen is billed to a separate "
            "DashScope account; its prices are provisional until checked against the console.",
            _arrival_note(),
            *extra_notes,
        ],
    )


def band_table(rows: Sequence[RunRow]) -> Table:
    groups = agg.group(rows, lambda r: r.method_id)
    bands = agg.ordered_bands(rows)
    out: List[List[str]] = []
    for mid in agg.ordered_methods(rows):
        cells = [_label(mid)]
        for b in bands:
            sub = [r for r in groups[mid] if r.band == b]
            s = agg.method_stats(sub) if sub else None
            cells += [str(len(sub)), _pct(s.sr if s else None), _f(s.spl if s else None, 3)]
        out.append(cells)
    cols = ["ID"]
    for b in bands:
        cols += [f"{b} n", f"{b} SR (%)", f"{b} SPL"]
    return Table(
        "by_difficulty", "Results by difficulty (shortest path 3–6 / 6–12 / 12–25 m)", cols, out
    )


def latency_table(rows: Sequence[RunRow]) -> Table:
    groups = agg.group(rows, lambda r: f"{r.method_id}|{r.service_tier or DASH}")
    out: List[List[str]] = []
    for mid in agg.ordered_methods(rows):
        for key in sorted(k for k in groups if k.startswith(mid + "|")):
            sub = groups[key]
            s = agg.method_stats(sub)
            out.append(
                [
                    _label(mid),
                    key.split("|")[1],
                    str(s.n),
                    _f(s.act_med_s, 2),
                    _f(s.act_p95_s, 2),
                    _f(s.act_max_s, 2),
                    _f(s.compute_per_sim, 2),
                    _f(agg._median([r.vlm_api_med_s for r in sub]), 2),
                    _f(agg._median([r.server_ms_med for r in sub]), 0),
                ]
            )
    return Table(
        "latency",
        "Decision latency",
        [
            "ID",
            "Tier",
            "n",
            "act() median (s)",
            "act() p95 (s)",
            "act() max (s)",
            "Compute / sim time",
            "VLM API median (s)",
            "Server median (ms)",
        ],
        out,
        [
            "Compute / sim time < 1 means the method could keep up in real time at this decision "
            "rate.",
            "Flex-tier wall-clock includes queueing at the provider; see the standard-tier "
            "probe for model-only latency.",
        ],
    )


def vlm_diag_table(rows: Sequence[RunRow]) -> Table:
    vlm_rows = [r for r in rows if r.vlm is not None]
    groups = agg.group(vlm_rows, lambda r: r.method_id)
    out: List[List[str]] = []
    for mid in agg.ordered_methods(vlm_rows):
        sub = groups[mid]
        n = len(sub)
        calls = sum(r.calls for r in sub)
        inp = sum(r.input_tokens for r in sub)
        out.append(
            [
                _label(mid),
                str(n),
                str(sum(r.premature_done for r in sub)),
                str(sum(r.ever_within and not r.success for r in sub)),
                _f(sum(r.invalid_outputs for r in sub) / n if n else None, 2),
                _pct(sum(r.cached_tokens for r in sub) / inp if inp else None),
                _f(sum(r.reasoning_tokens for r in sub) / calls if calls else None, 0),
                _f(sum(r.output_tokens for r in sub) / calls if calls else None, 0),
                str(sum(r.vlm_done_ignored for r in sub)) if sub[0].done_judge == "pose" else DASH,
            ]
        )
    return Table(
        "vlm_diagnostics",
        "VLM diagnostics",
        [
            "ID",
            "n",
            "Premature done",
            "Reached, not stopped",
            "Invalid replies / ep",
            "Cache hit (%)",
            "Thinking tok / call",
            "Output tok / call",
            "VLM done not used",
        ],
        out,
        [
            "VLM done not used: M2 / M3 end by pose (within 0.5 m and 30°, as B0 / M1); "
            "the VLM's own done claims are only counted.",
            "Premature done: declared done outside the success region. Reached, not stopped: was "
            "inside the success region at some point but the episode failed.",
        ],
    )


def nav_diag_table(rows: Sequence[RunRow]) -> Table:
    """Tabulate stuck/proximity counts and means; these counters do not change success, PLR or SPL.
    """
    groups = agg.group(rows, lambda r: r.method_id)
    out: List[List[str]] = []
    for mid in agg.ordered_methods(rows):
        sub = groups[mid]
        n = len(sub)
        stuck = sum(r.stuck_events for r in sub)
        prox = sum(r.proximity_events for r in sub)
        out.append([
            _label(mid), str(n), str(stuck), _f(stuck / n if n else None, 2),
            str(prox), _f(prox / n if n else None, 2),
        ])  # fmt: skip
    return Table(
        "nav_diagnostics",
        "Navigation diagnostics (counts only, do not affect success / PLR / SPL)",
        ["ID", "n", "Stuck events", "Stuck / ep", "Proximity events", "Proximity / ep"],
        out,
        [
            "Stuck event: 15 s window with < 0.3 m net displacement (metrics.StuckMonitor); "
            "counted whether or not the episode later recovered.",
            "Proximity event: the depth frame's center crop has median depth < 0.3 m at a "
            "decision step (metrics.is_proximity); may under-count contact outside that crop.",
        ],
    )


def contrast_table(rows: Sequence[RunRow]) -> Table:
    groups = agg.group(rows, lambda r: r.method_id)
    out: List[List[str]] = []
    for ma, mb, what in agg.contrast_pairs(rows):
        d = agg.paired_diff(groups[ma], groups[mb], ma, mb)
        out.append(
            [
                f"{_label(ma)} − {_label(mb)}",
                what,
                str(d.n_pairs),
                _signed_pct(d.d_sr),
                _ci_pct(d.d_sr_ci),
                _signed(d.d_spl, 3),
                _ci(d.d_spl_ci, 3),
                _signed_pct(d.d_sr_loose),
                _ci_pct(d.d_sr_loose_ci),
            ]
        )
    return Table(
        "contrasts",
        "Paired contrasts (same episode pairs, both scored)",
        [
            "Contrast",
            "Question",
            "Pairs",
            "ΔSR (pt)",
            "95% CI",
            "ΔSPL",
            "95% CI",
            f"Δ{loose_label()} (pt)",
            "95% CI",
        ],  # fmt: skip
        out,
        [
            "95% CI: bootstrap over episode pairs (10,000 resamples). B0 − M2 and M1 − M3 swap the "
            "planner (SRU on depth vs VLM on RGB), which also changes the sensing and the "
            "decision rate."
        ],
    )


def repeats_table(all_rows: Sequence[RunRow]) -> Table:
    out: List[List[str]] = []
    for mid in agg.ordered_methods(all_rows):
        spread = agg.repeat_spread(all_rows, mid)
        if spread is not None:
            mean, std, k = spread
            out.append([_label(mid), str(k), f"{100 * mean:.1f} ± {100 * std:.1f}"])
    return Table(
        "repeats",
        "Run-to-run spread (methods run more than once)",
        ["ID", "Runs", "SR (%) mean ± std"],
        out,
    )


def build_all(all_rows: Sequence[RunRow], extra_notes: Sequence[str] = ()) -> List[Table]:
    """Build all tables after validating prompt fingerprints and endpoints.

    Append extra_notes to the main table notes.
    """
    agg.validate(all_rows)
    main = agg.scored(all_rows)
    common = agg.common_episodes(main)
    shared = [r for r in main if r.episode_id in common]
    extra = [r for r in main if r.method_id in agg.methods_beyond(main, common)]
    tables = [main_table(shared, extra_notes=extra_notes)]
    if extra:
        tables.append(main_table(
            extra, "supplementary",
            "Supplementary: configurations run on more episode pairs (all their pairs, first run)",
        ))  # fmt: skip
    return tables + [
        band_table(main),
        latency_table(main),
        vlm_diag_table(main),
        nav_diag_table(main),
        contrast_table(main),
        repeats_table(all_rows),
    ]


# Rendering.


def to_markdown(t: Table) -> str:
    lines = [
        f"### {t.title}",
        "",
        "| " + " | ".join(t.columns) + " |",
        "|" + "---|" * len(t.columns),
    ]
    lines += ["| " + " | ".join(r) + " |" for r in t.rows]
    if t.notes:
        lines += [""] + [f"*{n}*" for n in t.notes]
    return "\n".join(lines) + "\n"


_TEX_ESCAPE = {
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "$": r"\$",
    "°": r"$^\circ$",
    "±": r"$\pm$",
    "−": r"$-$",
    "–": "--",
    "Δ": r"$\Delta$",
}


def _tex(text: str) -> str:
    return "".join(_TEX_ESCAPE.get(ch, ch) for ch in text)


def to_latex(t: Table) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_tex(t.title)}}}",
        rf"\label{{tab:{t.key}}}",
        r"\begin{tabular}{" + "l" * len(t.columns) + "}",
        r"\toprule",
        " & ".join(_tex(c) for c in t.columns) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(_tex(c) for c in r) + r" \\" for r in t.rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if t.notes:
        lines.append(r"\par\footnotesize " + " ".join(_tex(n) for n in t.notes))
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def to_csv(t: Table) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(t.columns)
    writer.writerows(t.rows)
    return buf.getvalue()


def write_all(tables: Sequence[Table], out_dir: Path) -> Dict[str, Path]:
    """Write each table as .md, .tex and .csv plus combined tables.md; return name-to-Markdown
    paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for t in tables:
        for ext, fn in (("md", to_markdown), ("tex", to_latex), ("csv", to_csv)):
            (out_dir / f"{t.key}.{ext}").write_text(fn(t), encoding="utf-8")
        paths[t.key] = out_dir / f"{t.key}.md"
    (out_dir / "tables.md").write_text("\n".join(to_markdown(t) for t in tables), encoding="utf-8")
    return paths
