"""Compare decision computation time using one stacked horizontal bar per configuration.

Read act_wall_s and debug.latency from each episode's steps.jsonl. act_wall_s spans observation
receipt through action output. VLM vlm_api_s measures the API round trip; server_ms measures
Astra server processing (unavailable for Qwen, recorded as zero). Remaining act time covers
prompt/image preparation, memory and logging. SRU controller_s includes inference and sidecar
IPC; the remainder is act_wall_s minus controller_s.

The real-time ratio is total compute seconds divided by total simulated seconds, using the
executed hold_s for each step. Ratios above one indicate computation slower than real time. Run:
python -m frontier_nav_bench.report.latency_figure <campaign> <output-directory>
This writes latency_by_method.png/.pdf and latency_by_method.md/.csv.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from frontier_nav_bench.fairness import parse_method_id
from frontier_nav_bench.report import aggregate as agg
from frontier_nav_bench.report import figures, tables
from frontier_nav_bench.report.colors import code_color_hex
from frontier_nav_bench.report.results_io import REPORT_METHOD_IDS

OUT_NAME = "latency_by_method"
# Server-segment shade as a fraction of black mixed with the configuration color.
SERVER_SHADE = 0.55
OTHER_ALPHA = 0.35  # Opacity of the remaining-overhead segment.


@dataclass
class MethodLatency:
    method_id: str
    act_s: np.ndarray  # Wall-clock seconds per act() call.
    core_s: np.ndarray  # Main per-step work: VLM API round trip or SRU inference.
    # Server processing time, available only for Astra; zero for other providers.
    server_s: np.ndarray
    sim_s: np.ndarray  # Simulated seconds represented by each step.
    episodes: int

    @property
    def vlm(self) -> bool:
        return parse_method_id(self.method_id).vlm is not None

    @property
    def has_server(self) -> bool:
        return bool(np.any(self.server_s > 0))


def load_latency(campaign_dir: Path) -> List[MethodLatency]:
    """Return every step in scored first runs (r0), in table order."""
    per: Dict[str, Dict[str, list]] = {}
    for run_json in sorted(campaign_dir.glob(f"runs/*/*/r{agg.MAIN_REPEAT}/run.json")):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        if run.get("status") != "scored":
            continue
        d = per.setdefault(run["method_id"], {"act": [], "core": [], "server": [], "sim": [],
                                              "n": 0})  # fmt: skip
        d["n"] += 1
        for line in (run_json.parent / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            step = json.loads(line)
            lat = (step.get("debug") or {}).get("latency") or {}
            d["act"].append(float(step["act_wall_s"]))
            d["core"].append(float(lat.get("vlm_api_s", lat.get("controller_s", 0.0))))
            d["server"].append(float(lat.get("server_ms", 0.0)) / 1000.0)
            d["sim"].append(float(step["applied"]["hold_s"]))
    return [
        MethodLatency(mid, np.array(per[mid]["act"]), np.array(per[mid]["core"]),
                      np.array(per[mid]["server"]), np.array(per[mid]["sim"]), per[mid]["n"])
        for mid in REPORT_METHOD_IDS if mid in per
    ]  # fmt: skip


def _label(mid: str) -> str:
    return tables._label(mid)


def _code(mid: str) -> str:
    return _label(mid).split("-")[0]


def _rt_ratio(m: MethodLatency) -> float:
    return float(m.act_s.sum() / m.sim_s.sum())


def _sec(x: float) -> str:
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def _breakdown(m: MethodLatency) -> str:
    """Label each bar with its mean total and segment means; log-axis segment widths hide
    proportions.
    """
    act, core, server = m.act_s.mean(), m.core_s.mean(), m.server_s.mean()
    other = max(act - core, 0.0)
    if not m.vlm:
        return f"{_sec(act)} s = {_sec(core)} SRU + {_sec(other)} other"
    if m.has_server:
        return (f"{_sec(act)} s = {_sec(server)} provider + {_sec(core - server)} API rest "
                f"+ {_sec(other)} other")  # fmt: skip
    return f"{_sec(act)} s = {_sec(core)} API + {_sec(other)} other (provider time n/a)"


def latency_table(ms: Sequence[MethodLatency]) -> tables.Table:
    rows = []
    for m in ms:
        rows.append([
            _label(m.method_id), m.method_id, f"{m.act_s.mean():.3f}",
            f"{np.median(m.act_s):.3f}", f"{np.percentile(m.act_s, 95):.3f}",
            f"{m.act_s.max():.3f}", f"{len(m.act_s) / m.episodes:.1f}",
            f"{m.act_s.sum() / m.episodes:.1f}", f"{np.median(m.sim_s):g}",
            f"{_rt_ratio(m):.2f}",
        ])  # fmt: skip
    return tables.Table(
        OUT_NAME,
        "Per-decision compute time (all steps of all episodes, first run)",
        ["ID", "Method", "Mean act (s)", "Median (s)", "p95 (s)", "Max (s)", "Decisions / ep",
         "Compute / ep (s)", "Sim per step (s)", "Real-time ratio"],  # fmt: skip
        rows,
        [
            "act: wall-clock seconds of one act() call, from receiving the observation to "
            "returning the command. The simulator is paused meanwhile, so this time is not "
            "charged to the 90 s budget.",
            "Real-time ratio = total compute time / total simulated time of the executed steps; "
            "above 1 the same algorithm could not keep up in real time on a robot.",
            "astra runs on the Flex tier: its wall-clock includes queueing at the provider.",
        ],
    )


def latency_bars(
    ms: Sequence[MethodLatency], out_stem: Path,
    title: Optional[str] = None, caption: Optional[str] = None,
    compact_text: bool = False, legend_ncol: int = 2,
) -> Path:
    figures._style()
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    ms = list(ms)[::-1]  # Bar order from top to bottom follows the table rows.
    fs = 1.3 if compact_text else 1.0  # Enlarge blog text and reserve room for a single-row legend.
    figsize = (14, 0.6 * len(ms) + 2.0) if compact_text else (11.5, 0.5 * len(ms) + 1.6)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(ms))
    for i, m in enumerate(ms):
        color = code_color_hex(_code(m.method_id))
        act, core, server = m.act_s.mean(), m.core_s.mean(), m.server_s.mean()
        dark = mcolors.to_hex(np.array(mcolors.to_rgb(color)) * (1 - SERVER_SHADE))
        if m.has_server:
            ax.barh(i, server, color=dark, edgecolor="white", height=0.7)
            ax.barh(i, core - server, left=server, color=color, edgecolor="white", height=0.7)
        else:
            ax.barh(i, core, color=color, edgecolor="white", height=0.7)
        ax.barh(i, max(act - core, 0.0), left=core, color=color, alpha=OTHER_ALPHA,
                edgecolor="white", height=0.7)  # fmt: skip
        if compact_text:
            other = max(act - core, 0.0)
            if m.vlm and m.has_server:
                label = (f"{_sec(act)} s (VLM processing {_sec(server)} + network/API "
                         f"{_sec(core - server)} + framework {_sec(other)})")
            elif m.vlm:
                label = f"{_sec(act)} s (API {_sec(core)} + framework {_sec(other)})"
            else:
                label = f"{_sec(act)} s (SRU inference {_sec(core)} + framework {_sec(other)})"
            ax.text(act * 1.12, i, label, va="center", fontsize=8.5 * fs)
        else:
            p95 = float(np.percentile(m.act_s, 95))
            ax.plot([act, p95], [i, i], color="#333333", lw=1.2)
            ax.plot([p95, p95], [i - 0.18, i + 0.18], color="#333333", lw=1.2)
            label = _breakdown(m)
            ax.text(p95 * 1.12, i, label, va="center", fontsize=8.5)
            ax.text(1.02, i, f"{np.median(m.sim_s):g} s sim | x{_rt_ratio(m):.2f}", va="center",
                    fontsize=9, transform=ax.get_yaxis_transform())  # fmt: skip
    ax.set_xscale("log")
    lo = min(float(m.core_s[m.core_s > 0].mean()) if np.any(m.core_s > 0) else 0.01 for m in ms)
    # Compact segment labels are longer, requiring extra space at the right edge.
    right_mult = 70 if compact_text else 40
    ax.set_xlim(lo / 3, max(float(np.percentile(m.act_s, 95)) for m in ms) * right_mult)
    ax.set_yticks(y)
    ax.set_yticklabels([_label(m.method_id) for m in ms])
    if compact_text:
        ax.set_xlabel("Seconds per decision (mean), log scale",
                      fontsize=ax.xaxis.label.get_fontsize() * fs)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_tick_params(which="both",
                                 labelsize=axis.get_ticklabels()[0].get_fontsize() * fs)
    else:
        ax.set_xlabel("Seconds per decision (mean; whisker = p95), log scale")
        ax.text(1.02, len(ms) - 0.3, "sim per step | real-time ratio", fontsize=9,
                fontweight="bold", transform=ax.get_yaxis_transform())  # fmt: skip
    from matplotlib.patches import Patch

    grey = "#777777"
    server_label = "VLM processing" if compact_text else "VLM provider processing (astra only)"
    core_label = "Network/API or SRU inference" if compact_text else (
        "VLM API round trip (rest) / SRU network inference"
    )
    other_label = "Framework" if compact_text else "Other framework overhead"
    handles = [
        Patch(color=mcolors.to_hex(np.array(mcolors.to_rgb(grey)) * (1 - SERVER_SHADE)),
              label=server_label),
        Patch(color=grey, label=core_label),
        Patch(color=grey, alpha=OTHER_ALPHA, label=other_label),
    ]  # fmt: skip
    # Compact (blog) mode keeps the legend and caption close under the axis; the legacy
    # spacing below is unchanged for the default report figure.
    # Must clear the x-axis label (sits just under the ticks) before the gap gets any tighter.
    legend_y = -0.09 - 0.12 / len(ms) if compact_text else -0.1 - 0.25 / len(ms)
    caption_gap = -34 if compact_text else -48
    # Legend text must not be smaller than the axis tick labels it sits below.
    legend_fontsize = ax.yaxis.get_ticklabels()[0].get_fontsize() if compact_text else 9 * fs
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, legend_y),
              ncol=len(handles) if compact_text else legend_ncol,
              frameon=False, fontsize=legend_fontsize)  # fmt: skip
    ax.set_title(title if title is not None else (
        "Compute time per decision (astra: Flex tier, includes queueing; segments are "
        "stacked on a log axis, read the numbers)"
    ), fontsize=11 * fs)
    if caption is not None:
        ax.annotate(
            caption, xy=(0.5, legend_y), xycoords="axes fraction",
            xytext=(0, caption_gap), textcoords="offset points", ha="center", va="top",
            fontsize=8 * fs, annotation_clip=False,
        )
    return figures._save(fig, out_stem)


def build(campaign_dir: Path, out_dir: Path) -> Path:
    ms = load_latency(campaign_dir)
    t = latency_table(ms)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{OUT_NAME}.md").write_text(tables.to_markdown(t), encoding="utf-8")
    (out_dir / f"{OUT_NAME}.csv").write_text(tables.to_csv(t), encoding="utf-8")
    return latency_bars(ms, out_dir / OUT_NAME)


if __name__ == "__main__":
    print(build(Path(sys.argv[1]), Path(sys.argv[2])))
