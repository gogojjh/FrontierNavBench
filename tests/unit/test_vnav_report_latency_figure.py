"""Decision-latency figure tests: per-step aggregation, component breakdown and wall/simulation
ratios.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vnav_benchmark.report import latency_figure as lf


def _write(root: Path, mid: str, ep: str, steps: list) -> None:
    d = root / "runs" / mid / ep / "r0"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"method_id": mid, "status": "scored"}))
    (d / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps))


def _step(act: float, hold: float, **lat: float) -> dict:
    return {"act_wall_s": act, "applied": {"hold_s": hold}, "debug": {"latency": lat}}


def test_aggregate_and_table(tmp_path: Path) -> None:
    _write(tmp_path, "dense-contloc-ptgoal", "ep_a", [_step(0.02, 0.2, controller_s=0.015)] * 10)
    _write(tmp_path, "nomap-noloc-imggoal-astra", "ep_a",
           [_step(4.0, 1.0, vlm_api_s=3.9, server_ms=3000.0)] * 4)  # fmt: skip
    _write(tmp_path, "nomap-noloc-imggoal-qwen", "ep_a", [_step(2.0, 1.0, vlm_api_s=1.98)] * 3)
    ms = lf.load_latency(tmp_path)
    assert [m.method_id for m in ms] == [
        "dense-contloc-ptgoal", "nomap-noloc-imggoal-astra", "nomap-noloc-imggoal-qwen",
    ]  # fmt: skip
    b0, astra, qwen = ms
    assert lf._rt_ratio(b0) == pytest.approx(0.1) and lf._rt_ratio(astra) == pytest.approx(4.0)
    assert astra.has_server and not qwen.has_server
    assert "3.00 provider + 0.90 API rest" in lf._breakdown(astra)
    assert "provider time n/a" in lf._breakdown(qwen)
    t = lf.latency_table(ms)
    assert t.rows[0][6] == "10.0" and t.rows[1][-1] == "4.00"
    with patch.object(lf.figures, "_save", wraps=lf.figures._save) as save:
        assert lf.build(tmp_path, tmp_path / "out").exists()
    ax = save.call_args.args[0].axes[0]
    assert any("provider +" in t.get_text() and "API rest" in t.get_text() for t in ax.texts)
    legend = ax.get_legend()
    assert getattr(legend, "_ncols", getattr(legend, "_ncol", None)) == 2


@pytest.mark.parametrize("legend_ncol", [1, 2, 4])
def test_compact_latency_text_and_single_row_legend(tmp_path: Path, legend_ncol: int) -> None:
    _write(tmp_path, "dense-contloc-ptgoal", "ep_a", [_step(0.024, 0.2, controller_s=0.02)])
    _write(tmp_path, "nomap-noloc-imggoal-astra", "ep_a",
           [_step(3.62, 1.0, vlm_api_s=3.60, server_ms=2630.0)])
    _write(tmp_path, "nomap-noloc-imggoal-qwen", "ep_a", [_step(2.0, 1.0, vlm_api_s=1.98)])
    with patch.object(lf.figures, "_save", wraps=lf.figures._save) as save:
        lf.latency_bars(lf.load_latency(tmp_path), tmp_path / "compact",
                        compact_text=True, legend_ncol=legend_ncol)
    ax = save.call_args.args[0].axes[0]
    labels = [t.get_text() for t in ax.texts]
    assert "3.62 s (VLM processing 2.63 + network/API 0.97 + framework 0.020)" in labels
    assert "0.024 s (SRU inference 0.020 + framework 0.004)" in labels
    assert "2.00 s (API 1.98 + framework 0.020)" in labels
    assert all("s (" in t and t.endswith(")") for t in labels)
    assert not any("provider" in t or "API rest" in t or " other" in t for t in labels)
    assert not any("sim" in t or "real-time" in t for t in labels)
    assert not ax.lines
    assert ax.get_xlabel() == "Seconds per decision (mean), log scale"
    assert "p95" not in ax.get_xlabel() and "whisker" not in ax.get_xlabel()
    legend = ax.get_legend()
    assert getattr(legend, "_ncols", getattr(legend, "_ncol", None)) == len(legend.texts) == 3


def test_compact_latency_fonts_are_larger(tmp_path: Path) -> None:
    _write(tmp_path, "dense-contloc-ptgoal", "ep_a", [_step(0.024, 0.2, controller_s=0.02)])
    axes = []
    for compact_text in (False, True):
        with patch.object(lf.figures, "_save", wraps=lf.figures._save) as save:
            lf.latency_bars(lf.load_latency(tmp_path), tmp_path / str(compact_text),
                            caption="Caption", compact_text=compact_text)
        axes.append(save.call_args.args[0].axes[0])
    legacy, compact = axes
    assert compact.title.get_fontsize() > legacy.title.get_fontsize()
    assert compact.xaxis.label.get_fontsize() > legacy.xaxis.label.get_fontsize()
    assert compact.get_xticklabels()[0].get_fontsize() > legacy.get_xticklabels()[0].get_fontsize()
    assert compact.get_yticklabels()[0].get_fontsize() > legacy.get_yticklabels()[0].get_fontsize()
    assert compact.texts[0].get_fontsize() > legacy.texts[0].get_fontsize()
    assert compact.texts[-1].get_fontsize() > legacy.texts[-1].get_fontsize()
    assert (compact.get_legend().texts[0].get_fontsize()
            > legacy.get_legend().texts[0].get_fontsize())


@pytest.mark.parametrize("legend_ncol", [1, 4])
def test_legacy_latency_respects_legend_ncol(tmp_path: Path, legend_ncol: int) -> None:
    _write(tmp_path, "dense-contloc-ptgoal", "ep_a", [_step(0.024, 0.2, controller_s=0.02)])
    with patch.object(lf.figures, "_save", wraps=lf.figures._save) as save:
        lf.latency_bars(lf.load_latency(tmp_path), tmp_path / "legacy", legend_ncol=legend_ncol)
    legend = save.call_args.args[0].axes[0].get_legend()
    assert getattr(legend, "_ncols", getattr(legend, "_ncol", None)) == legend_ncol
