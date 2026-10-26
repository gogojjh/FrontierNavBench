"""Synthetic report tables: criteria, mixed-fingerprint/dev-endpoint rejection, formats and paired
deltas.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from vnav_report_fixtures import make_run, synthetic, write_campaign

from vnav_benchmark.report import aggregate as agg
from vnav_benchmark.report import tables as tb
from vnav_benchmark.report.results_io import load_runs, row_from_run


def _rows(seed: int = 0):  # type: ignore[no-untyped-def]
    return [row_from_run(r, s) for r, s in synthetic(seed)]


def test_load_runs_from_campaign_dir(tmp_path: Path) -> None:
    runs = synthetic()
    write_campaign(tmp_path, runs)
    rows = load_runs(tmp_path)
    assert len(rows) == len(runs) == 2 * 3 * 8 + 10 * 8
    m2 = next(r for r in rows if r.code == "M4" and r.vlm == "astra")
    assert m2.endpoint == "astra_flex" and m2.service_tier == "flex" and m2.calls > 0
    b0 = next(r for r in rows if r.code == "B0")
    assert b0.vlm is None and b0.endpoint == "" and b0.calls == 0


def test_main_table_numbers_match_hand_count() -> None:
    t = tb.build_all(_rows())[0]
    rows = agg.scored(_rows())
    assert [r[0] for r in t.rows] == ["B0", "M1"] + [
        f"M{i}-{m}" for i in range(2, 7) for m in ("astra", "qwen")
    ]
    for line in t.rows:
        sub = [r for r in rows if r.method_id == line[1]]
        assert line[6] == str(len(sub)) == "8"
        assert line[7] == f"{100 * np.mean([r.success for r in sub]):.1f}"
        assert line[9] == f"{np.mean([r.spl for r in sub]):.3f}"
        ok = [r.plr for r in sub if r.success]
        assert line[10] == (f"{np.mean(ok):.2f}" if ok else tb.DASH)


def test_main_table_notes_thinking_settings() -> None:
    notes = " ".join(tb.main_table(agg.scored(_rows())).notes)
    assert "astra cannot disable thinking" in notes and "Qwen thinking disabled" in notes


def test_main_uses_first_repeat_only_and_drops_unscored() -> None:
    rows = _rows()
    rng = np.random.default_rng(1)
    extra = [
        row_from_run(*make_run("dense-contloc-ptgoal", 0, 0, True, rng, status="unscored_infra")),
        row_from_run(*make_run("dense-contloc-ptgoal", 1, 0, True, rng, termination="cost_guard")),
    ]
    main = agg.scored(rows + extra)
    assert all(r.repeat == 0 for r in main)
    assert sum(r.method_id == "dense-contloc-ptgoal" for r in main) == 8


def test_rejects_mixed_fingerprints_and_dev_endpoint() -> None:
    rows = _rows()
    rng = np.random.default_rng(2)
    mixed = rows + [
        row_from_run(
            *make_run("nomap-noloc-imggoal-astra", 0, 1, True, rng, fingerprint="sha256:other")
        )
    ]
    with pytest.raises(agg.ReportInputError, match="指纹"):
        tb.build_all(mixed)
    luna = rows + [
        row_from_run(*make_run("nomap-noloc-imggoal-astra", 0, 1, True, rng, endpoint="luna_dev"))
    ]
    with pytest.raises(agg.ReportInputError, match="luna_dev"):
        tb.build_all(luna)


def test_three_formats_have_identical_cells(tmp_path: Path) -> None:
    tables = tb.build_all(_rows())
    paths = tb.write_all(tables, tmp_path)
    assert set(paths) == {
        "main",
        "by_difficulty",
        "latency",
        "vlm_diagnostics",
        "nav_diagnostics",
        "contrasts",
        "repeats",
    }
    for t in tables:
        parsed = list(csv.reader(io.StringIO((tmp_path / f"{t.key}.csv").read_text())))
        assert parsed[0] == t.columns and parsed[1:] == t.rows
        md_rows = (tmp_path / f"{t.key}.md").read_text().splitlines()
        body = [ln for ln in md_rows if ln.startswith("| ")][1:]
        assert [[c.strip() for c in ln.strip("| ").split(" | ")] for ln in body] == t.rows
        tex = (tmp_path / f"{t.key}.tex").read_text()
        assert tex.count(r" \\") == len(t.rows) + 1 and r"\toprule" in tex
        for row in t.rows:
            assert " & ".join(tb._tex(c) for c in row) + r" \\" in tex
    assert "%" not in (tmp_path / "main.tex").read_text().replace(r"\%", "")


def test_paired_diff_uses_common_scored_pairs_only() -> None:
    rows = agg.scored(_rows())
    a = [r for r in rows if r.method_id == "dense-contloc-ptgoal"]
    b = [r for r in rows if r.method_id == "sparse-contloc-ptgoal"][:5]
    d = agg.paired_diff(a, b, "B0", "M1")
    ia = {r.episode_id: r for r in a}
    want = np.mean([ia[r.episode_id].success - r.success for r in b])
    assert d.n_pairs == 5 and d.d_sr == pytest.approx(want)
    assert d.d_sr_ci[0] <= d.d_sr <= d.d_sr_ci[1]


def test_contrast_pairs_cover_plan_and_models() -> None:
    names = [(a, b) for a, b, _ in agg.contrast_pairs(agg.scored(_rows()))]
    assert ("dense-contloc-ptgoal", "sparse-contloc-ptgoal") in names
    assert ("dense-initloc-imggoal-qwen", "nomap-noloc-imggoal-qwen") in names  # M4 − M6
    assert ("dense-contloc-ptgoal", "dense-contloc-imggoal-astra") in names  # B0 − M2
    assert ("sparse-contloc-ptgoal", "sparse-contloc-imggoal-qwen") in names  # M1 − M3
    assert ("dense-contloc-imggoal-astra", "dense-initloc-imggoal-astra") in names  # M2 − M4
    assert ("sparse-initloc-imggoal-astra", "sparse-initloc-imggoal-qwen") in names
    assert len(names) == len(set(names)) == 1 + 2 * 8 + 5


def test_repeats_table_only_multi_run_methods() -> None:
    t = tb.repeats_table(_rows())
    assert [r[0] for r in t.rows] == ["B0", "M1"] and all(r[1] == "3" for r in t.rows)


def test_band_and_diag_tables_shapes() -> None:
    main = agg.scored(_rows())
    band = tb.band_table(main)
    assert len(band.columns) == 1 + 3 * 3
    assert all(r[1] == "3" for r in band.rows)
    diag = tb.vlm_diag_table(main)
    assert len(diag.rows) == 10
    m6 = next(r for r in diag.rows if r[0] == "M6-astra")
    m2 = next(r for r in diag.rows if r[0] == "M2-astra")
    assert "Degraded" not in diag.columns and "Initial loc. OK" not in diag.columns
    assert m2[8] == "16" and m6[8] == tb.DASH  # Rejected VLM done reports: two per episode across
    # eight episodes
    assert next(r for r in diag.rows if r[0] == "M4-astra")[5] == "60.0"  # 1800 / 3000


def test_summary_read_from_run_json_method_stats() -> None:
    run, summary = synthetic()[-1]
    run = dict(run, method_stats={"summary": summary})
    row = row_from_run(run)
    assert (
        row.endpoint == "qwen"
        and row.fingerprint
        and row.subgoal_advances == summary["subgoal_advances"]
    )


def test_main_uses_common_pairs_and_supplement_lists_the_rest() -> None:
    # Main tables use shared routes; supplementary tables show extras when configurations differ.
    assert "supplementary" not in {t.key for t in tb.build_all(_rows())}
    rows = [r for r in _rows() if r.vlm != "astra" or int(r.episode_id[2:]) < 4]
    tables = {t.key: t for t in tb.build_all(rows)}
    main, sup = tables["main"], tables["supplementary"]
    assert all(r[6] == "4" for r in main.rows) and len(main.rows) == 12
    assert any("4 episode pairs × 1 run" in n and "25 percentage points" in n for n in main.notes)
    assert any("Budget: 90 s" in n for n in main.notes)
    assert [r[0] for r in sup.rows] == ["B0", "M1"] + [f"M{i}-qwen" for i in range(2, 7)]
    contrasts = {r[0]: r for r in tables["contrasts"].rows}
    assert contrasts["B0 − M1"][2] == "8" and contrasts["M2-astra − M4-astra"][2] == "4"
    assert len(tables["contrasts"].columns) == 9


def test_loose_label_follows_metrics_constants(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from vnav_benchmark import metrics

    monkeypatch.setattr(metrics, "SUCCESS_DIST_LOOSE_M", 2.0)
    monkeypatch.setattr(metrics, "SUCCESS_YAW_LOOSE_DEG", 90.0, raising=False)
    assert tb.loose_label() == "SR@2m/90°"
    assert "SR@2m/90° (%)" in tb.main_table(agg.scored(_rows())).columns


def test_main_table_notes_arrival_threshold_and_extra() -> None:
    t = tb.build_all(_rows(), ["extra note"])[0]
    assert any("within 0.5 m and 30°" in n for n in t.notes)
    assert t.notes[-1] == "extra note"

