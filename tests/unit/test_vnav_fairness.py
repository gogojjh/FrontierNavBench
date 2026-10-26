import math

import pytest

from vnav_benchmark import fairness as fz
from vnav_benchmark.interfaces import Action, LocResult, Pose2D
from vnav_benchmark.methods import FACTORIES, registered_ids


def test_v1_registry_invariants() -> None:
    """Enforce seven-method/twelve-configuration fairness rules from plan section 2.

    contloc uses pose-based subgoal/arrival checks; initloc/noloc with imggoal
    uses VLM checks. nomap receives only the final goal image.
    """
    assert len(fz.V1_METHOD_IDS) == 12 and len(set(fz.V1_METHOD_IDS)) == 12
    assert len({fz.method_label(m).split("-")[0] for m in fz.V1_METHOD_IDS}) == 7
    for mid in fz.V1_METHOD_IDS:
        p = fz.parse_method_id(mid)
        if p.loc_mode == "contloc":
            assert (p.progress_judge, p.done_judge) == ("pose", "pose")
        else:
            assert (p.progress_judge, p.done_judge) == ("vlm", "vlm")
        assert (p.loc_mode == "contloc") == (p.self_loc_quota == fz.UNLIMITED)
        assert (p.map_variant is None) == (p.loc_mode == "noloc") == (p.goal_loc_quota == 0)
        assert (p.planner == "imggoal") == (p.vlm in fz.VLMS)
        assert not p.uses_truth
        assert mid in FACTORIES
    assert set(registered_ids()) == set(fz.V1_METHOD_IDS) | set(fz.DEBUG_METHOD_IDS)


@pytest.mark.parametrize(
    "mid,variant,loc,quota",
    [
        ("dense-contloc-ptgoal", "dense", "contloc", fz.UNLIMITED),
        ("sparse-initloc-imggoal-qwen", "sparse", "initloc", 1),
        ("nomap-noloc-imggoal-astra", None, "noloc", 0),
    ],
)
def test_parse_resources(mid: str, variant: object, loc: str, quota: int) -> None:
    p = fz.parse_method_id(mid)
    assert (p.map_variant, p.loc_mode, p.self_loc_quota) == (variant, loc, quota)


@pytest.mark.parametrize(
    "bad",
    [
        "dense-contloc",
        "dense-contloc-ptgoal-astra",  # ptgoal has no model suffix
        "dense-initloc-ptgoal",  # ptgoal requires continuous localization
        "dense-initloc-imggoal",  # imggoal requires a model suffix
        "dense-initloc-imggoal-gpt4",
        "nomap-initloc-imggoal-astra",  # Localization requires a map
        "dense-noloc-imggoal-astra",
        "medium-contloc-ptgoal",
    ],
)
def test_parse_rejects(bad: str) -> None:
    with pytest.raises(fz.MethodIdError):
        fz.parse_method_id(bad)


def test_contloc_imggoal_registered() -> None:
    p = fz.parse_method_id("sparse-contloc-imggoal-qwen")
    assert (p.progress_judge, p.done_judge, p.self_loc_quota) == ("pose", "pose", fz.UNLIMITED)
    assert FACTORIES["sparse-contloc-imggoal-qwen"].endswith("imggoal_vlm:make")
    assert fz.method_label("sparse-contloc-imggoal-qwen") == "M3-qwen"
    assert fz.method_label("nomap-noloc-imggoal-astra") == "M6-astra"
    assert fz.method_label("dense-contloc-ptgoal") == "B0"


def test_debug_ids() -> None:
    for mid in fz.DEBUG_METHOD_IDS:
        p = fz.parse_method_id(mid)
        assert p.uses_truth and p.is_debug and p.map_variant is None


def test_clamp_and_quantize() -> None:
    a, r = fz.clamp_action(Action(vx=1.5, vy=-0.9, wz=0.2, hold_s=0.26))
    assert (a.vx, a.vy, a.wz, a.hold_s) == (0.8, -0.3, 0.2, 0.3)
    assert r.clamped_vx and r.clamped_vy and not r.clamped_wz and r.hold_changed and r.any
    a, r = fz.clamp_action(Action(vx=-0.3, vy=0.0, wz=-2.0, hold_s=5.0))
    assert (a.vx, a.wz, a.hold_s) == (0.0, -0.7, 2.0) and r.clamped_vx  # Reverse motion is disabled
    a, r = fz.clamp_action(Action(vx=float("nan"), vy=0.1, wz=float("inf"), hold_s=float("nan")))
    assert (a.vx, a.wz, a.hold_s) == (0.0, 0.0, 0.1) and r.nonfinite
    a, r = fz.clamp_action(Action(vx=0.5, vy=0.0, wz=0.0, hold_s=1.0))
    assert not r.any and a.hold_s == 1.0
    a, _ = fz.clamp_action(Action(vx=0.5, vy=0.2, wz=0.3, hold_s=0.04, done=True))
    assert (a.vx, a.vy, a.wz, a.hold_s, a.done) == (0.0, 0.0, 0.0, 0.1, True)


def test_quantize_hold_grid() -> None:
    for k in range(1, 21):
        assert math.isclose(fz.quantize_hold(k * 0.1), k * 0.1)
        assert math.isclose(fz.quantize_hold(k * 0.1 + 0.049), k * 0.1)


class _Inner:
    def __init__(self) -> None:
        self.calls = 0

    def localize(self, rgb_jpeg, depth_m, K, *, purpose):  # type: ignore[no-untyped-def]
        self.calls += 1
        return LocResult(True, Pose2D(0.0, 0.0, 0.0), None)


def _loc(mid: str) -> "tuple[fz.QuotaLocalizer, _Inner]":
    inner = _Inner()
    return fz.QuotaLocalizer(inner, fz.parse_method_id(mid)), inner


def test_quota_contloc_once_per_frame() -> None:
    q, inner = _loc("dense-contloc-ptgoal")
    assert q.localize(b"", None, (), purpose="goal").ok
    assert q.localize(b"", None, (), purpose="goal").reason == "quota"
    for step in range(5):
        q.on_frame(step)
        assert q.localize(b"", None, (), purpose="self").ok
        assert q.localize(b"", None, (), purpose="self").reason == "quota"
    assert inner.calls == 6 and q.stats()["denied"] == {"goal": 1, "self": 5}


def test_quota_initloc_once_total() -> None:
    q, inner = _loc("sparse-initloc-imggoal-astra")
    q.on_frame(0)
    assert q.localize(b"", None, (), purpose="self").ok
    q.on_frame(1)
    assert not q.localize(b"", None, (), purpose="self").ok
    assert inner.calls == 1


def test_quota_noloc_gets_nothing() -> None:
    q, inner = _loc("nomap-noloc-imggoal-qwen")
    q.on_frame(0)
    assert not q.localize(b"", None, (), purpose="self").ok
    assert not q.localize(b"", None, (), purpose="goal").ok
    assert inner.calls == 0
    with pytest.raises(ValueError):
        q.localize(b"", None, (), purpose="other")
