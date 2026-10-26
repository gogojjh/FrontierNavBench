"""Run AM, AMem and AMM against fixed, published RGB map selections.

AM changes the sparse map, AMem changes memory, and AMM changes both. The base
method is sparse-initloc-imggoal-astra. Published selections are immutable inputs;
this runner reproduces navigation with those selections, not paid map reselection.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from frontier_nav_bench import mapview, oracle, pins
from frontier_nav_bench.budget import BudgetGuard
from frontier_nav_bench.episodes import load_episode_set
from frontier_nav_bench.interfaces import FatalInfraError, MapHandle, Pose2D
from frontier_nav_bench.methods import make_method
from frontier_nav_bench.methods.active_memory_policy import ActiveMemoryMethod
from frontier_nav_bench.paths import data_root, repository_root
from frontier_nav_bench.preflight import check_identity, local_git
from frontier_nav_bench.rundir import write_json_atomic
from frontier_nav_bench.runner import (
    LOCK_NAME,
    BenchConfig,
    Campaign,
    CampaignStopped,
    IsaacProvider,
    KinematicProvider,
    RunContext,
    coverage,
)
from frontier_nav_bench.sim.orchestrator import IsaacOrchestrator

METHOD = "sparse-initloc-imggoal-astra"
VARIANTS = ("AM", "AMem", "AMM")
EPISODE_SETS = ("env0_test_v3", "env1_test_v1", "env3_test_v1")


def parse_variants(text: str) -> Tuple[str, ...]:
    """Parse an ordered, nonempty list without duplicates or unknown variants."""
    variants = tuple(item.strip() for item in text.split(","))
    if not variants or len(set(variants)) != len(variants) or any(
        item not in VARIANTS for item in variants
    ):
        raise ValueError("variants must be a unique comma-separated subset of AM,AMem,AMM")
    return variants


def variant_inputs(variant: str, baseline: Any, active: Any) -> Tuple[Any, Callable]:
    """Select the two experimental factors while retaining the base method ID."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown experimental variant: {variant!r}")
    return (baseline if variant == "AMem" else active,
            make_method if variant == "AM" else ActiveMemoryMethod)


class TaggedGuard(BudgetGuard):
    """Keep one durable campaign ledger with the actual variant on every call."""

    variant = ""

    def record(self, row: Dict[str, Any]) -> None:
        super().record(dict(row, experimental_variant=self.variant))


@contextmanager
def reserve_slot(slot: str, backend: str) -> Iterator[None]:
    """Exclusively reserve a package simulator slot; kinematic needs no slot."""
    if backend == "kinematic":
        yield
        return
    if backend != "isaac" or slot not in ("s1", "s2"):
        raise ValueError("Expected backend kinematic/isaac and slot s1/s2")
    path = data_root() / "locks" / f"isaac-{slot}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignStopped(f"Isaac slot {slot} is reserved by another campaign") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class CheckedProvider(IsaacProvider):
    """Reject occupied slots and check frozen identity before method construction."""

    def __init__(self, orchestrator: IsaacOrchestrator, episodes: Path, robot: str) -> None:
        super().__init__(orchestrator, robot)
        self.episodes = episodes

    def open(self, scene: str, start: Pose2D, third: bool, log_dir: Path) -> Any:
        if self.orch._port_state()[0]:
            raise FatalInfraError(f"Isaac slot {self.orch.slot} port is occupied; leaving it alone")
        session = super().open(scene, start, third, log_dir)
        try:
            checks = check_identity(session.identity, scene, self.episodes / "episodes.json")
            write_json_atomic(log_dir / "server_checks.json", [asdict(check) for check in checks])
            if not all(check.ok for check in checks):
                failed = ", ".join(check.name for check in checks if not check.ok)
                raise FatalInfraError("Server identity failed: " + failed)
        except Exception:
            session.close()
            raise
        return session


class ActiveMaps:
    """Expose verified sparse maps only, recording both experimental factors."""

    def __init__(
        self, entries: Mapping[str, mapview.MapEntry], root: Path,
        camera_forward_m: float, experimental_variant: str,
    ) -> None:
        if experimental_variant not in VARIANTS:
            raise ValueError(f"Unknown experimental variant: {experimental_variant!r}")
        self.entries, self.root = dict(entries), root
        self.camera_forward_m, self.variant = camera_forward_m, experimental_variant

    def get(self, route: str, variant: str) -> Tuple[MapHandle, Sequence[float], Dict[str, Any]]:
        """Return a sanitized map with source hash and explicit ablation provenance."""
        if variant != "sparse" or route not in self.entries:
            raise ValueError(f"Unexpected ablation map request: {route!r}/{variant!r}")
        entry = self.entries[route]
        if (entry.source / mapview.VIEW_META_NAME).is_file():
            handle = mapview.SanitizedMap(
                entry.source, entry.map_id, "sparse", entry.edge_type, self.camera_forward_m,
                expected_sha256=entry.view_sha256, scene=entry.scene,
                map_to_world=entry.map_to_world.ravel(),
            )
        else:
            handle = mapview.materialize(entry, self.root, self.camera_forward_m)
        return handle, entry.map_to_world.ravel().tolist(), {
            "map_id": entry.map_id, "sha256": handle.sha256, "variant": "sparse",
            "role": "experimental_ablation", "route": route, "edge_type": entry.edge_type,
            "experimental_variant": self.variant, "active_map": self.variant != "AMem",
            "active_memory": self.variant != "AM", "formal_result": False,
        }


def active_entry(
    baseline: mapview.MapEntry, selection: Mapping[str, Any], root: Path
) -> mapview.MapEntry:
    """Verify a published selected view and preserve the route's rigid transform."""
    selected, original = selection["selected_ids"], selection["baseline_ids"]
    count = selection["candidate_count"]
    if (not selected or len(selected) != len(original) or len(set(selected)) != len(selected)
            or not isinstance(count, int) or count <= 0
            or any(not isinstance(index, int) or not 0 <= index < count for index in selected)
            or len(selected) != selection["keyframe_count"]):
        raise ValueError("Invalid published RGB selection")
    source = root / selection["map_path"]
    # Published route paths are fixed; selection metadata cannot escape its bundle.
    if source.resolve() != (root / "data" / "active_maps" / selection["route"]).resolve():
        raise ValueError("Published active-map path does not match its route")
    mapview.verify_view(source, selection["view_sha256"])
    return replace(
        baseline, source=source, map_id="active_" + selection["route"],
        view_sha256=selection["view_sha256"], num_keyframes=len(selected), trav_src_sha256=None,
    )


def _experiment(root: Path, header: Dict[str, Any]) -> None:
    path = root / "experiment.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != header:
        raise ValueError("Ablation inputs or code changed; use a new campaign name")
    write_json_atomic(path, header)


def run_ablations(
    episodes: str, campaign: str, variants: Sequence[str] = VARIANTS,
    backend: str = "kinematic", slot: str = "s1", max_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare and resume separate variant campaigns with a shared spending cap.

    max_jobs limits new navigation runs across all variants. Set it to zero to
    verify inputs and write plans without opening a simulator or calling a model.
    Kinematic runs still use the selected VLM when max_jobs permits execution;
    they are rehearsals and cannot reproduce Isaac physics or camera images.
    """
    variants = tuple(variants)
    parse_variants(",".join(variants))
    if episodes not in EPISODE_SETS or backend not in ("kinematic", "isaac"):
        raise ValueError("Unknown frozen episode set or simulation backend")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", campaign):
        raise ValueError("campaign must be a single directory name")
    if max_jobs is not None and max_jobs < 0:
        raise ValueError("max_jobs must be nonnegative")
    root = data_root() / "campaigns" / campaign
    root.mkdir(parents=True, exist_ok=True)
    with (root / LOCK_NAME).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignStopped("Another process is using this ablation campaign") from exc
        return _run_locked(root, episodes, variants, backend, slot, max_jobs)


def _run_locked(
    root: Path, episodes: str, variants: Tuple[str, ...], backend: str,
    slot: str, max_jobs: Optional[int],
) -> Dict[str, Any]:
    pins.require_pins()
    repo = repository_root()
    episode_dir = repo / "configs" / "episodes" / episodes
    es, maps = load_episode_set(episode_dir), mapview.load_maps_config()
    selection_path = repo / "data" / "active_maps" / "selections.json"
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    if selections.get("schema") != "vnav.active-map-selections/1":
        raise ValueError("Unknown published active-map selection schema")
    by_label = {item["route"]: item for item in selections["routes"]}
    if len(by_label) != len(selections["routes"]):
        raise ValueError("Duplicate published active-map route")
    baseline, active, selected = {}, {}, {}
    for rec in es.records.values():
        entry = maps.for_route(rec.route, "sparse")
        if entry.scene != es.payload["scene"]:
            raise ValueError("Map scene differs from frozen episode scene")
        baseline[rec.route] = entry
        active[rec.route] = active_entry(entry, by_label[rec.label], repo)
        selected[rec.label] = by_label[rec.label]
    sources = {}
    fingerprints = {}
    for variant in variants:
        entries, _ = variant_inputs(variant, baseline, active)
        source = ActiveMaps(entries, root / variant / "maps", maps.camera_forward_m, variant)
        sources[variant] = source
        fingerprints[variant] = {route: source.get(route, "sparse")[2]
                                 for route in sorted(entries)}
    code = [Path(__file__), repo / "scripts" / "ablation_run.py"]
    code += sorted((repo / "frontier_nav_bench" / "methods").glob("active_*.py"))
    code += [repo / "configs" / "methods" / (METHOD + ".yaml")]
    bench = replace(BenchConfig.load(), max_attempts=1)
    header = {
        "schema": "vnav.ablations/1", "method_id": METHOD,
        "variants": list(variants), "episode_set": es.name,
        "episode_set_sha256": es.manifest_sha256, "backend": backend,
        "slot": slot if backend == "isaac" else None, "formal_result": False,
        "map_selection": "fixed published RGB selections; no reselection performed",
        "selections": selected, "maps": fingerprints,
        "bench": bench.settings(scene=es.payload["scene"]),
        "code": {path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in code},
    }
    guard = TaggedGuard(root / "ledger.jsonl")
    header["budget"] = {"caps_usd": guard.caps, "episode_cap_usd": guard.episode_cap_usd}
    _experiment(root, header)
    orch = IsaacOrchestrator(slot) if backend == "isaac" and max_jobs != 0 else None
    provider = (CheckedProvider(orch, episode_dir, bench.robot) if orch is not None else
                KinematicProvider(oracle.build_truth_grid(es.payload["scene"])))
    remaining = max_jobs
    results = {}
    with reserve_slot(slot, backend):
        try:
            for variant in variants:
                _, factory = variant_inputs(variant, baseline, active)
                guard.variant = variant
                ctx = RunContext(
                    root=root / variant, episode_set=es, sim=provider, maps=sources[variant],
                    guard=guard, bench=bench, method_factory=factory, git=local_git(),
                )
                tokens = tuple(es.tokens())
                before = coverage(ctx.root, (METHOD,), tokens, 1)["done"]
                result = Campaign(ctx, (METHOD,), tokens, 1).run(max_jobs=remaining)
                results[variant] = result
                if remaining is not None:
                    remaining -= result["done"] - before
            write_json_atomic(root / "coverage.json", {
                "schema": "vnav.ablation-coverage/1", "variants": results,
                "total": sum(value["total"] for value in results.values()),
                "done": sum(value["done"] for value in results.values()),
            })
        finally:
            if orch is not None:
                orch.stop()
    return results
