# Runtime structure

The package runs from a source checkout installed in editable mode. Configuration,
bundled inputs and simulator assets live alongside the Python package; a Python
wheel by itself is not a complete benchmark distribution.

```text
scripts/campaign_run.py or scripts/ablation_run.py
    -> runner / ablations: campaign lifecycle, identity and budget checks
        -> methods: navigation decisions with permitted observations
            -> methods/sru: separate learned controller process
            -> vlm: provider clients, prompts, memory and usage accounting
        -> sim: isolated control transport and kinematic/Isaac backends
        -> oracle + metrics: evaluator-only truth and scoring
    -> rundir: resumable run artifacts
scripts/report_build.py
    -> report: paired aggregation, tables and figures from recorded runs
```

`interfaces.py` defines method inputs and outputs. `fairness.py` defines method
IDs and access permissions. `mapview.py` verifies sanitized map contents;
`topo.py` owns graph loading and path search. `episodes.py` loads frozen tasks.

The evaluator's world poses and grids must not enter VLM prompts or methods
without the method's declared truth/localization access. Simulation and SRU use
separate Python environments; transport boundaries avoid importing Isaac into
the CPU evaluator.

`report/episode_video.py`, `report/compare_video.py` and `report/latency_figure.py`
are executable modules, even though they are not imported by campaign execution.
`report/failures.py` provides failure classification for recorded results. These
are retained reporting APIs, not abandoned navigation implementations.

The published ablation runner consumes fixed active-map selections. The unused
paid keyframe-selection implementation was removed; constructing new selections
is a separate experiment. Test clients and synthetic test fixtures live under
`tests/unit/` and are not runtime dependencies.

See [CONTRIBUTING](../CONTRIBUTING.md) for offline checks and manifest maintenance.
