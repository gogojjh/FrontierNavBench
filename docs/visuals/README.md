# README visual sources

The README combines figures generated from the distributed dataset with two
preserved historical artifacts. None of these images is a new simulator run.

## Figures generated from this checkout

| File | Inputs and interpretation |
| --- | --- |
| `goals.png` | All 15 RGB goal images named by the three frozen `configs/episodes/*/episodes.json` files, sorted by route number. Images are resized without cropping. Labels come from the route and goal-room fields. |
| `maps.png` | R5 dense and sparse `poses.txt` and `edges_trav.txt`, selected through `configs/maps.yaml`; four dense-map RGB keyframes; ENV0 `scene.json` and `truth.npz` for the gray evaluator floor plan. |

For the map figure, `frontier_nav_bench.topo` converts camera poses to body poses using
the registry's camera offset. The registry's `map_to_world` transform places them
on the evaluator floor plan. The statistics count graph nodes, unique undirected
traversal links and connected components. The gray background is evaluator-only
geometry; it is not an input supplied to every method. Traversal links are graph
connections and can span a curved route; a straight drawn link is not a claim of
straight-line collision-free motion.

From an evaluation environment installed with `.[evaluation]`, run:

```bash
python scripts/readme_figures.py
```

This overwrites `goals.png`, `maps.png` and `sources.json` only. To preview changes
in a separate directory:

```bash
python scripts/readme_figures.py --output-dir /tmp/vnav-readme-figures
```

The script uses `frontier_nav_bench.report.style` for palette and Palatino-compatible
typography (URW P052, with DejaVu Serif fallback). It records SHA-256 checksums of
its source code, rendering style and all data inputs in `sources.json`. It makes
no model API calls and starts no simulator. Rendering bytes can differ between
font or Matplotlib versions even when the input checksums match.

## Preserved recorded results

| File | Historical artifact |
| --- | --- |
| `env1_recorded_trajectories.png` | Existing documentation plot of B0 and M1–M6 trajectories on ENV1 routes R6–R10, with goals shown as stars. |
| `env0_m2_recorded.jpg` | Existing recorded M2/Astra frame on ENV0 R5, showing RGB, the goal image, subgoal, evaluator diagnostics and decision state. It is an intermediate step. |

These files were copied without resizing, cropping, recoloring or annotation from
the local documentation staging archive. Its associated `results.json` identifies
the historical snapshot as **v18**, generated **2026-10-24**, with source commit
`88a6f8e` and upstream commit `4eb61c9`. [archive.json](archive.json) records the
original artifact names, file checksums and checksum of that historical summary.
The staging archive is not a runtime dependency or a distributed result dataset.

The figures illustrate recorded behavior and the reporting interface. They do
not certify current-code performance. Raw historical run directories are not
bundled, so these specific trajectories and the exact recorded frame cannot be
regenerated from this checkout alone. `readme_figures.py` leaves them unchanged.

To produce new result figures, run a configured Isaac campaign, then:

```bash
python scripts/report_build.py data/campaigns/YOUR_CAMPAIGN
python -m frontier_nav_bench.report.episode_video RUN_DIR --stills
```

Replace `YOUR_CAMPAIGN` with the campaign name and `RUN_DIR` with a run directory
under its `runs/` tree. New outputs report that campaign's actual data; they do not
reconstruct the archived snapshot. See the root README for setup and credentials.

Dataset and scene provenance remain in `data/episode_provenance.json`,
`data/scenes/*/PROVENANCE.json` and `assets/provenance/`. See the root `NOTICE` and
`assets/LICENSES/` for retained asset licenses.
