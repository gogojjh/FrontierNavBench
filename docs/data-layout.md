# Published data layout

`data/maps/` contains 30 baseline map views, one dense/sparse pair per route.
Names use a two-digit route number for natural lexical ordering:

```text
data/maps/
  R01_dense/     # R1: keyframes selected at 3 m / 45 degrees
  R01_sparse/    # R1: keyframes selected at 9 m / 120 degrees
  ...
  R15_dense/
  R15_sparse/
```

Human route labels remain `R1`–`R15`. Routes R1–R5 belong to the single-floor
office, R6–R10 to the garden, and R11–R15 to the two-floor office. The authoritative
route-to-scene and map-to-world bindings live in `configs/maps.yaml`; do not
infer scene identities or transforms from file names.

Each view includes poses, camera intrinsics, graph edges, timestamps, sanitized
metadata and paired RGB/depth keyframes under `seq/`. The map fingerprint covers
the contents of the view. Renaming its enclosing directory does not change it.
Methods must continue to receive only the sanitized view, not evaluator truth.

## Naming migration

[`maps_renamed.json`](../data/maps_renamed.json) records all previous names, new
names, route labels, variants and unchanged view fingerprints. For example,
`route_20261022-220738-f1fb8e_kf3m45` became `R06_dense`.

Current map IDs, source paths, episode bindings and asset inventory use the new
names. Historical construction IDs in `data/episode_provenance.json`, asset
`source_relative` fields and old frozen manifests remain provenance. The naming
migration changed no goal image, map file, numeric task record or scoring rule.
Episode directory manifests were refreshed for their changed source metadata;
the frozen layer has a new version linked to the previous one.

## Other inputs and outputs

| Path | Purpose |
| --- | --- |
| `data/active_maps/R1`–`R15` | Fixed selections used by AM/AMM, with `selections.json` |
| `data/scene_metadata/` | Original recording session identities and transforms |
| `data/scenes/` | Evaluator scene layouts, truth grids and provenance |
| `configs/episodes/` | Frozen tasks and goal RGB/depth images |
| `assets/manifest.json` | Published asset hashes, origins and licensing |
| `data/campaigns/` | Generated results, ignored by Git |

`VNAV_DATA_DIR` overrides the data root; a relative value is resolved from this
checkout. To relocate inputs, copy the complete published `data/` tree and retain
its internal layout. `VNAV_ASSET_DIR` similarly selects simulator/model assets.
