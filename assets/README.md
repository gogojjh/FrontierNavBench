# Assets and data

This guide describes the files used by FrontierNavBench and how to prepare the
licensed robot model. For the installation and evaluation workflow, start with
the [main README](../README.md).

## What is included

Paths below are relative to the repository root.

| Path | Contents | Used by |
| --- | --- | --- |
| `assets/scenes/` | Three frozen scene USDs and 129 referenced textures | Isaac Sim |
| `assets/weights/sru/` | Depth encoder and navigation-policy ONNX models | SRU CPU controller |
| `assets/weights/go2/` | MoE-CTS walking policy, optional ONNX export and parameters | Go2 simulation |
| `data/maps/` | 30 sanitized dense/sparse route maps | B0–M5 navigation |
| `data/active_maps/` | 15 selected map views and their selection records | AM and AMM |
| `data/scene_metadata/` | Recorded route transforms and camera-pose metadata | Evaluator identity checks |
| `data/scenes/` | Frozen truth grids, floor connections and layout identities | Evaluator |
| `configs/episodes/` | 15 tasks, goal photographs and goal depth images | Evaluation |

All bundled inputs are ordinary local files. Keep `scenes/textures/` alongside
the scene USDs so their relative references resolve. Raw recording images and
historical run/API logs are excluded from the input bundle.

The map and goal-image bytes preserve their original fingerprints. Episode
source metadata points to the bundled final evaluation maps; original map
bindings are recorded in
[`data/episode_provenance.json`](../data/episode_provenance.json).

## Check the installation

From the repository root, in the evaluation environment:

```bash
python scripts/check_install.py --smoke
# After preparing the local robot:
python scripts/check_install.py --robot
```

[`manifest.json`](manifest.json) records file sizes, SHA256 values, origins and
redistribution status. The check also verifies map fingerprints, frozen task
definitions and runtime sources. `--smoke` runs one temporary CPU debugging cell
without model API calls.

## Prepare the Go2 robot

The official NVIDIA Go2 model and its locally generated Odin payload layer are
installed under `assets/robots/go2/`, which is excluded from public Git. Use a
licensed Isaac Sim installation and run these commands from the repository root:

```bash
# Preview without downloading or writing anything.
conda run -n vnav-sim python setup/robot.py --dry-run

# Download official files, verify their SHA256 values and build the payload.
conda run -n vnav-sim python setup/robot.py

# Verify the resulting local robot; no network or writes.
conda run -n vnav-sim python setup/robot.py --check
```

The helper reuses identical files and refuses changed files or destination
symlinks. It imports OpenUSD directly and does not start a simulator. OpenUSD is
available in the simulation environment; checking the exact frozen file bytes
also works with standard Python.

| Override | Use |
| --- | --- |
| `python setup/robot.py --directory PATH` | Install the complete robot asset tree in another directory |
| `VNAV_GO2_USD` | Select the generated `go2_odin.usd` when overriding the default robot path |
| `VNAV_ASSET_DIR` | Runtime root for scene assets and model weights; default is this repository's `assets/` |

Use absolute paths for `.env` overrides. The NVIDIA robot files retain their
upstream terms; the benchmark's MIT license applies to the project's own code
and data. Isaac Sim supplies `OmniPBR.mdl` through its material system.

<details>
<summary>Robot payload and byte-level reproducibility</summary>

The generated robot includes two 1.5 kg payloads, invisible collision boxes,
fixed joints and the original IMU/radar mass corrections. These properties
reproduce the benchmark robot.

The benchmark's USD libraries reproduce this exact Odin SHA256:

```text
03fed2e83fad447448e88b308ae81e6dcf0a68fd90d5b657ed44b2906c85076b
```

Other OpenUSD versions can serialize the same physical properties differently.
The helper checks the authored payload layer and reports both `semantic_match`
and `exact_frozen_bytes`. Building or checking a different serialization requires
OpenUSD (`pxr`).

</details>

## Licenses and provenance

| Asset | License / source |
| --- | --- |
| Benchmark geometry, maps, tasks and procedural textures | Project MIT notices |
| Poly Haven models, textures and lighting | [CC0-1.0](LICENSES/CC0-1.0.txt) |
| SRU-Odin models and inference code | [MIT](LICENSES/MIT-SRU-Odin.txt) |
| MoE-CTS Go2 policy | [Apache-2.0](LICENSES/Apache-2.0-go2-policy.txt) |
| Official NVIDIA robot and Isaac Sim | Installed locally under NVIDIA's upstream terms |

Poly Haven source URLs and original checksums are listed in
[`provenance/scene_sources.json`](provenance/scene_sources.json). Walking-policy
provenance is retained in
[`weights/go2/README.upstream.md`](weights/go2/README.upstream.md). Each evaluator
scene export has its own `PROVENANCE.json` and license under `data/scenes/`.
