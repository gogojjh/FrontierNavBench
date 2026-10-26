# Contributing

Start with the [README](README.md) for evaluation and simulation setup. Small,
focused pull requests make changes to benchmark behavior easier to review.

## Local development

Use Linux and Python 3.8 or 3.10. The evaluation setup keeps these interpreters
separate from Isaac Sim's Python 3.11 environment.

```bash
VNAV_EVAL_PYTHON=python3.10 bash setup/evaluation.sh
source .venv-eval/bin/activate
python -m pip install -e '.[evaluation,dev]'
bash scripts/verify.sh
```

The checks cover lint, offline regression tests, bundled input checksums and one
temporary CPU debugging cell. Socket tests use isolated loopback ports; allow
local socket creation if your sandbox restricts it. Tests never need API keys,
the simulator or a robot. Fake VLM responses belong in `tests/`, not runtime code.
GPU simulation and real robot checks are separate and must be reported separately.

## Regression suite

`tests/unit/` separates navigation methods, simulator transport, evaluator fairness
and metrics, model clients, input integrity and reports. The fixture helpers are
shared synthetic-data builders; `golden/` contains fixed model-request and action
traces used by replay tests. Keep these inputs with their consumers when cleaning
up tests. Bytecode caches under `__pycache__/` are generated and ignored by Git.

For a focused metrics check, run `python -m pytest tests/unit/test_vnav_metrics.py`.
Run `bash scripts/verify.sh` before publishing a change.

## Code and data boundaries

- Runtime code lives in `frontier_nav_bench/`; CLI and maintenance entrypoints in
  `scripts/`; environment setup in `setup/`; tests in `tests/unit/`.
- Shared code must remain compatible with Python 3.8. Follow the existing Ruff
  configuration, use type hints and explain public interfaces in docstrings.
- Keep world poses and truth grids in the evaluator. Methods receive only the
  observations and map/localization access permitted by their method ID.
- Preserve map and goal-image bytes when reorganizing paths. See
  [data layout](docs/data-layout.md) for names and migration provenance.
- Keep the existing MIT license and third-party notices. Do not add robot
  binaries, model-provider credentials, raw experiment logs or private datasets.

## Integrity manifests

Source changes intentionally invalidate the runtime and frozen-layer checks.
After reviewing a change, update only the changed entries of
`configs/runtime_manifest.json` using their SHA256 values. A removed runtime file
must be removed from that manifest. Then create a new frozen layer:

```bash
python -c 'from pathlib import Path; from frontier_nav_bench.frozen import freeze_layer; print(freeze_layer(Path.cwd(), Path("configs"), note="Describe the reviewed change"))'
```

Preserve existing frozen manifests and their parent chain. Never update a hash
to hide an unexplained difference. Frozen episode changes require reviewing the
numeric records and goal-image hashes as well as updating their manifest. Run
`bash scripts/verify.sh` after the final manifest is created.

## Pull requests

Open a branch from the current default branch. Explain the problem, changed
behavior and validation commands. Include a regression test for a behavioral
fix and update documentation for public-facing changes. Use commit messages
such as `fix(maps): reject an invalid view` or `docs(readme): clarify setup`.
Report skipped or unavailable checks explicitly; a CPU test is not a GPU result.

For vulnerabilities, follow [SECURITY.md](SECURITY.md). For other bugs, include
the command, version, expected result and redacted error output in an issue.
