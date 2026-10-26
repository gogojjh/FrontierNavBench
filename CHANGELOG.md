# Changelog

## Unreleased

- Name the project FrontierNavBench and highlight its GPT-6 Astra navigation comparisons.
- Rename the checkout directory to `FrontierNavBench` and the Python package to
  `frontier_nav_bench`, with matching imports, scripts and documentation.
- Remove redundant test scaffolding and unused fixture summaries while retaining
  navigation, network-isolation and golden-replay coverage.
- Keep only `main` locally and on GitHub, with all published `opensource`
  changes retained. Disable scheduled Dependabot branch creation.
- Rename baseline maps consistently to `R01_dense`/`R01_sparse` through R15,
  preserving all map data and numeric task definitions.
- Ship portable offline regression tests and a shared local/CI check command.
- Add contributor, security, architecture and data-layout documentation.
- Add README visuals drawn from bundled inputs and identified recorded output.
- Remove unused runtime code for paid active-map selection; fixed-selection
  ablations remain available.

## 0.2.0 — 2026-10-26

- Distribute evaluation code, maps, tasks, scene assets and model weights in a
  standalone source checkout with integrity and CPU smoke checks.
