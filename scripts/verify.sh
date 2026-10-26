#!/usr/bin/env bash
# The same offline checks locally and in CI; no simulator or model API calls.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=${VNAV_VERIFY_PYTHON:-python3}
"$python_bin" -m ruff check frontier_nav_bench scripts tests
"$python_bin" -m pytest -q
"$python_bin" scripts/check_install.py --smoke
