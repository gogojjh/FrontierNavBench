#!/usr/bin/env bash
# Prepare Isaac Sim and CPU SRU environments; this script never starts a simulator.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dry_run=0
case "${1:-}" in
    --dry-run) dry_run=1 ;;
    -h|--help)
        echo 'Usage: bash setup/simulation.sh [--dry-run]'
        echo 'Overrides: VNAV_CONDA (conda), VNAV_SIM_ENV (vnav-sim), VNAV_SRU_ENV (vnav-sru)'
        exit 0 ;;
    '') ;;
    *) echo 'Unknown argument; use --help.' >&2; exit 2 ;;
esac
if (($# > 1)); then echo 'Too many arguments.' >&2; exit 2; fi
conda_bin=${VNAV_CONDA:-conda}
sim_env=${VNAV_SIM_ENV:-vnav-sim}
sru_env=${VNAV_SRU_ENV:-vnav-sru}
if [[ ! "$sim_env" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ||
      ! "$sru_env" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ || "$sim_env" == "$sru_env" ||
      "$sim_env" == base || "$sru_env" == base ]]; then
    echo 'Choose two distinct, non-base Conda environment names.' >&2
    exit 2
fi
run() {
    if ((dry_run)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if ((!dry_run)); then
    # Refuse existing names before installing anything; do not reuse active environments.
    env_json=$("$conda_bin" env list --json)
    VNAV_ENV_JSON="$env_json" python3 - "$sim_env" "$sru_env" <<'PY'
import json, os, sys
from pathlib import Path
existing = {Path(p).name for p in json.loads(os.environ["VNAV_ENV_JSON"])["envs"]}
if existing.intersection(sys.argv[1:]):
    sys.exit("Environment already exists; choose new VNAV_SIM_ENV and VNAV_SRU_ENV.")
PY
fi
run "$conda_bin" create --yes --name "$sim_env" python=3.11 pip
run "$conda_bin" run --no-capture-output --name "$sim_env" python -m pip install --upgrade pip
run "$conda_bin" run --no-capture-output --name "$sim_env" python -m pip install \
    -r "$repo_root/setup/requirements-simulation.txt"
run "$conda_bin" run --no-capture-output --name "$sim_env" python -m pip install \
    -e "$repo_root"
run "$conda_bin" create --yes --name "$sru_env" python=3.10 pip
run "$conda_bin" run --no-capture-output --name "$sru_env" python -m pip install \
    -e "$repo_root" -r "$repo_root/setup/requirements-sru.txt"
if ((dry_run)); then
    echo 'Dry-run only; no environments were created.'
else
    echo 'Environments ready. Download the required assets before running the simulator.'
    echo "Use VNAV_SIM_ENV=$sim_env or set VNAV_SIM_PYTHON to its Python executable."
fi
