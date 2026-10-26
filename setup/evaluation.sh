#!/usr/bin/env bash
# Create an isolated evaluation environment. Never activate or alter a running one.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dry_run=0
case "${1:-}" in
    --dry-run) dry_run=1 ;;
    -h|--help)
        echo 'Usage: bash setup/evaluation.sh [--dry-run]'
        echo 'Overrides: VNAV_EVAL_PYTHON (python3.8), VNAV_EVAL_VENV (.venv-eval)'
        exit 0 ;;
    '') ;;
    *) echo 'Unknown argument; use --help.' >&2; exit 2 ;;
esac
if (($# > 1)); then echo 'Too many arguments.' >&2; exit 2; fi
python_bin=${VNAV_EVAL_PYTHON:-python3.8}
venv_dir=${VNAV_EVAL_VENV:-$repo_root/.venv-eval}
run() {
    if ((dry_run)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if [[ -e "$venv_dir" || -L "$venv_dir" ]]; then
    echo 'Environment already exists; choose a new VNAV_EVAL_VENV.' >&2
    exit 1
fi
run "$python_bin" -c 'import sys; assert sys.version_info[:2] in ((3, 8), (3, 10)), "Use Python 3.8 or 3.10"'
if "$python_bin" -c 'import ensurepip' >/dev/null 2>&1; then
    run "$python_bin" -m venv "$venv_dir"
elif command -v uv >/dev/null 2>&1; then
    # Some distro Python installations omit ensurepip; uv can seed an isolated venv.
    run uv venv --python "$python_bin" --seed "$venv_dir"
else
    echo 'Install the matching python-venv package (or uv), then retry.' >&2
    exit 1
fi
run "$venv_dir/bin/python" -m pip install 'pip<25.1' 'setuptools>=61,<75' wheel
run "$venv_dir/bin/python" -m pip install -r "$repo_root/setup/requirements-evaluation.txt"
run "$venv_dir/bin/python" -m pip install --no-build-isolation -e "$repo_root[evaluation]"
if ((dry_run)); then
    echo 'Dry-run only; no environment was created.'
else
    echo 'Evaluation environment prepared. See README.md for runtime requirements.'
fi
