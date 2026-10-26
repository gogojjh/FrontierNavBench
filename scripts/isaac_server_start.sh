#!/usr/bin/env bash
# Start an isolated process group. All simulator arguments remain individual argv entries.
set -euo pipefail
if (( $# < 6 )); then
  echo 'Usage: isaac_server_start.sh LOG_DIR SCENE X Y YAW_DEGREES PORT [SERVER_ARGS...]' >&2
  exit 2
fi
log_dir=$1; scene=$2; x=$3; y=$4; yaw=$5; port=$6; shift 6
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p -- "$log_dir"
log="$log_dir/isaac_${port}_$(date +%H%M%S).log"
# Use a bootstrap interpreter only to read configuration, never to import Isaac.
config_python=${VNAV_CONFIG_PYTHON:-python3}
read_setting() {
  PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" "$config_python" -c \
    'import sys; from vnav_benchmark.paths import setting; print(setting(sys.argv[1], sys.argv[2]))' \
    "$1" "$2"
}
sim_python=$(read_setting VNAV_SIM_PYTHON '')
if [[ -n "$sim_python" ]]; then
  interpreter=("$sim_python")
else
  conda_bin=$(read_setting VNAV_CONDA conda)
  sim_env=$(read_setting VNAV_SIM_ENV vnav-sim)
  interpreter=("$conda_bin" run --no-capture-output --name "$sim_env" python)
fi
export OMNI_KIT_ACCEPT_EULA=YES CLEANUP_PKILL_BY_NAME=0
export PYTHONPATH="$repo_root"
cd -- "$repo_root"
nohup setsid "${interpreter[@]}" -m vnav_benchmark.sim.isaac_server \
  --scene "$scene" --start "$x" "$y" "$yaw" --port "$port" "$@" \
  > "$log" 2>&1 < /dev/null &
echo "ISAAC_PID $! LOG $log"
