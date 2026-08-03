#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${project_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv-memorax/bin/python}"
config_dir="${project_dir}/stream_rl/experiments/configs/campaigns/passive_tmaze_ppo_gru_100k"

cd "${project_dir}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl-passive-tmaze-ppo-gru-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/rl-passive-tmaze-ppo-gru-cache}"

for length in 2 3 5 10 15 30 50; do
  for seed in 0 1 2; do
    echo "Starting Passive T-Maze PPO-GRU: L=${length}, seed=${seed}, budget=100000 steps"
    "${python_bin}" -m stream_rl.experiments.runners.base_runner \
      --runner-config "${config_dir}/runner_L${length}.yaml" \
      --seed "${seed}"
  done
done

echo "Passive T-Maze PPO-GRU sweep completed."
