#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"

cd "${project_dir}"
mkdir -p .cache/matplotlib logs slurm_logs Saved_Models

if command -v mamba >/dev/null 2>&1; then
    env_tool="mamba"
elif command -v conda >/dev/null 2>&1; then
    env_tool="conda"
else
    echo "Error: neither mamba nor conda is available on PATH." >&2
    exit 1
fi

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${env_name}"; then
    echo "Updating Conda environment: ${env_name}"
    "${env_tool}" env update --name "${env_name}" --file environment.yaml --prune
else
    echo "Creating Conda environment: ${env_name}"
    "${env_tool}" env create --name "${env_name}" --file environment.yaml
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy

conda run --no-capture-output --name "${env_name}" \
    python -c 'import sys; print(f"Python environment ready: {sys.executable}")'

echo "Quest setup complete: ${project_dir} (${env_name})"
echo "Run 'sbatch setup/quest_smoke.sbatch' to verify on a compute node."
