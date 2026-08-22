#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --job-name=exp01-vlm-gap
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
experiment_id="${EXP01_EXPERIMENT_ID:-exp01_perception_action_gap_${SLURM_JOB_ID}}"
git_commit="${QUEST_GIT_COMMIT:-}"

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "Error: OPENROUTER_API_KEY must be supplied through the Quest job environment." >&2
    exit 1
fi

cd "${project_dir}"
mkdir -p .cache/matplotlib results/peekbench slurm_logs

export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy

echo "EXP-01 experiment_id=${experiment_id}"
echo "EXP-01 git_commit=${git_commit}"

conda run --no-capture-output --name "${env_name}" \
    python -B -m benchmarks.peekbench exp01 \
    --config configs/peekbench/exp01.yaml \
    --experiment-id "${experiment_id}"

echo "EXP-01 summary=${project_dir}/results/peekbench/${experiment_id}/exp01_summary.json"
