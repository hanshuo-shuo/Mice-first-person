#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=exp04-gaze
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
experiment_id="${EXP04_EXPERIMENT_ID:-exp04_active_gaze_equal_compute_${SLURM_JOB_ID}}"
git_commit="${QUEST_GIT_COMMIT:-}"

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
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
unset OPENROUTER_API_KEY

echo "EXP-04 experiment_id=${experiment_id}"
echo "EXP-04 git_commit=${git_commit}"

conda run --no-capture-output --name "${env_name}" \
  python -B -m benchmarks.peekbench exp04 \
  --config configs/peekbench/exp04.yaml \
  --experiment-id "${experiment_id}"

echo "EXP-04 summary=${project_dir}/results/peekbench/${experiment_id}/exp04_summary.json"
