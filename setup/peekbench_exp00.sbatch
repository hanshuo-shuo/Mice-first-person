#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --job-name=exp00-gaze
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
experiment_id="${EXP00_EXPERIMENT_ID:-exp00_gaze_oracle_headroom_${SLURM_JOB_ID}}"

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

echo "EXP-00 experiment_id=${experiment_id}"
echo "EXP-00 git_commit=$(git rev-parse HEAD)"

conda run --no-capture-output --name "${env_name}" \
    python -B -m benchmarks.peekbench headroom \
    --config configs/peekbench/exp00.yaml \
    --experiment-id "${experiment_id}"

echo "EXP-00 summary=${project_dir}/results/peekbench/${experiment_id}/headroom_summary.json"
