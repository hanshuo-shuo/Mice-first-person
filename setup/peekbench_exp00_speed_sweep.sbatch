#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --array=0-7
#SBATCH --job-name=exp00-speed
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:-}"
speed_ratios=(${EXP00_SPEED_RATIOS:-0.08 0.09 0.095 0.10 0.12 0.15 0.25 0.35})
task_id="${SLURM_ARRAY_TASK_ID:-0}"

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi

if (( task_id < 0 || task_id >= ${#speed_ratios[@]} )); then
    echo "Error: task ${task_id} has no configured ratio." >&2
    echo "Configured ratios: ${speed_ratios[*]}" >&2
    exit 1
fi

ratio="${speed_ratios[${task_id}]}"
array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
experiment_prefix="${EXP00_EXPERIMENT_PREFIX:-exp00_gaze_oracle_speed_${array_job_id}_${task_id}}"

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

echo "EXP-00 speed ratio=${ratio}"
echo "EXP-00 speed experiment_prefix=${experiment_prefix}"
echo "EXP-00 speed git_commit=${git_commit}"

conda run --no-capture-output --name "${env_name}" \
    python -B -m benchmarks.peekbench.speed_sweep \
    --config configs/peekbench/exp00.yaml \
    --experiment-prefix "${experiment_prefix}" \
    --ratios "${ratio}"

echo "EXP-00 speed summary=${project_dir}/results/peekbench/${experiment_prefix}_speed_sweep_summary.json"
