#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-99
#SBATCH --job-name=matched-sac-test
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:-}"
train_array_id="${MATCHED_TRAIN_ARRAY_JOB_ID:-}"
conditions=(active fixed_center fixed_p60 fixed_scan)
seeds=(2026082401 2026082402 2026082403 2026082404 2026082405)
shard_count=5
tasks_per_shard=200
task_id="${SLURM_ARRAY_TASK_ID:-0}"
run_index=$((task_id / shard_count))
shard_index=$((task_id % shard_count))
condition_index=$((run_index / 5))
seed_index=$((run_index % 5))

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi
if [[ ! "${train_array_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: MATCHED_TRAIN_ARRAY_JOB_ID must contain the training array ID." >&2
    exit 1
fi

condition="${conditions[${condition_index}]}"
training_seed="${seeds[${seed_index}]}"
experiment_id="matched_v1_${condition}_seed${training_seed}_${train_array_id}"
source_dir="${project_dir}/results/sac_matched/${experiment_id}"
evaluation_array_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
output_dir="${project_dir}/results/sac_matched/matched_v1_matrix_${train_array_id}/test_${evaluation_array_id}/${condition}_seed${training_seed}/shards"
start_index=$((shard_index * tasks_per_shard))

cd "${project_dir}"
mkdir -p .cache/matplotlib "${output_dir}" slurm_logs
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
unset OPENROUTER_API_KEY

echo "Matched test condition=${condition} seed=${training_seed} shard=${shard_index}/${shard_count}"
conda run --no-capture-output --name "${env_name}" \
    python -B evaluate_matched_sac.py \
    --config "${source_dir}/resolved_config.yaml" \
    --model "${source_dir}/best/best_model.zip" \
    --output-dir "${output_dir}" \
    --split test \
    --name "test_shard_${shard_index}" \
    --start-index "${start_index}" \
    --count "${tasks_per_shard}" \
    --device cpu

echo "Matched test shard complete=${output_dir}"
