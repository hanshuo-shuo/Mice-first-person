#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-19
#SBATCH --job-name=matched-sac-train
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:-}"
conditions=(active fixed_center fixed_p60 fixed_scan)
seeds=(2026082401 2026082402 2026082403 2026082404 2026082405)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
condition_index=$((task_id / 5))
seed_index=$((task_id % 5))

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi
if (( condition_index < 0 || condition_index >= ${#conditions[@]} )); then
    echo "Error: array task ${task_id} has no matched condition." >&2
    exit 1
fi

condition="${conditions[${condition_index}]}"
training_seed="${seeds[${seed_index}]}"
array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
experiment_id="matched_v1_${condition}_seed${training_seed}_${array_job_id}"

cd "${project_dir}"
mkdir -p .cache/matplotlib .cache/torch results/sac_matched slurm_logs

export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export TORCH_HOME="${project_dir}/.cache/torch"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONHASHSEED=20260824
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
unset OPENROUTER_API_KEY

echo "Matched training condition=${condition} seed=${training_seed}"
echo "Matched training experiment_id=${experiment_id}"
echo "Matched training git_commit=${git_commit}"
echo "Matched training node=${SLURM_JOB_NODELIST} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

conda run --no-capture-output --name "${env_name}" \
    python -B train_first_person_sac.py \
    --config configs/sac_matched_generalization.yaml \
    --condition "${condition}" \
    --seed "${training_seed}" \
    --experiment-id "${experiment_id}"

output_dir="${project_dir}/results/sac_matched/${experiment_id}"
best_model="${output_dir}/best/best_model.zip"
resolved_config="${output_dir}/resolved_config.yaml"
if [[ ! -f "${best_model}" || ! -f "${resolved_config}" ]]; then
    echo "Error: matched training did not produce best model/config." >&2
    exit 1
fi

conda run --no-capture-output --name "${env_name}" \
    python -B evaluate_matched_sac.py \
    --config "${resolved_config}" \
    --model "${best_model}" \
    --output-dir "${output_dir}/validation" \
    --split validation \
    --name validation_best \
    --count 256 \
    --device cuda

echo "Matched training complete=${output_dir}"
