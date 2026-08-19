#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mice-sac-cnn
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
experiment_id="${SAC_CNN_EXPERIMENT_ID:-sac_cnn_active_gaze_${SLURM_JOB_ID}}"
git_commit="${QUEST_GIT_COMMIT:-}"

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi

cd "${project_dir}"
mkdir -p .cache/matplotlib .cache/torch results/sac slurm_logs

export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export TORCH_HOME="${project_dir}/.cache/torch"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONHASHSEED=20260819
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
unset OPENROUTER_API_KEY

output_dir="${project_dir}/results/sac/${experiment_id}"

echo "SAC-CNN experiment_id=${experiment_id}"
echo "SAC-CNN git_commit=${git_commit}"
echo "SAC-CNN node=${SLURM_JOB_NODELIST}"
echo "SAC-CNN cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

conda run --no-capture-output --name "${env_name}" \
    python -B train_first_person_sac.py \
    --config configs/sac_cnn_active_gaze.yaml \
    --experiment-id "${experiment_id}"

resolved_config="${output_dir}/resolved_config.yaml"
best_model="${output_dir}/best/best_model.zip"
final_model="${output_dir}/checkpoints/final_model.zip"

if [[ ! -f "${best_model}" || ! -f "${final_model}" ]]; then
    echo "Error: training did not produce both best and final checkpoints." >&2
    exit 1
fi

conda run --no-capture-output --name "${env_name}" \
    python -B evaluate_first_person_sac.py \
    --config "${resolved_config}" \
    --model "${best_model}" \
    --output-dir "${output_dir}" \
    --name evaluation_best \
    --episodes 40 \
    --device cuda \
    --render-count 3

conda run --no-capture-output --name "${env_name}" \
    python -B evaluate_first_person_sac.py \
    --config "${resolved_config}" \
    --model "${final_model}" \
    --output-dir "${output_dir}" \
    --name evaluation_final \
    --episodes 40 \
    --device cuda \
    --render-count 0

echo "SAC-CNN results=${output_dir}"
