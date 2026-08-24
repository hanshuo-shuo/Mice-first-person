#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=pcie
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=matched-sac-accept
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:-}"
conditions=(active fixed_center fixed_p60 fixed_scan)

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi

cd "${project_dir}"
mkdir -p .cache/matplotlib .cache/torch results/sac_matched_acceptance slurm_logs
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export TORCH_HOME="${project_dir}/.cache/torch"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
unset OPENROUTER_API_KEY

echo "Matched acceptance git_commit=${git_commit} node=${SLURM_JOB_NODELIST}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

for condition in "${conditions[@]}"; do
    experiment_id="accept_${condition}_${SLURM_JOB_ID}"
    conda run --no-capture-output --name "${env_name}" \
        python -B train_first_person_sac.py \
        --config configs/sac_matched_generalization.yaml \
        --condition "${condition}" \
        --seed 2026082499 \
        --experiment-id "${experiment_id}" \
        --output-root results/sac_matched_acceptance \
        --smoke \
        --num-envs 4 \
        --device cuda
    output_dir="${project_dir}/results/sac_matched_acceptance/${experiment_id}"
    conda run --no-capture-output --name "${env_name}" \
        python -B evaluate_matched_sac.py \
        --config "${output_dir}/resolved_config.yaml" \
        --model "${output_dir}/best/best_model.zip" \
        --output-dir "${output_dir}/test_smoke" \
        --split test \
        --name test_smoke \
        --count 1 \
        --device cuda
done

echo "Matched CUDA acceptance complete"
