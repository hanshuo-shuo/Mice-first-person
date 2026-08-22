#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sac-density-1k
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:-}"
source_experiment="${SAC_SOURCE_EXPERIMENT_ID:-sac_cnn_active_gaze_9903898}"
seed_start="${SAC_DENSITY_SEED_START:-1000000}"
episodes="${SAC_DENSITY_EPISODES:-1000}"
workers="${SAC_DENSITY_WORKERS:-8}"

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi

cd "${project_dir}"
mkdir -p .cache/matplotlib results/sac slurm_logs

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

source_dir="${project_dir}/results/sac/${source_experiment}"
config_path="${source_dir}/resolved_config.yaml"
model_path="${source_dir}/checkpoints/final_model.zip"
output_dir="${source_dir}/trajectory_density_1000_${SLURM_JOB_ID}"

if [[ ! -f "${config_path}" || ! -f "${model_path}" ]]; then
    echo "Error: source config/model missing under ${source_dir}." >&2
    exit 1
fi

echo "SAC density source=${source_experiment}"
echo "SAC density git_commit=${git_commit}"
echo "SAC density episodes=${episodes} seed_start=${seed_start} workers=${workers}"

conda run --no-capture-output --name "${env_name}" \
    python -B -m analysis.sac_trajectory_density \
    --config "${config_path}" \
    --model "${model_path}" \
    --output-dir "${output_dir}" \
    --episodes "${episodes}" \
    --seed-start "${seed_start}" \
    --workers "${workers}"

echo "SAC density results=${output_dir}"
