#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=exp05-gaze-aggregate
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:-}"
source_experiment="${SAC_SOURCE_EXPERIMENT_ID:-sac_cnn_active_gaze_9903898}"
array_job_id="${EXP05_ARRAY_JOB_ID:-}"
seed_start="${EXP05_SEED_START:-1000000}"
episodes="${EXP05_EPISODES:-1000}"
shard_count=10

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi
if [[ ! "${array_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: EXP05_ARRAY_JOB_ID must contain the shard array job ID." >&2
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
output_dir="${source_dir}/exp05_gaze_ablation_1000_${array_job_id}"

echo "EXP-05 aggregate array_job_id=${array_job_id}"
echo "EXP-05 git_commit=${git_commit}"

conda run --no-capture-output --name "${env_name}" \
    python -B -m analysis.sac_gaze_ablation aggregate \
    --config "${config_path}" \
    --model "${model_path}" \
    --output-dir "${output_dir}" \
    --episodes "${episodes}" \
    --seed-start "${seed_start}" \
    --shard-count "${shard_count}"

echo "EXP-05 results=${output_dir}"
