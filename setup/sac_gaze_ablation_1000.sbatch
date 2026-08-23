#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-59
#SBATCH --job-name=exp05-gaze-shard
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:-}"
source_experiment="${SAC_SOURCE_EXPERIMENT_ID:-sac_cnn_active_gaze_9903898}"
seed_start="${EXP05_SEED_START:-1000000}"
episodes="${EXP05_EPISODES:-1000}"
shard_count=10
methods=(
    sac_active_gaze
    sac_fixed_0
    sac_fixed_p60
    sac_fixed_m60
    sac_fixed_p30
    sac_fixed_scan
)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
method_index=$((task_id / shard_count))
shard_index=$((task_id % shard_count))

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
array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
output_dir="${source_dir}/exp05_gaze_ablation_1000_${array_job_id}"

if [[ ! -f "${config_path}" || ! -f "${model_path}" ]]; then
    echo "Error: source config/model missing under ${source_dir}." >&2
    exit 1
fi
if (( method_index < 0 || method_index >= ${#methods[@]} )); then
    echo "Error: array task ${task_id} has no method assignment." >&2
    exit 1
fi
method="${methods[${method_index}]}"

echo "EXP-05 source=${source_experiment}"
echo "EXP-05 git_commit=${git_commit}"
echo "EXP-05 method=${method} shard=${shard_index}/${shard_count}"
echo "EXP-05 episodes=${episodes} seed_start=${seed_start}"

conda run --no-capture-output --name "${env_name}" \
    python -B -m analysis.sac_gaze_ablation rollout \
    --config "${config_path}" \
    --model "${model_path}" \
    --output-dir "${output_dir}" \
    --episodes "${episodes}" \
    --seed-start "${seed_start}" \
    --method "${method}" \
    --shard-index "${shard_index}" \
    --shard-count "${shard_count}"

echo "EXP-05 shard complete=${output_dir}/shards"
