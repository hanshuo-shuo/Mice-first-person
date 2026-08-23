#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=matched-sac-aggregate
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
train_array_id="${MATCHED_TRAIN_ARRAY_JOB_ID:-}"
test_array_id="${MATCHED_TEST_ARRAY_JOB_ID:-}"

if [[ ! "${train_array_id}" =~ ^[0-9]+$ || ! "${test_array_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: matched train/test array IDs are required." >&2
    exit 1
fi

cd "${project_dir}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
unset OPENROUTER_API_KEY

result_root="${project_dir}/results/sac_matched/matched_v1_matrix_${train_array_id}/test_${test_array_id}"
conda run --no-capture-output --name "${env_name}" \
    python -B -m analysis.matched_training_aggregate \
    --result-root "${result_root}" \
    --shard-count 5

echo "Matched aggregate complete=${result_root}/aggregate"
