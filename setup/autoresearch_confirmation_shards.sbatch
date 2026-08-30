#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --job-name=autoresearch-confirm-shard
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:?QUEST_PROJECT_DIR is required}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:?QUEST_GIT_COMMIT is required}"
run_tag="${AUTORESEARCH_RUN_TAG:?AUTORESEARCH_RUN_TAG is required}"
authorization="${AUTORESEARCH_CONFIRM_AUTH:?AUTORESEARCH_CONFIRM_AUTH is required}"
authorization_sha256="${AUTORESEARCH_CONFIRM_AUTH_SHA256:?AUTORESEARCH_CONFIRM_AUTH_SHA256 is required}"
driver_sha256="${AUTORESEARCH_CONFIRM_DRIVER_SHA256:?AUTORESEARCH_CONFIRM_DRIVER_SHA256 is required}"
candidate_source="${AUTORESEARCH_CANDIDATE_SOURCE:?AUTORESEARCH_CANDIDATE_SOURCE is required}"
candidate_sha256="${AUTORESEARCH_CANDIDATE_SHA256:?AUTORESEARCH_CANDIDATE_SHA256 is required}"
incumbent_source="${AUTORESEARCH_INCUMBENT_SOURCE:?AUTORESEARCH_INCUMBENT_SOURCE is required}"
incumbent_sha256="${AUTORESEARCH_INCUMBENT_SHA256:?AUTORESEARCH_INCUMBENT_SHA256 is required}"
shard_count="${AUTORESEARCH_SHARD_COUNT:-50}"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed_set="confirmation"
config_path="configs/autoresearch/gaze_dev.yaml"
base_dir="results/autoresearch/${run_tag}/quest/confirmation/C0001"
output_dir="${base_dir}/evaluation"

if [[ ! "${git_commit}" =~ ^[0-9a-f]{40}$ || "${project_dir##*/}" != "${git_commit}" ]]; then
    echo "Error: confirmation worktree commit identity mismatch." >&2
    exit 1
fi
if [[ ! "${authorization_sha256}" =~ ^[0-9a-f]{64}$ || ! "${driver_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: confirmation authorization/driver hash is invalid." >&2
    exit 1
fi
if [[ ! "${candidate_sha256}" =~ ^[0-9a-f]{64}$ || ! "${incumbent_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: confirmation controller hash is invalid." >&2
    exit 1
fi
if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]] || (( shard_count > 1000 )); then
    echo "Error: confirmation shard count is invalid." >&2
    exit 1
fi
maximum_tasks=$((3 * shard_count))
if [[ ! "${task_id}" =~ ^[0-9]+$ ]] || (( task_id >= maximum_tasks )); then
    echo "Error: confirmation array task is outside the exact method grid." >&2
    exit 1
fi

methods=(candidate incumbent fixed_p60)
method_index=$((task_id / shard_count))
shard_index=$((task_id % shard_count))
method="${methods[${method_index}]}"

cd "${project_dir}"
IFS= read -r gitdir_line < "${project_dir}/.git"
if [[ "${gitdir_line}" != "gitdir: "* ]]; then
    echo "Error: confirmation project is not a detached commit worktree." >&2
    exit 1
fi
for identity in \
    "${authorization}:${authorization_sha256}" \
    "autoresearch/confirmation.py:${driver_sha256}" \
    "${candidate_source}:${candidate_sha256}" \
    "${incumbent_source}:${incumbent_sha256}"; do
    path="${identity%%:*}"
    expected="${identity##*:}"
    if [[ ! -f "${path}" || -L "${path}" ]]; then
        echo "Error: confirmation identity file is missing or a symlink: ${path}" >&2
        exit 1
    fi
    digest_line="$(sha256sum "${path}")"
    if [[ "${digest_line%% *}" != "${expected}" ]]; then
        echo "Error: confirmation identity hash mismatch: ${path}" >&2
        exit 1
    fi
done

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

echo "confirmation method=${method} shard=${shard_index}/${shard_count}"
conda run --no-capture-output --name "${env_name}" \
    python -B -m autoresearch.confirmation \
    --repo-root "${project_dir}" \
    rollout \
    --run-tag "${run_tag}" \
    --config "${config_path}" \
    --candidate-source "${candidate_source}" \
    --incumbent-source "${incumbent_source}" \
    --authorization "${authorization}" \
    --authorization-sha256 "${authorization_sha256}" \
    --output-dir "${output_dir}" \
    --shard-count "${shard_count}" \
    --method "${method}" \
    --shard-index "${shard_index}"

