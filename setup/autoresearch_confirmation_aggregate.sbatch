#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=autoresearch-confirm-aggregate
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

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
config_path="configs/autoresearch/gaze_dev.yaml"
output_dir="results/autoresearch/${run_tag}/quest/confirmation/C0001/evaluation"

if [[ ! "${git_commit}" =~ ^[0-9a-f]{40}$ || "${project_dir##*/}" != "${git_commit}" ]]; then
    echo "Error: confirmation aggregate worktree identity mismatch." >&2
    exit 1
fi
cd "${project_dir}"
IFS= read -r gitdir_line < "${project_dir}/.git"
if [[ "${gitdir_line}" != "gitdir: "* ]]; then
    echo "Error: confirmation aggregate project is not a commit worktree." >&2
    exit 1
fi
for identity in \
    "${authorization}:${authorization_sha256}" \
    "autoresearch/confirmation.py:${driver_sha256}" \
    "${candidate_source}:${candidate_sha256}" \
    "${incumbent_source}:${incumbent_sha256}"; do
    path="${identity%%:*}"
    expected="${identity##*:}"
    digest_line="$(sha256sum "${path}")"
    if [[ "${digest_line%% *}" != "${expected}" ]]; then
        echo "Error: confirmation aggregate identity mismatch: ${path}" >&2
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

conda run --no-capture-output --name "${env_name}" \
    python -B -m autoresearch.confirmation \
    --repo-root "${project_dir}" \
    aggregate \
    --run-tag "${run_tag}" \
    --config "${config_path}" \
    --candidate-source "${candidate_source}" \
    --incumbent-source "${incumbent_source}" \
    --authorization "${authorization}" \
    --authorization-sha256 "${authorization_sha256}" \
    --output-dir "${output_dir}" \
    --shard-count "${shard_count}"
