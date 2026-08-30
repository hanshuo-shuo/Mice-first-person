#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --job-name=autoresearch-gaze-shard
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
git_commit="${QUEST_GIT_COMMIT:-}"
mode="${AUTORESEARCH_SHARD_MODE:-}"
run_tag="${AUTORESEARCH_RUN_TAG:-}"
evaluation_tag="${AUTORESEARCH_EVALUATION_TAG:-}"
candidate_source="${AUTORESEARCH_CANDIDATE_SOURCE:-}"
candidate_sha256="${AUTORESEARCH_CANDIDATE_SHA256:-}"
incumbent_source="${AUTORESEARCH_INCUMBENT_SOURCE:-}"
incumbent_sha256="${AUTORESEARCH_INCUMBENT_SHA256:-}"
shard_count="${AUTORESEARCH_SHARD_COUNT:-16}"
task_id="${SLURM_ARRAY_TASK_ID:-}"
config_path="configs/autoresearch/gaze_dev.yaml"
seed_set="development"

if [[ ! "${git_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must be a 40-character commit SHA." >&2
    exit 1
fi
if [[ "${mode}" != "baseline" && "${mode}" != "experiment" ]]; then
    echo "Error: AUTORESEARCH_SHARD_MODE must be baseline or experiment." >&2
    exit 1
fi
if [[ ! "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]]; then
    echo "Error: invalid AUTORESEARCH_RUN_TAG." >&2
    exit 1
fi
if [[ ! "${evaluation_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]]; then
    echo "Error: invalid AUTORESEARCH_EVALUATION_TAG." >&2
    exit 1
fi
if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]] || (( shard_count > 128 )); then
    echo "Error: AUTORESEARCH_SHARD_COUNT must lie in 1..128." >&2
    exit 1
fi
if [[ ! "${task_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is required." >&2
    exit 1
fi
if [[ ! "${candidate_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: AUTORESEARCH_CANDIDATE_SHA256 must be a lowercase SHA-256." >&2
    exit 1
fi
if [[ ! "${incumbent_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: AUTORESEARCH_INCUMBENT_SHA256 must be a lowercase SHA-256." >&2
    exit 1
fi
base_dir="results/autoresearch/${run_tag}/quest/${seed_set}"
expected_candidate_source="${base_dir}/sources/candidates/${candidate_sha256}.py"
expected_incumbent_source="${base_dir}/sources/incumbents/${incumbent_sha256}.py"
if [[ "${candidate_source}" != "${expected_candidate_source}" || "${incumbent_source}" != "${expected_incumbent_source}" ]]; then
    echo "Error: Slurm accepts only commit-extracted versioned controller snapshots." >&2
    exit 1
fi

if [[ "${mode}" == "baseline" ]]; then
    maximum_tasks=$((2 * shard_count))
    if (( task_id >= maximum_tasks )); then
        echo "Error: baseline array task lies outside its exact method/shard grid." >&2
        exit 1
    fi
    method_index=$((task_id / shard_count))
    shard_index=$((task_id % shard_count))
    methods=(incumbent fixed_p60)
    method="${methods[${method_index}]}"
    output_dir="${base_dir}/baselines/${incumbent_sha256}"
else
    if (( task_id >= shard_count )); then
        echo "Error: experiment array task lies outside its exact candidate shard grid." >&2
        exit 1
    fi
    method="candidate"
    shard_index="${task_id}"
    output_dir="${base_dir}/experiments/${evaluation_tag}/against-${incumbent_sha256}"
fi

cd "${project_dir}"
worktree_commit="${project_dir##*/}"
if [[ "${worktree_commit}" != "${git_commit}" || ! -f "${project_dir}/.git" ]]; then
    echo "Error: Quest commit worktree identity does not match submission." >&2
    exit 1
fi
IFS= read -r gitdir_line < "${project_dir}/.git"
if [[ "${gitdir_line}" != "gitdir: "* ]]; then
    echo "Error: Quest project is not a detached commit worktree." >&2
    exit 1
fi
if [[ ! -f "${config_path}" || ! -f "${candidate_source}" || ! -f "${incumbent_source}" ]]; then
    echo "Error: registered config or versioned controller snapshot is missing." >&2
    exit 1
fi
incumbent_digest_line="$(sha256sum "${incumbent_source}")"
if [[ "${incumbent_digest_line%% *}" != "${incumbent_sha256}" ]]; then
    echo "Error: incumbent source changed after submission." >&2
    exit 1
fi
candidate_digest_line="$(sha256sum "${candidate_source}")"
if [[ "${candidate_digest_line%% *}" != "${candidate_sha256}" ]]; then
    echo "Error: candidate snapshot changed after submission." >&2
    exit 1
fi
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

echo "autoresearch shard mode=${mode} evaluation=${evaluation_tag} method=${method} shard=${shard_index}/${shard_count}"

conda run --no-capture-output --name "${env_name}" \
    python -B -m autoresearch.sharding rollout \
    --config "${config_path}" \
    --seed-set "${seed_set}" \
    --method "${method}" \
    --shard-index "${shard_index}" \
    --shard-count "${shard_count}" \
    --repeat 2 \
    --candidate-source "${candidate_source}" \
    --incumbent-source "${incumbent_source}" \
    --output-dir "${output_dir}" \
    --project-root "${project_dir}"

echo "autoresearch shard complete"
