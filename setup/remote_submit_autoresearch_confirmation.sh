#!/usr/bin/env bash

set -euo pipefail

run_tag="$1"
driver_branch="$2"
driver_commit="$3"
driver_sha256="$4"
candidate_branch="$5"
candidate_commit="$6"
candidate_sha256="$7"
incumbent_commit="$8"
incumbent_sha256="$9"
authorization_path="${10}"
authorization_sha256="${11}"
shard_count="${12}"
remote_project="${13}"

if [[ ! "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]]; then
    echo "Error: invalid confirmation run tag." >&2
    exit 1
fi
if ! git check-ref-format --branch "${driver_branch}" >/dev/null 2>&1 || ! git check-ref-format --branch "${candidate_branch}" >/dev/null 2>&1; then
    echo "Error: invalid confirmation branch identity." >&2
    exit 1
fi
if [[ ! "${driver_commit}" =~ ^[0-9a-f]{40}$ || ! "${candidate_commit}" =~ ^[0-9a-f]{40}$ || ! "${incumbent_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Error: invalid confirmation commit identity." >&2
    exit 1
fi
if [[ ! "${driver_sha256}" =~ ^[0-9a-f]{64}$ || ! "${candidate_sha256}" =~ ^[0-9a-f]{64}$ || ! "${incumbent_sha256}" =~ ^[0-9a-f]{64}$ || ! "${authorization_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: invalid confirmation SHA-256 identity." >&2
    exit 1
fi
if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]] || (( shard_count > 1000 )); then
    echo "Error: invalid confirmation shard count." >&2
    exit 1
fi

case "${remote_project}" in
    /*) main_project_dir="${remote_project}" ;;
    *) main_project_dir="${HOME}/${remote_project}" ;;
esac
cd "${main_project_dir}"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: Quest main checkout has tracked modifications." >&2
    exit 1
fi
git fetch --quiet origin "${driver_branch}" "${candidate_branch}"
if [[ "$(git rev-parse "origin/${driver_branch}")" != "${driver_commit}" ]]; then
    echo "Error: driver branch no longer resolves to submitted commit." >&2
    exit 1
fi
if [[ "$(git rev-parse "origin/${candidate_branch}")" != "${candidate_commit}" ]]; then
    echo "Error: candidate branch no longer resolves to submitted commit." >&2
    exit 1
fi
if ! git cat-file -e "${incumbent_commit}^{commit}" 2>/dev/null; then
    echo "Error: confirmation incumbent commit is unavailable." >&2
    exit 1
fi

worktree_root="${HOME}/projects/Mice-autoresearch-worktrees"
project_dir="${worktree_root}/${driver_commit}"
mkdir -p "${worktree_root}"
if [[ -e "${project_dir}" || -L "${project_dir}" ]]; then
    if [[ -L "${project_dir}" || ! -d "${project_dir}" || "$(git -C "${project_dir}" rev-parse HEAD)" != "${driver_commit}" ]]; then
        echo "Error: existing confirmation driver worktree is unsafe." >&2
        exit 1
    fi
else
    git worktree add --quiet --detach "${project_dir}" "${driver_commit}"
fi
if [[ -n "$(git -C "${project_dir}" status --porcelain)" ]]; then
    echo "Error: confirmation driver worktree is not clean." >&2
    exit 1
fi

main_results="${main_project_dir}/results/autoresearch"
main_exp05="${main_project_dir}/results/sac/sac_cnn_active_gaze_9903898"
if [[ ! -d "${main_results}" || -L "${main_results}" || ! -d "${main_exp05}" || -L "${main_exp05}" ]]; then
    echo "Error: shared confirmation result/source roots are unsafe." >&2
    exit 1
fi
ensure_link() {
    link_path="$1"
    target_path="$2"
    mkdir -p "$(dirname -- "${link_path}")"
    if [[ -L "${link_path}" ]]; then
        [[ "$(readlink -f "${link_path}")" == "$(readlink -f "${target_path}")" ]] || exit 1
    elif [[ -e "${link_path}" ]]; then
        echo "Error: confirmation link path contains non-symlink data." >&2
        exit 1
    else
        ln -s "${target_path}" "${link_path}"
    fi
}
ensure_link "${project_dir}/results/autoresearch" "${main_results}"
ensure_link "${project_dir}/results/sac/sac_cnn_active_gaze_9903898" "${main_exp05}"
cd "${project_dir}"

base_dir="results/autoresearch/${run_tag}/quest/confirmation/C0001"
source_dir="${base_dir}/sources"
mkdir -p "${source_dir}/candidates" "${source_dir}/incumbents" slurm_logs
snapshot_from_commit() {
    source_commit="$1"
    expected_sha="$2"
    destination="$3"
    if [[ -f "${destination}" && ! -L "${destination}" ]]; then
        digest_line="$(sha256sum "${destination}")"
        [[ "${digest_line%% *}" == "${expected_sha}" ]] || exit 1
        return
    fi
    temporary="$(mktemp "${destination}.tmp.XXXXXX")"
    git show "${source_commit}:autoresearch/candidate.py" > "${temporary}"
    digest_line="$(sha256sum "${temporary}")"
    [[ "${digest_line%% *}" == "${expected_sha}" ]] || { rm -f "${temporary}"; exit 1; }
    chmod 0444 "${temporary}"
    mv "${temporary}" "${destination}"
}
candidate_source="${source_dir}/candidates/${candidate_sha256}.py"
incumbent_source="${source_dir}/incumbents/${incumbent_sha256}.py"
snapshot_from_commit "${candidate_commit}" "${candidate_sha256}" "${candidate_source}"
snapshot_from_commit "${incumbent_commit}" "${incumbent_sha256}" "${incumbent_source}"

if [[ "${authorization_path}" != "${base_dir}/control/authorization.json" || ! -f "${authorization_path}" || -L "${authorization_path}" ]]; then
    echo "Error: spent confirmation authorization marker is missing or misplaced." >&2
    exit 1
fi
auth_digest_line="$(sha256sum "${authorization_path}")"
driver_digest_line="$(sha256sum autoresearch/confirmation.py)"
if [[ "${auth_digest_line%% *}" != "${authorization_sha256}" || "${driver_digest_line%% *}" != "${driver_sha256}" ]]; then
    echo "Error: confirmation authorization or driver hash mismatch." >&2
    exit 1
fi
if [[ ! -f "results/autoresearch/${run_tag}/run.json" || ! -f "results/autoresearch/${run_tag}/run.sha256" ]]; then
    echo "Error: frozen run manifest was not synchronized for confirmation." >&2
    exit 1
fi

unset OPENROUTER_API_KEY
exports="ALL,QUEST_PROJECT_DIR=${project_dir},QUEST_GIT_COMMIT=${driver_commit},AUTORESEARCH_RUN_TAG=${run_tag},AUTORESEARCH_CONFIRM_AUTH=${authorization_path},AUTORESEARCH_CONFIRM_AUTH_SHA256=${authorization_sha256},AUTORESEARCH_CONFIRM_DRIVER_SHA256=${driver_sha256},AUTORESEARCH_CANDIDATE_SOURCE=${candidate_source},AUTORESEARCH_CANDIDATE_SHA256=${candidate_sha256},AUTORESEARCH_INCUMBENT_SOURCE=${incumbent_source},AUTORESEARCH_INCUMBENT_SHA256=${incumbent_sha256},AUTORESEARCH_SHARD_COUNT=${shard_count}"
array_last=$((3 * shard_count - 1))
array_job_id="$(sbatch --parsable --array="0-${array_last}" --export="${exports}" setup/autoresearch_confirmation_shards.sbatch)"
array_job_id="${array_job_id%%;*}"
if [[ ! "${array_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid confirmation shard job ID." >&2
    exit 1
fi
aggregate_job_id="$(sbatch --parsable --dependency="afterok:${array_job_id}" --export="${exports},AUTORESEARCH_ARRAY_JOB_ID=${array_job_id}" setup/autoresearch_confirmation_aggregate.sbatch)"
aggregate_job_id="${aggregate_job_id%%;*}"
if [[ ! "${aggregate_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid confirmation aggregate job ID." >&2
    exit 1
fi
printf 'AUTORESEARCH_CONFIRMATION_SUBMISSION|%s|%s|%s|%s\n' "${array_job_id}" "${aggregate_job_id}" "${project_dir}" "${base_dir}/evaluation"
