#!/usr/bin/env bash

set -euo pipefail

mode="$1"
run_tag="$2"
evaluation_tag="$3"
submitted_branch="$4"
candidate_commit="$5"
candidate_sha256="$6"
incumbent_commit="$7"
incumbent_sha256="$8"
shard_count="$9"
remote_project="${10}"
submitted_commit="${candidate_commit}"

if [[ ! "${candidate_commit}" =~ ^[0-9a-f]{40}$ || "${candidate_commit}" != "${submitted_commit}" ]]; then
    echo "Error: candidate commit identity is invalid." >&2
    exit 1
fi
if [[ ! "${incumbent_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Error: incumbent commit identity is invalid." >&2
    exit 1
fi
if [[ ! "${candidate_sha256}" =~ ^[0-9a-f]{64}$ || ! "${incumbent_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: candidate/incumbent SHA-256 identity is invalid." >&2
    exit 1
fi
if [[ "${mode}" == "baseline" && ( "${candidate_commit}" != "${incumbent_commit}" || "${candidate_sha256}" != "${incumbent_sha256}" ) ]]; then
    echo "Error: baseline candidate/incumbent identities differ." >&2
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
git fetch --quiet origin "${submitted_branch}"
if [[ "$(git rev-parse FETCH_HEAD)" != "${submitted_commit}" ]]; then
    echo "Error: pushed branch no longer resolves to the submitted commit." >&2
    exit 1
fi
if ! git cat-file -e "${incumbent_commit}^{commit}" 2>/dev/null; then
    echo "Error: incumbent commit is not reachable from the fetched candidate history." >&2
    exit 1
fi

worktree_root="${HOME}/projects/Mice-autoresearch-worktrees"
project_dir="${worktree_root}/${submitted_commit}"
mkdir -p "${worktree_root}"
if [[ -e "${project_dir}" || -L "${project_dir}" ]]; then
    if [[ -L "${project_dir}" || ! -d "${project_dir}" ]]; then
        echo "Error: existing commit worktree path is unsafe." >&2
        exit 1
    fi
    if [[ "$(git -C "${project_dir}" rev-parse HEAD 2>/dev/null || true)" != "${submitted_commit}" ]]; then
        echo "Error: existing commit worktree has another HEAD." >&2
        exit 1
    fi
else
    git worktree add --quiet --detach "${project_dir}" "${submitted_commit}"
fi
if [[ -n "$(git -C "${project_dir}" status --porcelain)" ]]; then
    echo "Error: commit worktree is not clean." >&2
    exit 1
fi

main_autoresearch_results="${main_project_dir}/results/autoresearch"
main_exp05_results="${main_project_dir}/results/sac/sac_cnn_active_gaze_9903898"
mkdir -p "${main_autoresearch_results}"
if [[ ! -d "${main_autoresearch_results}" || -L "${main_autoresearch_results}" ]]; then
    echo "Error: shared autoresearch result directory is unsafe in the main checkout." >&2
    exit 1
fi
if [[ ! -d "${main_exp05_results}" || -L "${main_exp05_results}" ]]; then
    echo "Error: verified EXP-05 result directory is missing or unsafe in the main checkout." >&2
    exit 1
fi

ensure_results_link() {
    link_path="$1"
    target_path="$2"
    mkdir -p "$(dirname -- "${link_path}")"
    if [[ -L "${link_path}" ]]; then
        if [[ "$(readlink -f "${link_path}")" != "$(readlink -f "${target_path}")" ]]; then
            echo "Error: existing worktree result link points elsewhere: ${link_path}" >&2
            exit 1
        fi
        return
    fi
    if [[ -e "${link_path}" ]]; then
        echo "Error: worktree result link path already contains a non-symlink." >&2
        exit 1
    fi
    ln -s "${target_path}" "${link_path}"
}

ensure_results_link "${project_dir}/results/autoresearch" "${main_autoresearch_results}"
ensure_results_link "${project_dir}/results/sac/sac_cnn_active_gaze_9903898" "${main_exp05_results}"
cd "${project_dir}"

seed_set="development"
base_dir="results/autoresearch/${run_tag}/quest/${seed_set}"
source_dir="${base_dir}/sources"
mkdir -p "${source_dir}/candidates" "${source_dir}/incumbents" "${base_dir}/comparator_caches" slurm_logs

snapshot_controller_from_commit() {
    source_commit="$1"
    expected_sha256="$2"
    destination="$3"
    if [[ -L "${destination}" ]]; then
        echo "Error: controller snapshot may not be a symlink: ${destination}" >&2
        exit 1
    fi
    if [[ -f "${destination}" ]]; then
        existing_digest_line="$(sha256sum "${destination}")"
        if [[ "${existing_digest_line%% *}" != "${expected_sha256}" ]]; then
            echo "Error: immutable controller snapshot has a mismatched SHA-256: ${destination}" >&2
            exit 1
        fi
        chmod 0444 "${destination}"
        return
    fi
    temporary="$(mktemp "${destination}.tmp.XXXXXX")"
    if ! git show "${source_commit}:autoresearch/candidate.py" >"${temporary}"; then
        rm -f -- "${temporary}"
        echo "Error: fixed candidate source is absent from commit ${source_commit}." >&2
        exit 1
    fi
    extracted_digest_line="$(sha256sum "${temporary}")"
    if [[ "${extracted_digest_line%% *}" != "${expected_sha256}" ]]; then
        rm -f -- "${temporary}"
        echo "Error: extracted controller bytes do not match their registered SHA-256." >&2
        exit 1
    fi
    chmod 0444 "${temporary}"
    mv -- "${temporary}" "${destination}"
}

incumbent_snapshot="${source_dir}/incumbents/${incumbent_sha256}.py"
candidate_snapshot="${source_dir}/candidates/${candidate_sha256}.py"
snapshot_controller_from_commit "${incumbent_commit}" "${incumbent_sha256}" "${incumbent_snapshot}"
snapshot_controller_from_commit "${candidate_commit}" "${candidate_sha256}" "${candidate_snapshot}"
incumbent_source="${incumbent_snapshot}"
candidate_source="${candidate_snapshot}"
cache_path="${base_dir}/comparator_caches/${incumbent_sha256}.json"

if [[ "${mode}" == "baseline" ]]; then
    array_last=$((2 * shard_count - 1))
    array_dependency=()
else
    array_last=$((shard_count - 1))
    array_dependency=()
    if [[ ! -f "${cache_path}" ]]; then
        echo "Error: comparator cache is absent; wait for the baseline aggregate to finish." >&2
        exit 1
    fi
fi

unset OPENROUTER_API_KEY
exports="ALL,QUEST_PROJECT_DIR=${project_dir},QUEST_GIT_COMMIT=${submitted_commit},AUTORESEARCH_SHARD_MODE=${mode},AUTORESEARCH_RUN_TAG=${run_tag},AUTORESEARCH_EVALUATION_TAG=${evaluation_tag},AUTORESEARCH_CANDIDATE_SOURCE=${candidate_source},AUTORESEARCH_CANDIDATE_SHA256=${candidate_sha256},AUTORESEARCH_INCUMBENT_SOURCE=${incumbent_source},AUTORESEARCH_INCUMBENT_SHA256=${incumbent_sha256},AUTORESEARCH_SHARD_COUNT=${shard_count}"
array_job_id="$(
    sbatch --parsable \
        --array="0-${array_last}" \
        "${array_dependency[@]}" \
        --export="${exports}" \
        setup/autoresearch_gaze_shards.sbatch
)"
array_job_id="${array_job_id%%;*}"
if [[ ! "${array_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: sbatch returned an invalid shard job ID." >&2
    exit 1
fi

aggregate_exports="${exports},AUTORESEARCH_ARRAY_JOB_ID=${array_job_id}"
aggregate_job_id="$(
    sbatch --parsable \
        --dependency="afterok:${array_job_id}" \
        --export="${aggregate_exports}" \
        setup/autoresearch_gaze_aggregate.sbatch
)"
aggregate_job_id="${aggregate_job_id%%;*}"
if [[ ! "${aggregate_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: sbatch returned an invalid aggregate job ID." >&2
    exit 1
fi

printf 'AUTORESEARCH_SUBMISSION|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${mode}" "${array_job_id}" "${aggregate_job_id}" "${submitted_branch}" \
    "${candidate_commit}" "${candidate_sha256}" "${incumbent_commit}" \
    "${incumbent_sha256}" "${cache_path}" "${candidate_source}" "${incumbent_source}" \
    "${project_dir}"
