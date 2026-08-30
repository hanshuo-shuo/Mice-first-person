#!/usr/bin/env bash

# Submit the registered development seed set only.  Confirmation is
# intentionally absent from this interface and must remain runner-authorized.

set -euo pipefail

usage() {
    echo "Usage:" >&2
    echo "  $0 baseline --run-tag TAG --candidate-commit COMMIT --candidate-sha256 SHA256 --incumbent-commit COMMIT --incumbent-sha256 SHA256 [--shards N]" >&2
    echo "  $0 experiment --run-tag TAG --evaluation-tag TAG --candidate-commit COMMIT --candidate-sha256 SHA256 --incumbent-commit COMMIT --incumbent-sha256 SHA256 [--shards N]" >&2
}

if (( $# < 1 )); then
    usage
    exit 2
fi

mode="$1"
shift
if [[ "${mode}" != "baseline" && "${mode}" != "experiment" ]]; then
    usage
    exit 2
fi

run_tag=""
evaluation_tag=""
candidate_commit=""
candidate_sha256=""
incumbent_commit=""
incumbent_sha256=""
shard_count=16

while (( $# > 0 )); do
    case "$1" in
        --run-tag)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            run_tag="$2"
            shift 2
            ;;
        --evaluation-tag)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            evaluation_tag="$2"
            shift 2
            ;;
        --candidate-commit)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            candidate_commit="$2"
            shift 2
            ;;
        --candidate-sha256)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            candidate_sha256="$2"
            shift 2
            ;;
        --incumbent-commit)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            incumbent_commit="$2"
            shift 2
            ;;
        --incumbent-sha256)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            incumbent_sha256="$2"
            shift 2
            ;;
        --shards)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            shard_count="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown argument $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ! "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]]; then
    echo "Error: --run-tag must be a short filesystem-safe tag." >&2
    exit 2
fi
if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]] || (( shard_count > 128 )); then
    echo "Error: --shards must lie in 1..128." >&2
    exit 2
fi
if [[ ! "${candidate_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Error: --candidate-commit must be a lowercase 40-hex commit." >&2
    exit 2
fi
if [[ ! "${incumbent_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Error: --incumbent-commit must be a lowercase 40-hex commit." >&2
    exit 2
fi
if [[ ! "${candidate_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: --candidate-sha256 must be a lowercase SHA-256." >&2
    exit 2
fi
if [[ ! "${incumbent_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Error: --incumbent-sha256 must be a lowercase SHA-256." >&2
    exit 2
fi
if [[ "${mode}" == "baseline" ]]; then
    if [[ -n "${evaluation_tag}" ]]; then
        echo "Error: baseline does not accept --evaluation-tag." >&2
        exit 2
    fi
    if [[ "${candidate_commit}" != "${incumbent_commit}" || "${candidate_sha256}" != "${incumbent_sha256}" ]]; then
        echo "Error: baseline candidate/incumbent identities must be the same setup commit." >&2
        exit 2
    fi
    evaluation_tag="baseline"
else
    if [[ ! "${evaluation_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]]; then
        echo "Error: experiment requires a filesystem-safe --evaluation-tag." >&2
        exit 2
    fi
fi

control_path="${QUEST_CONTROL_PATH:-/tmp/quest.sock}"
remote_host="${QUEST_HOST:-quest.northwestern.edu}"
remote_project="${QUEST_PROJECT_DIR:-projects/Mice-first-person}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
remote_script="${script_dir}/remote_submit_autoresearch_gaze.sh"
cd "${project_dir}"

if [[ ! -S "${control_path}" ]] || ! ssh -O check -S "${control_path}" "${remote_host}" >/dev/null 2>&1; then
    echo "Error: active Quest connection not found at ${control_path}." >&2
    exit 1
fi
current_branch="$(git branch --show-current)"
if [[ -z "${current_branch}" ]] || ! git check-ref-format --branch "${current_branch}" >/dev/null 2>&1; then
    echo "Error: autoresearch submission requires a valid named branch." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: local working tree has uncommitted changes." >&2
    git status --short >&2
    exit 1
fi

local_head="$(git rev-parse HEAD)"
if [[ "${local_head}" != "${candidate_commit}" ]]; then
    echo "Error: local HEAD must equal --candidate-commit." >&2
    exit 1
fi
if [[ ! -f autoresearch/candidate.py ]]; then
    echo "Error: fixed candidate path autoresearch/candidate.py is missing." >&2
    exit 1
fi
local_candidate_digest_line="$(shasum -a 256 autoresearch/candidate.py)"
if [[ "${local_candidate_digest_line%% *}" != "${candidate_sha256}" ]]; then
    echo "Error: working candidate bytes do not match --candidate-sha256." >&2
    exit 1
fi
if ! git cat-file -e "${incumbent_commit}^{commit}" 2>/dev/null; then
    echo "Error: --incumbent-commit is not available in the local repository." >&2
    exit 1
fi
if ! local_incumbent_digest_line="$(git show "${incumbent_commit}:autoresearch/candidate.py" | shasum -a 256)"; then
    echo "Error: fixed candidate path is absent from --incumbent-commit." >&2
    exit 1
fi
if [[ "${local_incumbent_digest_line%% *}" != "${incumbent_sha256}" ]]; then
    echo "Error: incumbent commit bytes do not match --incumbent-sha256." >&2
    exit 1
fi

git fetch --quiet origin "${current_branch}"
remote_head="$(git rev-parse "origin/${current_branch}")"
if [[ "${local_head}" != "${remote_head}" ]]; then
    echo "Error: local HEAD must be pushed and equal origin/${current_branch}." >&2
    exit 1
fi

response="$(
    ssh -S "${control_path}" "${remote_host}" bash -s -- \
        "${mode}" \
        "${run_tag}" \
        "${evaluation_tag}" \
        "${current_branch}" \
        "${candidate_commit}" \
        "${candidate_sha256}" \
        "${incumbent_commit}" \
        "${incumbent_sha256}" \
        "${shard_count}" \
        "${remote_project}" < "${remote_script}" 2>&1
)" || {
    echo "Quest autoresearch submission failed:" >&2
    echo "${response}" >&2
    exit 1
}

result_line="${response##*$'\n'}"
IFS='|' read -r marker submitted_mode array_job_id aggregate_job_id git_branch candidate_commit candidate_sha256 incumbent_commit incumbent_sha256 cache_path candidate_snapshot incumbent_snapshot quest_worktree <<<"${result_line}"
if [[ "${marker}" != "AUTORESEARCH_SUBMISSION" || "${submitted_mode}" != "${mode}" ]]; then
    echo "Error: could not parse Quest submission response." >&2
    echo "${response}" >&2
    exit 1
fi

echo "Submitted autoresearch ${mode} shards: ${array_job_id}"
echo "Submitted autoresearch ${mode} aggregate: ${aggregate_job_id} (afterok:${array_job_id})"
echo "Git branch: ${git_branch}"
echo "Quest worktree: ${quest_worktree}"
echo "Candidate commit: ${candidate_commit}"
echo "Candidate SHA-256: ${candidate_sha256}"
echo "Candidate snapshot: ${candidate_snapshot}"
echo "Incumbent commit: ${incumbent_commit}"
echo "Incumbent SHA-256: ${incumbent_sha256}"
echo "Incumbent snapshot: ${incumbent_snapshot}"
echo "Comparator cache: ${cache_path}"
