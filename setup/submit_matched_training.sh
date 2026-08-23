#!/usr/bin/env bash

set -euo pipefail

control_path="${QUEST_CONTROL_PATH:-/tmp/quest.sock}"
remote_host="${QUEST_HOST:-quest.northwestern.edu}"
expected_branch="velocity-action-env"
acceptance_job_id="${MATCHED_ACCEPTANCE_JOB_ID:-}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_dir}"

if [[ ! -S "${control_path}" ]] || ! ssh -O check -S "${control_path}" "${remote_host}" >/dev/null 2>&1; then
    echo "Error: active Quest connection not found at ${control_path}." >&2
    exit 1
fi
if [[ "$(git branch --show-current)" != "${expected_branch}" ]]; then
    echo "Error: expected branch ${expected_branch}." >&2
    exit 1
fi
if [[ -n "${acceptance_job_id}" && ! "${acceptance_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: MATCHED_ACCEPTANCE_JOB_ID must be a numeric Slurm job ID." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: local working tree has uncommitted changes." >&2
    git status --short >&2
    exit 1
fi
git fetch --quiet origin "${expected_branch}"
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/${expected_branch}")" ]]; then
    echo "Error: local HEAD must match origin/${expected_branch}." >&2
    exit 1
fi

result="$({
    ssh -S "${control_path}" "${remote_host}" bash -s -- "${acceptance_job_id}" <<'REMOTE'
set -euo pipefail
acceptance_job_id="$1"
cd "${HOME}/projects/Mice-first-person"
git fetch --quiet origin velocity-action-env
git switch --quiet velocity-action-env
git pull --ff-only --quiet origin velocity-action-env
mkdir -p slurm_logs results/sac_matched

git_commit="$(git rev-parse HEAD)"
dependency_args=()
if [[ -n "${acceptance_job_id}" ]]; then
    dependency_args+=("--dependency=afterok:${acceptance_job_id}")
fi
train_id="$(
    sbatch --parsable \
        "${dependency_args[@]}" \
        --export="ALL,QUEST_GIT_COMMIT=${git_commit}" \
        setup/matched_sac_train.sbatch
)"
test_id="$(
    sbatch --parsable \
        --dependency="afterok:${train_id}" \
        --export="ALL,QUEST_GIT_COMMIT=${git_commit},MATCHED_TRAIN_ARRAY_JOB_ID=${train_id}" \
        setup/matched_sac_test.sbatch
)"
aggregate_id="$(
    sbatch --parsable \
        --dependency="afterok:${test_id}" \
        --export="ALL,MATCHED_TRAIN_ARRAY_JOB_ID=${train_id},MATCHED_TEST_ARRAY_JOB_ID=${test_id}" \
        setup/matched_sac_test_aggregate.sbatch
)"
printf '%s|%s|%s|%s\n' "${train_id}" "${test_id}" "${aggregate_id}" "${git_commit}"
REMOTE
} 2>&1)" || {
    echo "Matched training submission failed:" >&2
    echo "${result}" >&2
    exit 1
}

IFS='|' read -r train_id test_id aggregate_id git_commit <<<"${result}"
echo "Submitted matched training array: ${train_id}"
if [[ -n "${acceptance_job_id}" ]]; then
    echo "Matched training dependency: afterok:${acceptance_job_id}"
fi
echo "Submitted matched test array: ${test_id} (afterok:${train_id})"
echo "Submitted matched aggregate: ${aggregate_id} (afterok:${test_id})"
echo "Git commit: ${git_commit}"
