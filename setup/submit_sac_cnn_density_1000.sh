#!/usr/bin/env bash

set -euo pipefail

control_path="${QUEST_CONTROL_PATH:-/tmp/quest.sock}"
remote_host="${QUEST_HOST:-quest.northwestern.edu}"
expected_branch="velocity-action-env"

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
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: local working tree has uncommitted changes." >&2
    git status --short >&2
    exit 1
fi

git fetch --quiet origin "${expected_branch}"
local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "origin/${expected_branch}")"
if [[ "${local_head}" != "${remote_head}" ]]; then
    echo "Error: local HEAD must match origin/${expected_branch}." >&2
    exit 1
fi

result="$({
    ssh -S "${control_path}" "${remote_host}" bash -s <<'REMOTE'
set -euo pipefail

cd "${HOME}/projects/Mice-first-person"
git fetch --quiet origin velocity-action-env
git switch --quiet velocity-action-env
git pull --ff-only --quiet origin velocity-action-env
mkdir -p slurm_logs results/sac

git_commit="$(git rev-parse HEAD)"
array_job_id="$(
    sbatch --parsable \
        --export="ALL,QUEST_GIT_COMMIT=${git_commit}" \
        setup/sac_cnn_density_1000.sbatch
)"
aggregate_job_id="$(
    sbatch --parsable \
        --dependency="afterok:${array_job_id}" \
        --export="ALL,QUEST_GIT_COMMIT=${git_commit},SAC_DENSITY_ARRAY_JOB_ID=${array_job_id}" \
        setup/sac_cnn_density_aggregate.sbatch
)"
printf '%s|%s|%s\n' "${array_job_id}" "${aggregate_job_id}" "${git_commit}"
REMOTE
} 2>&1)" || {
    echo "Quest density submission failed:" >&2
    echo "${result}" >&2
    exit 1
}

IFS='|' read -r array_job_id aggregate_job_id git_commit <<<"${result}"
echo "Submitted SAC density array: ${array_job_id}"
echo "Submitted SAC density aggregate: ${aggregate_job_id} (afterok:${array_job_id})"
echo "Git commit: ${git_commit}"
