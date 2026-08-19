#!/usr/bin/env bash

set -euo pipefail

control_path="${QUEST_CONTROL_PATH:-/tmp/quest.sock}"
remote_host="${QUEST_HOST:-quest.northwestern.edu}"
expected_branch="velocity-action-env"
job_file="${1:-setup/sac_train.sbatch}"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash setup/submit_quest.sh [setup/job_file.sbatch]" >&2
    exit 2
fi

case "${job_file}" in
    /*|*..*|*[!A-Za-z0-9_./-]*)
        echo "Error: job file must be a safe repository-relative path." >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_dir}"

if [[ ! -S "${control_path}" ]]; then
    echo "Error: Quest connection ${control_path} does not exist." >&2
    echo "Run:" >&2
    echo "  ssh -M -S ${control_path} -o ControlPersist=8h -fN ${remote_host}" >&2
    exit 1
fi

if ! ssh -O check -S "${control_path}" "${remote_host}" >/dev/null 2>&1; then
    echo "Error: ${control_path} is not an active Quest connection." >&2
    echo "Close the stale connection with ssh -O exit, then create it again." >&2
    exit 1
fi

current_branch="$(git branch --show-current)"
if [[ "${current_branch}" != "${expected_branch}" ]]; then
    echo "Error: current branch is ${current_branch}; expected ${expected_branch}." >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: the local working tree has uncommitted changes." >&2
    echo "Commit and push the intended experiment before submitting:" >&2
    git status --short >&2
    exit 1
fi

git fetch --quiet origin "${expected_branch}"
local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "origin/${expected_branch}")"

if [[ "${local_head}" != "${remote_head}" ]]; then
    read -r behind ahead < <(
        git rev-list --left-right --count \
            "origin/${expected_branch}...${expected_branch}"
    )
    echo "Error: local and GitHub commits differ (behind=${behind}, ahead=${ahead})." >&2
    echo "Run git pull --ff-only or git push as appropriate, then submit again." >&2
    exit 1
fi

job_id="$({
    ssh -S "${control_path}" "${remote_host}" \
        bash -s -- "${job_file}" <<'REMOTE'
set -euo pipefail

job_file="$1"
cd "${HOME}/projects/Mice-first-person"
git fetch --quiet origin velocity-action-env
git switch --quiet velocity-action-env
git pull --ff-only --quiet origin velocity-action-env
mkdir -p logs slurm_logs Saved_Models

if [[ ! -f "${job_file}" ]]; then
    echo "Error: remote job file does not exist: ${job_file}" >&2
    exit 1
fi

quest_git_commit="$(git rev-parse HEAD)"
sbatch --parsable \
    --export="ALL,QUEST_GIT_COMMIT=${quest_git_commit}" \
    "${job_file}"
REMOTE
} 2>&1)" || {
    echo "Quest submission failed:" >&2
    echo "${job_id}" >&2
    exit 1
}

echo "Submitted Quest job: ${job_id}"
echo "Monitor: ssh -S ${control_path} ${remote_host} 'squeue -j ${job_id}'"
echo "Logs: ~/projects/Mice-first-person/slurm_logs/"
