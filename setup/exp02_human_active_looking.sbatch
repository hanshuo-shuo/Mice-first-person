#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=exp02-human-look
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
data_root="${EXP02_DATA_ROOT:-datasets/human_demos}"
output_dir="${EXP02_OUTPUT_DIR:-results/human_active_looking_fingerprint/exp02_${SLURM_JOB_ID}}"
git_commit="${QUEST_GIT_COMMIT:-}"

if [[ ! "${git_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: QUEST_GIT_COMMIT must contain the submitted 40-character SHA." >&2
    exit 1
fi

cd "${project_dir}"
mkdir -p .cache/matplotlib results/human_active_looking_fingerprint slurm_logs

export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy

echo "EXP-02 git_commit=${git_commit}"
echo "EXP-02 data_root=${data_root}"
echo "EXP-02 output_dir=${output_dir}"

conda run --no-capture-output --name "${env_name}" \
    python -B -m analysis.human_active_looking_fingerprint \
    --data-root "${data_root}" \
    --output-dir "${output_dir}"

echo "EXP-02 summary=${project_dir}/${output_dir}/exp02_summary.json"
