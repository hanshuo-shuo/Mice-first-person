#!/usr/bin/env bash

# Verified default Slurm account for Quest user shv7753 on 2026-08-19.
#SBATCH --account=p31777
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=mice-sac
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"

cd "${project_dir}"
mkdir -p .cache/matplotlib logs Saved_Models

export MPLBACKEND=Agg
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Project: ${project_dir}"
echo "Commit: ${QUEST_GIT_COMMIT:-unknown}"
echo "Started: $(date --iso-8601=seconds)"

conda run --no-capture-output --name "${env_name}" \
    python -B SAC_train.py --config configs/sac_peeking_0406.yaml

echo "Finished: $(date --iso-8601=seconds)"
