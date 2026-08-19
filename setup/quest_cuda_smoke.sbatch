#!/usr/bin/env bash

#SBATCH --account=p31777
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=mice-cuda-smoke
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

project_dir="${QUEST_PROJECT_DIR:-${HOME}/projects/Mice-first-person}"
env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"

cd "${project_dir}"
export PYTHONDONTWRITEBYTECODE=1
export PYGAME_HIDE_SUPPORT_PROMPT=1

conda run --no-capture-output --name "${env_name}" python -c '
import torch
assert torch.cuda.is_available(), "CUDA device is not available to PyTorch"
device = torch.device("cuda:0")
left = torch.randn((1024, 1024), device=device)
right = torch.randn((1024, 1024), device=device)
value = (left @ right).mean()
torch.cuda.synchronize()
print(
    f"Quest CUDA smoke passed: torch={torch.__version__} "
    f"compiled_cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)} "
    f"mean={float(value):.6f}"
)
'
