#!/usr/bin/env bash

set -euo pipefail

env_name="${QUEST_CONDA_ENV:-Mice-BotEvade}"
pytorch_version="${QUEST_PYTORCH_VERSION:-2.5.1}"
cuda_version="${QUEST_PYTORCH_CUDA_VERSION:-12.4}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available on PATH." >&2
    exit 1
fi

conda install --yes --name "${env_name}" \
    --strict-channel-priority \
    -c pytorch -c nvidia -c conda-forge \
    "pytorch::pytorch=${pytorch_version}" \
    "pytorch::pytorch-cuda=${cuda_version}"

conda run --no-capture-output --name "${env_name}" python -c '
import torch
assert torch.backends.cuda.is_built(), "Installed PyTorch is still a CPU build"
assert torch.version.cuda is not None, "PyTorch reports no compiled CUDA version"
print(f"CUDA PyTorch build ready: torch={torch.__version__} compiled_cuda={torch.version.cuda}")
'
