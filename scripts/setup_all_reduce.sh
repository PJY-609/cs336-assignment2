#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Copy mode also works on filesystems that cannot link from uv's cache.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/cs336-uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv sync --frozen
.venv/bin/python - <<'PY'
import torch
import torch.distributed as dist
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
print(f"Gloo available: {dist.is_gloo_available()}; NCCL available: {dist.is_nccl_available()}")
print(f"Visible GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"NCCL version: {torch.cuda.nccl.version()}")
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index))
PY
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    nvidia-smi topo -m
fi
