#!/usr/bin/env bash
# Milestone-2 gate entrypoint. Run on a single CUDA GPU (pre-registered: RTX 4090).
# Exit 0 == gate passed: full smoke matrix ran, prefix-cache toggle evidenced,
# content-addressed manifest verified.
set -euo pipefail
cd "$(dirname "$0")"

# Default pin: newest vLLM whose PyPI wheels are CUDA-12.8-toolkit builds.
# vLLM >= 0.20 pins torch 2.11 and ships CUDA-13-built kernels
# (libcudart.so.13), which cannot run on r570-driver hosts (CUDA 12.8) — the
# common Runpod 4090 fleet. On a CUDA >= 13.0 host, override VLLM_VERSION.
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
MATRIX="${MATRIX:-configs/smoke.json}"
OUT_DIR="${OUT_DIR:-results/$(date -u +%Y%m%dT%H%M%SZ)-smoke}"

echo "[repro] gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
if ! python3 -c "
import vllm, torch
assert vllm.__version__ == '${VLLM_VERSION}', f'have {vllm.__version__}, want ${VLLM_VERSION}'
assert torch.cuda.is_available(), 'torch cannot reach the GPU (driver/toolkit mismatch?)'
" 2>/dev/null; then
  echo "[repro] installing vllm==${VLLM_VERSION}"
  pip install --no-cache-dir "vllm==${VLLM_VERSION}"
fi

python3 -m kvdrift.main run --matrix "$MATRIX" --out "$OUT_DIR"
python3 -m kvdrift.main verify --out "$OUT_DIR"

echo "GATE M2: PASS ($OUT_DIR)"
