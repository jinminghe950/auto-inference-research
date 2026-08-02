#!/usr/bin/env bash
# Milestone-3 sweep entrypoint. Requires a host whose driver supports the
# vLLM pin's CUDA build (CUDA >= 13.0 for vllm 0.26.0).
#
# vLLM 0.26 (torch 2.11) and LMDeploy 0.15 (torch <= 2.10) cannot share a
# python env, so each engine lives in its own venv; the harness driver is
# stdlib-only and runs on the system python.
set -euo pipefail
cd "$(dirname "$0")"

VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
LMDEPLOY_VERSION="${LMDEPLOY_VERSION:-0.15.0}"
VENVS="${VENVS:-/workspace/venvs}"
MATRIX="${MATRIX:-configs/sweep.json}"
OUT_DIR="${OUT_DIR:-results/$(date -u +%Y%m%dT%H%M%SZ)-sweep}"

echo "[sweep] gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo unknown)"

ensure_venv() { # name, pip-spec, import-module, pinned-version
  local venv="$VENVS/$1"
  if ! "$venv/bin/python" -c "import $3; assert $3.__version__ == '$4'" 2>/dev/null; then
    echo "[sweep] building venv $1 ($2)"
    python3 -m venv "$venv"
    "$venv/bin/pip" install --no-cache-dir -q --upgrade pip
    "$venv/bin/pip" install --no-cache-dir "$2"
  fi
}

ensure_venv vllm "vllm==${VLLM_VERSION}" vllm "${VLLM_VERSION}"
ensure_venv lmdeploy "lmdeploy==${LMDEPLOY_VERSION}" lmdeploy "${LMDEPLOY_VERSION}"

export VLLM_BIN="$VENVS/vllm/bin/vllm"
export LMDEPLOY_BIN="$VENVS/lmdeploy/bin/lmdeploy"

python3 -m kvdrift.main run --matrix "$MATRIX" --out "$OUT_DIR"
python3 -m kvdrift.main verify --out "$OUT_DIR"

echo "GATE M3: PASS ($OUT_DIR)"
