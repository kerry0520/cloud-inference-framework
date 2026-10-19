#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${CI_RUNTIME_DIR:-/opt/cloud-inference-framework}"
export CUDA_HOME="$runtime/cuda-toolkit"
export PATH="$runtime/.venv/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
out="$(mktemp -d "$project/results/norm-target-gpu-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
rmdir "$out"
set +e
"$runtime/.venv/bin/python" "$project/scripts/probe_norm_target_gpu.py" --out "$out" > "$out.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$out.exitcode"
tail -n 15 "$out.log"
printf 'RESULT=%s EXIT=%s\n' "$out" "$code"
exit "$code"
