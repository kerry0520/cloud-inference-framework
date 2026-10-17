#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir=/opt/cloud-inference-framework
export CUDA_HOME="$runtime_dir/cuda-toolkit"
export PATH="$runtime_dir/.venv/bin:$CUDA_HOME/bin:$PATH"
mode="${1:-baseline}"
case "$mode" in baseline|profile|baseline-pinned|profile-pinned|churn-pinned) ;; *) echo 'Invalid mode' >&2; exit 2;; esac
run_name="${mode}-$(date -u +%Y%m%dT%H%M%SZ)"
run_out="$runtime_dir/results/$run_name"
mkdir -p "$runtime_dir/scripts" "$project_dir/results"
cp "$project_dir/scripts/"*.py "$runtime_dir/scripts/"
options=()
if [[ "$mode" == *pinned ]]; then export VLLM_WSL2_ENABLE_PIN_MEMORY=1; fi
if [[ "$mode" == profile* ]]; then
    options+=(--profile --batches 2 --output-tokens 32 --repeats 1)
fi
if [[ "$mode" == churn* ]]; then
    options+=(--profile --churn --batches 8 --output-tokens 64 --repeats 1)
fi
set +e
TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=2 \
    "$runtime_dir/.venv/bin/python" "$runtime_dir/scripts/baseline.py" \
    --out "$run_out" "${options[@]}" > "$run_out.log" 2>&1
run_code=$?
set -e
printf '%s\n' "$run_code" > "$run_out.exitcode"
if [[ -d "$run_out" ]]; then cp -r "$run_out" "$project_dir/results/"; fi
cp "$run_out.log" "$run_out.exitcode" "$project_dir/results/"
tail -n 15 "$run_out.log"
printf 'RESULT=%s EXIT=%s\n' "$run_out" "$run_code"
exit "$run_code"
