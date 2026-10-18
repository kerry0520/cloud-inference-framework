#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${CI_RUNTIME_DIR:-/opt/cloud-inference-framework}"
export CUDA_HOME="$runtime_dir/cuda-toolkit"
export PATH="$runtime_dir/.venv/bin:$CUDA_HOME/bin:$PATH"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export MAX_JOBS=2
run_name="preflight-$(date -u +%Y%m%dT%H%M%SZ)"
out="$runtime_dir/results/$run_name"
mkdir "$out"
cp "$project_dir/scripts/"*.py "$runtime_dir/scripts/"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version > "$out/compiler.txt"
else
    # A rented host ships the driver and the CUDA runtime inside the wheels,
    # but often no nvcc.  Record the absence instead of failing the preflight.
    {
        printf 'nvcc not found on this host\n'
        cat /proc/driver/nvidia/version 2>/dev/null || nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true
    } > "$out/compiler.txt"
fi
run_code=0
for check in gpu pin_memory sampler; do
    set +e
    "$runtime_dir/.venv/bin/python" "$runtime_dir/scripts/check_$check.py" \
        > "$out/$check.log" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" > "$out/$check.exitcode"
    tail -n 8 "$out/$check.log"
    if [[ "$code" != 0 ]]; then run_code=$code; break; fi
done
cp -r "$out" "$project_dir/results/"
exit "$run_code"
