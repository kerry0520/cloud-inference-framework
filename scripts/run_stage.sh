#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir=/opt/cloud-inference-framework
export CUDA_HOME="$runtime_dir/cuda-toolkit"
export PATH="$runtime_dir/.venv/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$runtime_dir:$runtime_dir/scripts${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export MAX_JOBS=2
export TORCHINDUCTOR_COMPILE_THREADS=1
stage="${1:?stage required}"
extra=()
case "$stage" in shapes) script=probe_shapes.py;; minp-micro) script=minp_micro.py;; minp-engine) script=minp_engine.py;; minp-repeat) script=minp_engine.py; extra+=(--timing-only);; *) exit 2;; esac
name="$stage-$(date -u +%Y%m%dT%H%M%SZ)"
out="$runtime_dir/results/$name"
cp "$project_dir/scripts/"*.py "$runtime_dir/scripts/"
mkdir -p "$runtime_dir/kernels"
cp "$project_dir/kernels/"*.py "$runtime_dir/kernels/"
set +e
"$runtime_dir/.venv/bin/python" "$runtime_dir/scripts/$script" --out "$out" "${extra[@]}" \
    > "$out.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$out.exitcode"
if [[ -d "$out" ]]; then cp -r "$out" "$project_dir/results/"; fi
cp "$out.log" "$out.exitcode" "$project_dir/results/"
tail -n 20 "$out.log"
printf 'RESULT=%s EXIT=%s\n' "$out" "$code"
exit "$code"
