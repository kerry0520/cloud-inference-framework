#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${CI_RUNTIME_DIR:-/opt/cloud-inference-framework}"
export CI_RUNTIME_DIR="$runtime_dir"
if [[ -x "$runtime_dir/cuda-toolkit/bin/nvcc" ]]; then
    export CUDA_HOME="$runtime_dir/cuda-toolkit"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
export PATH="$runtime_dir/.venv/bin:$PATH"
export PYTHONPATH="$runtime_dir:$runtime_dir/scripts${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export MAX_JOBS=2
export TORCHINDUCTOR_COMPILE_THREADS=1
stage="${1:?stage required}"
extra=()
case "$stage" in
    shapes) script=probe_shapes.py;;
    minp-micro) script=minp_micro.py;;
    minp-engine) script=minp_engine.py;;
    minp-repeat) script=minp_engine.py; extra+=(--timing-only);;
    attribute) script=attribute_decode.py; extra+=("$runtime_dir/results/profile-pinned-20261017T020913Z");;
    tail) script=inspect_tail_args.py; extra+=("$runtime_dir/results/profile-pinned-20261017T020913Z");;
    bw) script=measure_bandwidth.py;;
    patch) script=verify_minp_patch.py; extra+=("--patch" "$project_dir/patches/0001-mrv2-min-p-two-stage.patch" "--patched" "$project_dir/patches/mrv2_min_p.py");;
    minp-patch-engine) script=minp_engine.py; extra+=(--patch-module "$project_dir/patches/mrv2_min_p.py" --patch "$project_dir/patches/0001-mrv2-min-p-two-stage.patch" --check-only);;
    abba) script=abba_patch.py; extra+=(--patch-module "$project_dir/patches/mrv2_min_p.py");;
    abba-scale) script=abba_patch.py; extra+=(--patch-module "$project_dir/patches/mrv2_min_p.py" --batches 1,2,4,8);;
    abba-knob) script=abba_patch.py; extra+=(--patch-module "$project_dir/patches/mrv2_min_p_knob.py" --batches 1,2,4,8);;
    roofline) script=roofline_model.py; extra+=(--attribution "$runtime_dir/results/attribute-20261017T093052Z/attribution.json");;
    figures) script=make_figures.py; extra+=(--attribution "$runtime_dir/results/attribute-20261017T093052Z/attribution.json" --results "$runtime_dir/results" --micro "$runtime_dir/results/minp-micro-20261017T070433Z/timing.jsonl");;
    minp-sweep) script=minp_param_sweep.py; extra+=(--patched "$project_dir/patches/mrv2_min_p.py");;
    minp-compare) script=minp_variant_compare.py; extra+=(--upstream "$runtime_dir/vllm-src/vllm/v1/worker/gpu/sample/min_p.py" --knob "$project_dir/patches/mrv2_min_p_knob.py" --pr-diff "$project_dir/results/upstream-pr-58907.diff");;
    quant-probe) script=quant_probe.py;;
    quant-decode) script=quant_decode.py;;
    quant-context) script=quant_decode.py; extra+=(--arms fp16 --prompt-lens 128,2048,8192 --batches 1,2 --rounds 2 --gpu-fraction 0.62);;
    quant-profile) script=quant_profile.py;;
    quant-family) script=attribute_decode.py; extra+=("${2:?trace root required}" --by-family);;
    minp-compare-f32) script=minp_variant_compare.py; extra+=(--upstream "$runtime_dir/vllm-src/vllm/v1/worker/gpu/sample/min_p.py" --knob "$project_dir/patches/mrv2_min_p_knob.py" --pr-diff "$project_dir/results/upstream-pr-58907.diff" --dtype float32);;
    *) exit 2;;
esac
if [[ "$stage" == quant-family ]]; then
    extra+=("${@:3}")
else
    extra+=("${@:2}")
fi
mkdir -p "$runtime_dir/results" "$runtime_dir/scripts" "$project_dir/results"
out="$(mktemp -d "$runtime_dir/results/$stage-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
# Python runners create their output with exist_ok=False. Release only this
# newly reserved empty directory, retaining the random suffix in the name.
rmdir "$out"
if [[ "$(realpath "$project_dir/scripts")" != "$(realpath "$runtime_dir/scripts")" ]]; then
    cp "$project_dir/scripts/"*.py "$runtime_dir/scripts/"
fi
mkdir -p "$runtime_dir/kernels"
if [[ "$(realpath "$project_dir/kernels")" != "$(realpath "$runtime_dir/kernels")" ]]; then
    cp "$project_dir/kernels/"*.py "$runtime_dir/kernels/"
fi
set +e
"$runtime_dir/.venv/bin/python" "$runtime_dir/scripts/$script" --out "$out" "${extra[@]}" \
    > "$out.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$out.exitcode"
if [[ "$(realpath "$runtime_dir")" != "$(realpath "$project_dir")" ]]; then
    if [[ -d "$out" ]]; then cp -r "$out" "$project_dir/results/"; fi
    cp "$out.log" "$out.exitcode" "$project_dir/results/"
fi
tail -n 20 "$out.log"
printf 'RESULT=%s EXIT=%s\n' "$out" "$code"
exit "$code"
