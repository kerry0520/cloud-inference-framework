#!/usr/bin/env bash
# Local validation of changed entry points using the existing measured venv.
# Does not install packages, rent GPUs, or claim remote bootstrap has passed.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
existing="${CI_RUNTIME_DIR:-/opt/cloud-inference-framework}"
out="$(mktemp -d "$project_dir/results/remote-readiness-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
fixture="$(mktemp -d /tmp/ci-remote-smoke-XXXXXX)"
exec > >(tee "$out/check.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$out/exitcode"; printf "RESULT=%s EXIT=%s\n" "$out" "$code"' EXIT
printf 'fixture=%s\nexisting=%s\n' "$fixture" "$existing" > "$out/paths.txt"
py="$existing/.venv/bin/python"
for file in remote_bootstrap preflight run_stage repro_l4 setup_cuda check_remote_local; do
    bash -n "$project_dir/scripts/$file.sh"
done
"$py" -m compileall -q "$project_dir/scripts" "$project_dir/kernels"
"$py" "$project_dir/scripts/check_remote_contract.py" > "$out/contracts.json"
"$py" "$project_dir/scripts/prepare_model.py" --verify-only \
    --dest "$existing/models/Qwen3-0.6B" --metadata "$project_dir/configs/model-metadata.json" \
    > "$out/model-verification.json"
ln -s "$existing/.venv" "$fixture/.venv"
ln -s "$existing/models" "$fixture/models"
CI_RUNTIME_DIR="$fixture" bash "$project_dir/scripts/setup_cuda.sh" > "$out/compiler.txt"
CI_RUNTIME_DIR="$fixture" bash "$project_dir/scripts/preflight.sh"
# Minimal real request with the custom runtime, exercising metric validation and
# explicit EngineCore shutdown. This is a smoke test, not a performance result.
export CI_RUNTIME_DIR="$fixture"
export CUDA_HOME="$fixture/cuda-toolkit"
export PATH="$existing/.venv/bin:$CUDA_HOME/bin:$PATH"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export MAX_JOBS=2
export TORCHINDUCTOR_COMPILE_THREADS=1
"$py" "$project_dir/scripts/quant_decode.py" --out "$out/engine-smoke" \
    --arms fp16 --rounds 1 --batches 1 --prompt-len 32 --max-tokens 4 \
    > "$out/engine-smoke.log" 2>&1
cp -r "$fixture/results" "$out/fixture-results"
"$py" - "$out" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
assert json.loads((out / 'engine-smoke/status.json').read_text())['state'] == 'passed'
files = []
for path in sorted(out.rglob('*')):
    if path.is_file() and path.name not in ('check.log', 'manifest.json', 'exitcode'):
        files.append(dict(path=str(path.relative_to(out)), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(out / 'manifest.json').write_text(json.dumps(dict(scope='local readiness only; no remote installation', files=files), indent=2))
PY
printf 'Local readiness checks passed. Temporary fixture retained at %s\n' "$fixture"
