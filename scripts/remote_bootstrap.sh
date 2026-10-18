#!/usr/bin/env bash
# Full-stack setup; facts-only never installs. torch-only is unsupported because
# the kernels import vllm.triton_utils. CI_DATA_ROOT moves large directories.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${CI_RUNTIME_DIR:-/opt/cloud-inference-framework}"
export CI_RUNTIME_DIR="$runtime_dir"
data_root="${CI_DATA_ROOT:-}"
vllm_rev=db9527a46873454610df6dbedf79a36d6bf1a7f6
mode=full
for arg in "$@"; do
    case "$arg" in
        --facts-only) mode=facts;;
        --torch-only) echo 'Unsupported: torch-only still needs vLLM imports. Use a compatible full-stack host.' >&2; exit 2;;
        -h|--help) echo 'Usage: remote_bootstrap.sh [--facts-only]'; exit 0;;
        *) printf 'unknown argument: %s\n' "$arg" >&2; exit 2;;
    esac
done
mkdir -p "$runtime_dir/results"
out="$(mktemp -d "$runtime_dir/results/bootstrap-remote-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
exec > >(tee "$out/bootstrap.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$out/exitcode"' EXIT
die() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }
{
    date -u
    uname -a
    nvidia-smi || true
    python3 --version || true
    df -h "$runtime_dir" "${data_root:-/}" || true
    free -g || true
} > "$out/facts.txt" 2>&1
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
printf 'runtime_dir=%s\ndriver=%s\nmode=%s\n' "$runtime_dir" "$driver" "$mode" > "$out/report.env"
cat "$out/facts.txt"
[[ "$mode" == facts ]] && exit 0
# Conservative gate for our CUDA-13 stack; actual CUDA/JIT must also pass.
driver_major="${driver%%.*}"
[[ "$driver_major" =~ ^[0-9]+$ ]] || die 'No NVIDIA driver detected'
(( driver_major >= 580 )) || die 'Pinned CUDA-13 stack requires a compatible host (project gate: driver >=580); no torch-only fallback.'
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Select a Python 3.12 host/interpreter for the measured stack"'
mkdir -p "$runtime_dir/scripts" "$runtime_dir/kernels"
for sub in .venv models vllm-src; do
    target="$runtime_dir/$sub"
    if [[ -n "$data_root" && ! -e "$target" && ! -L "$target" ]]; then
        mkdir -p "$data_root/ci-fw/$sub"
        ln -s "$data_root/ci-fw/$sub" "$target"
    fi
done
[[ -x "$runtime_dir/.venv/bin/python" ]] || python3 -m venv "$runtime_dir/.venv"
py="$runtime_dir/.venv/bin/python"
"$py" -m pip install 'vllm==0.31.0' 'torch==2.13.0' 'triton==3.7.1' 'flashinfer-python==0.7.0.post1'
"$py" -m pip install -r "$project_dir/configs/compiler-constraints.txt" 'ninja==1.13.2'
"$py" -m pip check
"$py" -m pip freeze > "$out/requirements-resolved.txt"
"$py" - <<'PY' > "$out/versions.txt"
import torch, triton, vllm, flashinfer
print(dict(torch=torch.__version__, triton=triton.__version__, vllm=vllm.__version__,
           flashinfer=flashinfer.__version__, cuda=torch.version.cuda))
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.version.cuda == '13.0', 'Unexpected CUDA build; do not silently compare different stacks'
x = torch.arange(16, device='cuda', dtype=torch.float32)
assert x.sum().item() == 120
torch.cuda.synchronize()
print(dict(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0)))
PY
cat "$out/versions.txt"
CI_RUNTIME_DIR="$runtime_dir" bash "$project_dir/scripts/setup_cuda.sh" > "$out/compiler.txt"
src="$runtime_dir/vllm-src"
if [[ -e "$src/vllm/v1/worker/gpu/sample/min_p.py" ]]; then
    [[ "$(git -C "$src" rev-parse HEAD)" == "$vllm_rev" ]] || die 'Existing source revision differs; preserve it and choose another runtime.'
    [[ -z "$(git -C "$src" status --porcelain)" ]] || die 'Existing vllm-src has changes; preserve them.'
else
    mkdir -p "$src"
    if [[ ! -d "$src/.git" ]]; then
        [[ -z "$(ls -A "$src")" ]] || die 'Nonempty source directory is not a Git checkout'
        git init -q "$src"
        git -C "$src" remote add origin https://github.com/vllm-project/vllm
    fi
    [[ -z "$(git -C "$src" status --porcelain)" ]] || die 'Source directory has changes'
    git -C "$src" fetch --depth 1 origin "$vllm_rev"
    git -C "$src" checkout --detach FETCH_HEAD
fi
git -C "$src" rev-parse HEAD > "$out/vllm-source.txt"
# Verify existing files too, against checked-in pinned HF blob metadata.
"$py" "$project_dir/scripts/prepare_model.py" \
    --metadata "$project_dir/configs/model-metadata.json" \
    --dest "$runtime_dir/models/Qwen3-0.6B" \
    --endpoint "${HF_ENDPOINT:-https://huggingface.co}" \
    > "$out/model-verification.json"
CI_RUNTIME_DIR="$runtime_dir" bash "$project_dir/scripts/preflight.sh"
printf 'Ready. Preview first:\n  CI_RUNTIME_DIR=%q bash %q/scripts/repro_l4.sh --dry-run\n' "$runtime_dir" "$project_dir"
printf 'Evidence: %s\n' "$out"
