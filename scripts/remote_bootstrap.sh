#!/usr/bin/env bash
# Prepare a freshly rented Linux GPU host so that scripts/repro_l4.sh can run.
#
# Why a script instead of a README: the rented box is ephemeral, the layout is
# unusual (small system disk, big data disk, conda python, no nvcc), and every
# number we later publish has to point at a run directory.  This script makes
# the host reproducible and writes down what it found, including failures.
#
# It is idempotent: rerun it after a reboot, after an interrupted pip, or with
# --facts-only to just collect the environment facts.
#
# Usage (on the rented host, from a checkout of this repo):
#   bash scripts/remote_bootstrap.sh                 # full: venv + vLLM + model + vllm-src
#   bash scripts/remote_bootstrap.sh --torch-only    # skip vLLM (older driver)
#   bash scripts/remote_bootstrap.sh --facts-only    # collect facts, install nothing
#   CI_DATA_ROOT=/root/autodl-tmp bash scripts/remote_bootstrap.sh
#
# Nothing secret is written: the repository is public, the model is public, and
# no token is needed.  The script never edits ~/.ssh or any credential store.
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${CI_RUNTIME_DIR:-/opt/cloud-inference-framework}"
data_root="${CI_DATA_ROOT:-}"
model_repo="${CI_MODEL_REPO:-Qwen/Qwen3-0.6B}"
model_rev="${CI_MODEL_REV:-c1899de289a04d12100db370d81485cdf75e47ca}"
vllm_rev="${CI_VLLM_REV:-db9527a46873454610df6dbedf79a36d6bf1a7f6}"
mode="full"

for arg in "$@"; do
    case "$arg" in
        --facts-only) mode="facts";;
        --torch-only) mode="torch-only";;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0;;
        *) printf 'unknown argument: %s\n' "$arg" >&2; exit 2;;
    esac
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$runtime_dir/results/bootstrap-remote-$stamp"
mkdir -p "$out"
log="$out/bootstrap.log"
exec > >(tee -a "$log") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*"; printf 'log: %s\n' "$log"; exit 1; }
mark() { printf '%s=%s\n' "$1" "$2" >> "$out/report.env"; }

say "host facts"
{
    printf 'utc=%s\n' "$stamp"
    uname -a
    printf '\n--- nvidia-smi ---\n'
    nvidia-smi || printf 'nvidia-smi failed\n'
    printf '\n--- python ---\n'
    python3 --version 2>&1 || true
    printf '\n--- disk ---\n'
    df -h "$runtime_dir" / 2>/dev/null || df -h
    printf '\n--- memory ---\n'
    free -g 2>/dev/null || true
} | tee "$out/facts.txt" >/dev/null
mark runtime_dir "$runtime_dir"
mark python "$(python3 --version 2>&1 | awk '{print $2}')"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
mark driver "$driver"
mark gpu "$gpu"
mark vram "$vram"
printf 'gpu=%s driver=%s vram=%s\n' "$gpu" "$driver" "$vram"
printf 'facts written to %s\n' "$out/facts.txt"

if [[ "$mode" == "facts" ]]; then
    say "facts-only: stopping before any installation"
    exit 0
fi

# A 2.13-era torch wheel is built for CUDA 13 and needs a 580+ driver.  Older
# drivers still work for the torch-only stages (the min-p kernel is Triton and
# never touches vLLM), so we do not hard-fail on the driver: we record it.
driver_major="${driver%%.*}"
if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major < 580 )) && [[ "$mode" == "full" ]]; then
    printf '\nWARNING: driver %s is older than 580; the default torch wheel may refuse to load.\n' "$driver"
    printf 'If step "venv" or "install" fails with a driver error, rerun with --torch-only\n'
    printf 'and run only the torch/Triton stages (quant-probe, minp-sweep, minp-compare).\n'
    mark driver_warning "older than 580"
fi

say "layout"
mkdir -p "$runtime_dir" "$runtime_dir/results" "$runtime_dir/scripts" "$runtime_dir/kernels"
for sub in .venv models vllm-src; do
    target="$runtime_dir/$sub"
    if [[ -n "$data_root" && ! -e "$target" ]]; then
        mkdir -p "$data_root/ci-fw"
        mkdir -p "$data_root/ci-fw/$(basename "$sub")"
        ln -s "$data_root/ci-fw/$(basename "$sub")" "$target"
        printf 'symlinked %s -> %s\n' "$target" "$(readlink -f "$target")"
    fi
done
mkdir -p "$runtime_dir/results"

say "venv"
if [[ ! -x "$runtime_dir/.venv/bin/python" ]]; then
    python3 -m venv "$runtime_dir/.venv" || die "python3 -m venv failed (install python3-venv?)"
fi
pip="$runtime_dir/.venv/bin/pip"
"$pip" install -q --upgrade pip || die "pip upgrade failed"

say "install python packages"
if [[ "$mode" == "torch-only" ]]; then
    "$pip" install -q torch triton 2>&1 | tail -20 || die "torch install failed"
else
    "$pip" install -q "vllm==0.31.0" 2>&1 | tail -20 || die "vllm install failed"
fi

say "verify imports"
"$runtime_dir/.venv/bin/python" - <<'PY' | tee "$out/versions.txt"
import importlib, platform
for name in ("torch", "triton", "vllm"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001 - report, do not hide
        print(f"{name}=MISSING ({type(exc).__name__}: {exc})")
try:
    import torch
    print(f"torch_cuda={torch.version.cuda} available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
except Exception as exc:  # noqa: BLE001
    print(f"torch_probe_failed={type(exc).__name__}: {exc}")
print(f"platform={platform.platform()}")
PY

say "pinned vLLM source (only needed by the min-p comparison stage)"
if [[ ! -f "$runtime_dir/vllm-src/vllm/v1/worker/gpu/sample/min_p.py" ]]; then
    git init -q "$runtime_dir/vllm-src" 2>/dev/null || true
    git -C "$runtime_dir/vllm-src" remote add origin https://github.com/vllm-project/vllm 2>/dev/null ||
        git -C "$runtime_dir/vllm-src" remote set-url origin https://github.com/vllm-project/vllm
    if git -C "$runtime_dir/vllm-src" fetch -q --depth 1 origin "$vllm_rev" &&
        git -C "$runtime_dir/vllm-src" checkout -q FETCH_HEAD; then
        printf 'checked out vllm %s\n' "$vllm_rev"
    else
        printf 'WARNING: could not fetch vLLM %s; stages needing vllm-src will fail.\n' "$vllm_rev"
        printf 'If GitHub is slow, try: source /etc/network_turbo   (AutoDL image helper)\n'
        mark vllm_src "missing"
    fi
else
    printf 'vllm-src already present\n'
fi

say "model $model_repo@$model_rev"
model_dir="$runtime_dir/models/$(basename "$model_repo")"
if [[ -f "$model_dir/config.json" ]]; then
    printf 'model already present at %s\n' "$model_dir"
else
    "$pip" install -q modelscope 2>&1 | tail -5 || printf 'modelscope install failed, will try huggingface\n'
    mkdir -p "$model_dir"
    if command -v modelscope >/dev/null 2>&1 &&
        modelscope download --model "$model_repo" --revision "$model_rev" --local_dir "$model_dir" 2>&1 | tail -5; then
        printf 'model downloaded with modelscope\n'
    else
        export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
        "$pip" install -q "huggingface_hub[cli]" 2>&1 | tail -5 || true
        "$runtime_dir/.venv/bin/huggingface-cli" download "$model_repo" --revision "$model_rev" \
            --local-dir "$model_dir" 2>&1 | tail -5 ||
            die "model download failed (modelscope and huggingface mirror)"
    fi
fi
if [[ -f "$model_dir/config.json" ]]; then
    mark model "$model_dir"
    printf 'model config: %s\n' "$model_dir/config.json"
fi

say "preflight (one real forward pass)"
CI_RUNTIME_DIR="$runtime_dir" bash "$project_dir/scripts/preflight.sh" || die "preflight failed"

say "done"
printf 'next, on this host:\n'
printf '  CI_RUNTIME_DIR=%s bash %s/scripts/repro_l4.sh --dry-run\n' "$runtime_dir" "$project_dir"
printf '  CI_RUNTIME_DIR=%s bash %s/scripts/repro_l4.sh\n' "$runtime_dir" "$project_dir"
printf 'report: %s\n' "$out/report.env"
