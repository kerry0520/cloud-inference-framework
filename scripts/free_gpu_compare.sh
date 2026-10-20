#!/usr/bin/env bash
# One-command kernel-level cross-check for a free or foreign GPU box.
#
# It needs only PyTorch + Triton (no vLLM, no model weights), so it runs on a
# free Colab/Kaggle T4, a rented 4090, or any other Linux GPU host.  The engine
# level numbers (abba-*) still need the full stack; see
# docs/远程GPU租用与迁移-20261018.md for that path.
#
# Usage, from a checkout of this repository:
#   bash scripts/free_gpu_compare.sh                     # fp16, vocab 151936
#   bash scripts/free_gpu_compare.sh --dtype float32
#   bash scripts/free_gpu_compare.sh --batches 1,2,4,8 --vocabs 32000,151936
#   PINNED_MINP=/path/to/min_p.py bash scripts/free_gpu_compare.sh  # offline
#
# Outputs: $OUT/pinned/min_p.py, $OUT/pr-module/, $OUT/compare/compare.json
# Copy the whole $OUT back and record its path in the stage document.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=""
VOCABS="151936"
BATCHES="1,2,4,8"
DTYPE="float16"
INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --vocabs) VOCABS="$2"; shift 2 ;;
    --batches) BATCHES="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT" ]]; then
  OUT="$ROOT/results/free-compare-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUT"

PINNED_COMMIT="db9527a46873454610df6dbedf79a36d6bf1a7f6"
PINNED_SHA256="4cb5e2f45e1ced6f9685283a7a5c8cfd7204d9a249822e866cfff756c7f04f62"
PR_COMMIT="95e9d21aaff3605e25a4797e54e975c0f93e83b7"
PR_DIFF="$ROOT/results/upstream-pr-58907-95e9d21.diff"

echo "== environment"
python3 -c 'import sys; print("python", sys.version.split()[0])'
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("capability", cap)
PY

if ! python3 -c 'import torch, triton' 2>/dev/null; then
  echo "missing torch or triton" >&2
  if [[ "$INSTALL" == "1" ]]; then
    echo "installing a CUDA build of torch + triton; adjust the index URL for your driver" >&2
    python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu126
    python3 -m pip install --quiet triton
  else
    echo "rerun with --install, or install torch/triton yourself" >&2
    exit 3
  fi
fi

echo "== pinned upstream min_p.py"
PINNED="$OUT/pinned/min_p.py"
if [[ -n "${PINNED_MINP:-}" ]]; then
  mkdir -p "$OUT/pinned"
  cp -f "$PINNED_MINP" "$PINNED"
else
  mkdir -p "$OUT/pinned"
  URL="https://raw.githubusercontent.com/vllm-project/vllm/$PINNED_COMMIT/vllm/v1/worker/gpu/sample/min_p.py"
  echo "GET $URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "$PINNED"
  else
    wget -q "$URL" -O "$PINNED"
  fi
fi
ACTUAL="$(python3 - "$PINNED" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
if [[ "$ACTUAL" != "$PINNED_SHA256" ]]; then
  echo "pinned min_p.py sha256 mismatch: $ACTUAL" >&2
  echo "expected $PINNED_SHA256 (vLLM $PINNED_COMMIT)" >&2
  exit 4
fi
echo "sha256 ok: $ACTUAL"

# The upstream kernel imports only `vllm.triton_utils` (tl, triton).  A host
# without vLLM can still measure the kernels through a two-line shim; record it
# so the evidence says the kernels were not run under the full package.
if ! python3 -c 'import vllm.triton_utils' 2>/dev/null; then
  mkdir -p "$OUT/shim/vllm"
  : > "$OUT/shim/vllm/__init__.py"
  cat > "$OUT/shim/vllm/triton_utils.py" <<'PY'
# Minimal stand-in for vllm.triton_utils on hosts without vLLM.
import triton
from triton import language as tl

__all__ = ["tl", "triton"]
PY
  export PYTHONPATH="$OUT/shim${PYTHONPATH:+:$PYTHONPATH}"
  echo "vllm absent: using $OUT/shim (kernels only need tl/triton)"
fi


echo "== reconstruct the PR kernel from its diff"
python3 "$ROOT/scripts/prepare_pr_module.py" \
  --out "$OUT/pr-module" \
  --target "$OUT/pr-module" \
  --upstream "$PINNED" \
  --diff "$PR_DIFF" \
  --commit "$PR_COMMIT"

echo "== compare upstream / knob / PR"
python3 "$ROOT/scripts/minp_variant_compare.py" \
  --out "$OUT/compare" \
  --upstream "$PINNED" \
  --knob "$ROOT/patches/mrv2_min_p_knob.py" \
  --pr-diff "$PR_DIFF" \
  --vocabs "$VOCABS" \
  --batches "$BATCHES" \
  --dtype "$DTYPE"

echo
echo "done: $OUT"
echo "copy back with:  rsync -az <host>:$OUT/ results/free-compare-<date>/"
