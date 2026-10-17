#!/usr/bin/env bash
set -euo pipefail
runtime_dir=/opt/cloud-inference-framework
pip_cuda="$runtime_dir/.venv/lib/python3.12/site-packages/nvidia/cu13"
toolkit="$runtime_dir/cuda-toolkit"
test -x "$pip_cuda/bin/nvcc"
test -f "$pip_cuda/lib/libcudart.so.13"
test -f /usr/lib/wsl/lib/libcuda.so.1
mkdir -p "$toolkit/lib64/stubs"
for part in bin include nvvm; do ln -sfn "$pip_cuda/$part" "$toolkit/$part"; done
ln -sfn "$pip_cuda/lib/libcudart.so.13" "$toolkit/lib64/libcudart.so.13"
ln -sfn "$pip_cuda/lib/libcudart.so.13" "$toolkit/lib64/libcudart.so"
ln -sfn /usr/lib/wsl/lib/libcuda.so.1 "$toolkit/lib64/libcuda.so"
ln -sfn /usr/lib/wsl/lib/libcuda.so.1 "$toolkit/lib64/stubs/libcuda.so"
"$toolkit/bin/nvcc" --version
