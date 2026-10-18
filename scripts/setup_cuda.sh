#!/usr/bin/env bash
set -euo pipefail
runtime_dir="${CI_RUNTIME_DIR:-/opt/cloud-inference-framework}"
pip_cuda="$("$runtime_dir/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")')"
toolkit="$runtime_dir/cuda-toolkit"
test -x "$pip_cuda/bin/nvcc"
test -f "$pip_cuda/lib/libcudart.so.13"
driver_lib="${CI_CUDA_DRIVER_LIB:-}"
if [[ -z "$driver_lib" ]]; then
    if [[ -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then
        driver_lib=/usr/lib/wsl/lib/libcuda.so.1
    else
        driver_lib="$(ldconfig -p | awk '/libcuda.so.1 .*x86-64/ {if (!p) p=$NF} END {print p}')"
    fi
fi
test -f "$driver_lib"
mkdir -p "$toolkit/lib64/stubs"
for part in bin include nvvm; do ln -sfn "$pip_cuda/$part" "$toolkit/$part"; done
ln -sfn "$pip_cuda/lib/libcudart.so.13" "$toolkit/lib64/libcudart.so.13"
ln -sfn "$pip_cuda/lib/libcudart.so.13" "$toolkit/lib64/libcudart.so"
ln -sfn "$driver_lib" "$toolkit/lib64/libcuda.so"
ln -sfn "$driver_lib" "$toolkit/lib64/stubs/libcuda.so"
"$toolkit/bin/nvcc" --version
