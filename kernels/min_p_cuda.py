"""Runtime loader for the CUDA min-p kernel (kernels/min_p_cuda.cu).

The extension is built on first use with ``torch.utils.cpp_extension.load`` and
cached under ``$CI_RUNTIME_DIR/cache/minp_cuda`` so that repeated stages (and
``vllm serve`` restarts) reuse the same ``.so``.  Building never touches
``vllm-src``: the call site is redirected at import time by
``scripts/sitecustomize.py``, exactly like the Triton candidate.

Numerical contract: byte-identical logits to upstream
``vllm.v1.worker.gpu.sample.min_p.apply_min_p``.  The wrapper only decides
*which* implementation runs; it must not change thresholds or dtypes.

NaN semantics were measured, not assumed: on rows containing NaN the pinned
upstream matches ``nan_mode=0`` (fmaxf, NaN ignored), while ``nan_mode=1``
(NaN propagated) diverges.  See ``scripts/minp_cuda.py``'s nan_probe output.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

import torch

_EXTENSION_NAME = "min_p_cuda_ext"

_module = None
_build_info: dict | None = None


def build_dir() -> Path:
    """Cache directory for the compiled extension (stable across stages)."""
    root = os.environ.get("CI_MINP_CUDA_BUILD")
    if root:
        return Path(root)
    runtime = os.environ.get("CI_RUNTIME_DIR") or str(Path.home() / ".cache")
    return Path(runtime) / "cache" / "minp_cuda"


def source_path() -> Path:
    return Path(__file__).resolve().with_suffix(".cu")


def _nvcc_version() -> str:
    nvcc = os.environ.get("CUDA_HOME")
    exe = str(Path(nvcc) / "bin" / "nvcc") if nvcc else "nvcc"
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
        for line in out.stdout.splitlines():
            if "release" in line:
                return line.strip()
    except Exception as exc:  # pragma: no cover - evidence only
        return f"unavailable: {exc!r}"
    return "unknown"


def build(verbose: bool = False):
    """Compiles (or reuses) the extension and records how it was built."""
    global _module, _build_info
    if _module is not None:
        return _module
    # The pinned python in the runtime ships no TORCH_CUDA_ARCH_LIST; state the
    # target explicitly instead of relying on device detection inside WSL.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    from torch.utils.cpp_extension import load

    target = build_dir()
    target.mkdir(parents=True, exist_ok=True)
    _module = load(
        name=_EXTENSION_NAME,
        sources=[str(source_path())],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        build_directory=str(target),
        verbose=verbose,
    )
    _build_info = {
        "extension": _EXTENSION_NAME,
        "source": str(source_path()),
        "build_dir": str(target),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvcc": _nvcc_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arch": "sm_120",
        "python": platform.python_version(),
    }
    (target / "build_info.json").write_text(json.dumps(_build_info, indent=2) + "\n")
    return _module


def build_info() -> dict:
    if _build_info is None:
        build()
    return dict(_build_info or {})


def apply_min_p(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
    *,
    nan_mode: int = 0,
    chunk: int | None = None,
    threads: int | None = None,
) -> None:
    """In-place min-p filtering; same ABI as the Triton/upstream kernels.

    ``chunk``/``threads`` exist for the launch-config sweep in
    ``scripts/minp_cuda.py``; production callers use the defaults measured on
    sm_120 (see ``kDefaultChunk``/``kDefaultThreads`` in the .cu file).
    """
    if logits.numel() == 0:
        return
    module = build()
    mapping = expanded_idx_mapping
    if mapping.dtype != torch.int32:
        mapping = mapping.to(torch.int32)
    if not mapping.is_contiguous():
        mapping = mapping.contiguous()
    probs = min_p if min_p.dtype == torch.float32 else min_p.to(torch.float32)
    if not probs.is_contiguous():
        probs = probs.contiguous()
    kwargs = {"nan_mode": nan_mode}
    if chunk is not None:
        kwargs["chunk"] = chunk
    if threads is not None:
        kwargs["threads"] = threads
    module.min_p_apply(logits, mapping, probs, **kwargs)


def apply_min_p_propagating_nan(logits, expanded_idx_mapping, min_p) -> None:
    """Kept for the acceptance probe: NaN-propagating variant (not upstream)."""
    apply_min_p(logits, expanded_idx_mapping, min_p, nan_mode=1)
