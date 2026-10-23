# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""min-p filtering through the project's CUDA kernel (P1).

Purpose is the same as ``patches/mrv2_min_p.py``: the pinned vLLM 0.31.0
``apply_min_p`` gives every row one CTA with 1024-wide blocks, which leaves most
SMs idle when a decode step only holds <= 8 rows.  This module routes those rows
to ``kernels/min_p_cuda.cu`` and keeps upstream for everything else, so the
routing decision is the only thing that changes.

The routing window mirrors the Triton candidate (rows <= 8, vocab >= 32000,
contiguous columns, the three dtypes the acceptance matrix covers).  The CUDA
kernel and upstream must produce byte-identical logits; that is asserted by
``scripts/minp_cuda.py`` and again in-path by ``scripts/serve_stage.py``'s
``--verify-first`` mode when this module is installed through
``scripts/sitecustomize.py``.

The extension is built lazily on first use (see ``kernels/min_p_cuda.py``), so
importing this module inside a worker process is cheap and the build lands in
``$CI_RUNTIME_DIR/cache/minp_cuda``.
"""

import torch

from kernels.min_p_cuda import apply_min_p as _cuda_apply_min_p
from vllm.v1.worker.gpu.sample.min_p import apply_min_p as _upstream_apply_min_p

CUDA_MAX_ROWS = 8
CUDA_MIN_VOCAB = 32000
CUDA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _use_cuda(logits: torch.Tensor, num_tokens: int, vocab_size: int) -> bool:
    return (
        1 <= num_tokens <= CUDA_MAX_ROWS
        and vocab_size >= CUDA_MIN_VOCAB
        and logits.stride(1) == 1
        and logits.dtype in CUDA_DTYPES
    )


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    num_tokens, vocab_size = logits.shape
    if _use_cuda(logits, num_tokens, vocab_size):
        _cuda_apply_min_p(logits, expanded_idx_mapping, min_p)
        return
    _upstream_apply_min_p(logits, expanded_idx_mapping, min_p)
