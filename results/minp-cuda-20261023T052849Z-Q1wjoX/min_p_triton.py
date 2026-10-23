"""Two-stage min-p filtering for large vocabularies on small GPU batches.

Contract follows vLLM v0.31.0 gpu/sample/min_p.py (Apache-2.0).
Each row is modified in place; rows with min_p=0 stay untouched.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _min_p_partial_max(X, Mapping, P, Partial, STRIDE: tl.constexpr,
                       V: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    req = tl.load(Mapping + row)
    p = tl.load(P + req)
    if p == 0.0:
        return
    cols = part * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + row * STRIDE + cols, cols < V, float("-inf")).to(tl.float32)
    maximum = tl.max(values, 0)
    tl.store(Partial + row * PARTS + part, maximum)


@triton.jit
def _min_p_apply_threshold(X, Mapping, P, Partial, STRIDE: tl.constexpr,
                           V: tl.constexpr, PARTS: tl.constexpr,
                           REDUCE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    req = tl.load(Mapping + row)
    p = tl.load(P + req).to(tl.float32)
    if p == 0.0:
        return
    indices = tl.arange(0, REDUCE)
    maxima = tl.load(Partial + row * PARTS + indices,
                     indices < PARTS, float("-inf"))
    threshold = tl.max(maxima, 0) + tl.log(p)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + row * STRIDE + cols, cols < V, float("-inf"))
    filtered = tl.where(values < threshold, float("-inf"), values)
    tl.store(X + row * STRIDE + cols, filtered, cols < V)


def apply_min_p(logits, expanded_idx_mapping, min_p):
    rows, vocab = logits.shape
    if rows == 0 or vocab == 0:
        return
    if logits.stride(1) != 1:
        raise ValueError("The min-p ABI requires contiguous columns")
    block = 4096
    parts = triton.cdiv(vocab, block)
    partial = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    _min_p_partial_max[(rows, parts)](
        logits, expanded_idx_mapping, min_p, partial, logits.stride(0),
        vocab, parts, block, num_warps=4,
    )
    _min_p_apply_threshold[(rows, parts)](
        logits, expanded_idx_mapping, min_p, partial, logits.stride(0),
        vocab, parts, triton.next_power_of_2(parts), block, num_warps=4,
    )


def apply_min_p_optimized(logits, expanded_idx_mapping, min_p):
    rows, vocab = logits.shape
    if 1 <= rows <= 8 and vocab >= 32000 and logits.stride(1) == 1:
        return apply_min_p(logits, expanded_idx_mapping, min_p)
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p as original
    return original(logits, expanded_idx_mapping, min_p)
