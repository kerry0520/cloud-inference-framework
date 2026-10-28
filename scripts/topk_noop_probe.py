"""Temp probe (not committed): is apply_top_k_top_p a no-op, and how do CUDA events
behave when the wrapped call enqueues nothing?
"""

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    torch.manual_seed(0)
    rows, vocab = 8, 151936
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.float32)
    masked = logits.clone()
    masked[:, vocab // 2 :] = float("-inf")

    def noargs(x):
        return apply_top_k_top_p(x, None, None)

    k_full = torch.full((rows,), vocab, device="cuda", dtype=torch.int32)
    p_one = torch.ones(rows, device="cuda", dtype=torch.float32)

    def with_full(x):
        return apply_top_k_top_p(x, k_full, p_one)

    k_small = torch.full((rows,), 20, device="cuda", dtype=torch.int32)

    def with_k20(x):
        return apply_top_k_top_p(x, k_small, None)

    cases = [
        ("random  k=None p=None", lambda: noargs(logits)),
        ("-inf    k=None p=None", lambda: noargs(masked)),
        ("random  k=vocab p=1", lambda: with_full(logits)),
        ("-inf    k=vocab p=1", lambda: with_full(masked)),
        ("random  k=20", lambda: with_k20(logits)),
        ("-inf    k=20", lambda: with_k20(masked)),
    ]
    print("case                         gpu_us")
    for name, fn in cases:
        out = fn()
        same = out.data_ptr() == logits.data_ptr() or out.data_ptr() == masked.data_ptr()
        print(f"{name:28s} {bench(fn):8.1f}   returned_same_tensor={same}")


if __name__ == "__main__":
    main()
