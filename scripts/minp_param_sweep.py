#!/usr/bin/env python3
"""Sweep the launch parameters of the *upstream* MRV2 min-p kernel.

The question this answers: how much of the two-stage speedup can be had by
only changing ``BLOCK_SIZE`` and ``num_warps`` of the existing kernel?  A
two-line change is far easier to defend upstream than a second kernel, so the
PR proposal depends on where the cheap knob tops out.

The kernel below is a verbatim copy of
``vllm/v1/worker/gpu/sample/min_p.py`` at vLLM 0.31.0 (db9527a) so that the
launch parameters can be varied.  Every variant is checked against the
upstream launch (BLOCK_SIZE=1024, default num_warps=4) for byte-identical
output before its timing is trusted.  The two-stage candidate is timed through
its real entry point so the comparison uses the code that would be shipped.
"""

import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import triton
from vllm.triton_utils import tl

REPO = Path(__file__).resolve().parent.parent


@triton.jit
def _min_p_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
    if min_p == 0.0:
        return

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)

    threshold = max_val + tl.log(min_p)
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        logits = tl.where(logits < threshold, float("-inf"), logits)
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def launch(logits, mapping, min_p, block, warps):
    num_tokens, vocab = logits.shape
    kwargs = {"BLOCK_SIZE": block}
    if warps is not None:
        kwargs["num_warps"] = warps
    _min_p_kernel[(num_tokens,)](
        logits, logits.stride(0), mapping, min_p, vocab, **kwargs
    )


def timed_interleaved(calls, rounds=12, iters=100):
    """Time every variant in a shuffled round-robin, keep the min per variant.

    Each measurement launches the call ``iters`` times back to back and divides,
    so a single launch's CPU overhead is amortised the way it is in a real
    decode loop (the engine launches these kernels without synchronising).  The
    round-robin order keeps clock drift from favouring whichever variant ran
    first, and the min across rounds is the "high clock" number.
    """
    for _, fn in calls:
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    best = {name: float("inf") for name, _ in calls}
    order = list(calls)
    for _ in range(rounds):
        random.shuffle(order)
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            best[name] = min(best[name], start.elapsed_time(end) * 1e3 / iters)
    return best


def load_patched_module(path):
    spec = importlib.util.spec_from_file_location("mrv2_min_p_patched", path)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--patched", default=str(REPO / "patches" / "mrv2_min_p.py"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "triton:", triton.__version__)

    variants = []
    for block in (1024, 2048, 4096, 8192, 16384):
        for warps in (4, 8, 16):
            if block // (32 * warps) < 2:
                continue
            variants.append((block, warps))
    patched = load_patched_module(Path(args.patched))

def measure_shape(vocab, batch, specs, patched):
    """Time the requested launch variants on one (vocab, batch) shape."""
    torch.manual_seed(0)
    base = torch.randn(batch, vocab, device="cuda", dtype=torch.float16) * 2
    mapping = torch.arange(batch, device="cuda", dtype=torch.int64)
    min_p = torch.full((batch,), 0.1, device="cuda", dtype=torch.float32)

    reference = base.clone()
    launch(reference, mapping, min_p, 1024, None)

    calls = []
    for spec in specs:
        if spec == "upstream":
            scratch = base.clone()
            calls.append(
                ("upstream", lambda s=scratch: launch(s, mapping, min_p, 1024, None))
            )
        elif spec == "two_stage":
            work = base.clone()
            patched.apply_min_p(work, mapping, min_p)
            if not torch.equal(work.view(torch.uint8), reference.view(torch.uint8)):
                raise SystemExit("two_stage changed bytes")
            scratch = base.clone()
            calls.append(
                ("two_stage", lambda s=scratch: patched.apply_min_p(s, mapping, min_p))
            )
        else:
            block, warps = spec
            work = base.clone()
            launch(work, mapping, min_p, block, warps)
            if not torch.equal(work.view(torch.uint8), reference.view(torch.uint8)):
                raise SystemExit(f"variant block{block}_warps{warps} changed bytes")
            scratch = base.clone()
            calls.append(
                (
                    f"block{block}_warps{warps}",
                    lambda s=scratch, b=block, w=warps: launch(s, mapping, min_p, b, w),
                )
            )

    timings = timed_interleaved(calls)
    return [
        {
            "vocab": vocab,
            "batch": batch,
            "variant": name,
            "min_us": round(min_us, 3),
            "speedup_vs_upstream": round(timings["upstream"] / min_us, 3),
            "byte_identical": True,
        }
        for name, min_us in timings.items()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--patched", default=str(REPO / "patches" / "mrv2_min_p.py"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "triton:", triton.__version__)

    variants = []
    for block in (1024, 2048, 4096, 8192, 16384):
        for warps in (4, 8, 16):
            if block // (32 * warps) < 2:
                continue
            variants.append((block, warps))
    patched = load_patched_module(Path(args.patched))

    rows = []
    for vocab in (32000, 151936):
        for batch in (1, 2, 4, 8):
            rows += measure_shape(vocab, batch, ["upstream"] + variants + ["two_stage"], patched)

    # Large-batch check: the knob change must not regress shapes where one CTA
    # per row already fills the machine.
    large_specs = ["upstream", (16384, 4), (16384, 8), (16384, 16), (8192, 8)]
    for vocab in (32000, 151936):
        for batch in (32, 64, 256):
            rows += measure_shape(vocab, batch, large_specs, patched)

    bad = [r for r in rows if not r["byte_identical"]]
    (out / "sweep.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'vocab':>7} {'batch':>5} {'variant':>18} {'min_us':>9} {'x':>6} ok")
    for r in sorted(rows, key=lambda r: (r["vocab"], r["batch"], r["min_us"])):
        print(
            f"{r['vocab']:>7} {r['batch']:>5} {r['variant']:>18} "
            f"{r['min_us']:>9.2f} {r['speedup_vs_upstream']:>6.2f} "
            f"{r['byte_identical']}"
        )
    print("byte-identical violations:", len(bad))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
