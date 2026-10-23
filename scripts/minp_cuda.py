"""Acceptance + benchmark for the CUDA min-p kernel (P1, kernels/min_p_cuda.cu).

Two questions, kept apart:

1. Numerics: on the same matrix the Triton candidate passed (three dtypes,
   vocab 1..151936, padding columns, -inf/inf/NaN rows), do the CUDA kernel and
   upstream produce byte-identical logits?  The same matrix also probes whether
   upstream's NaN semantics match NAN_MODE 0 (fmaxf, NaN ignored) or 1 (NaN
   propagated), so the choice is measured instead of assumed.
2. Speed: local kernel time at the shapes the sampler actually issues
   (vocab 151936, batch 1/2/4/8), against upstream and against the Triton
   two-stage candidate.  Host overhead is part of the number: this is what the
   sampler pays per decode step.

Writes correctness.json, timing.jsonl and status.json into --out.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocabs", default="1,31,1023,1024,1025,4095,4096,4097,32000,151936")
    parser.add_argument("--batches", default="1,2,8")
    parser.add_argument("--timing-vocabs", default="4096,32000,151936")
    parser.add_argument("--timing-batches", default="1,2,4,8")
    parser.add_argument("--dtypes", default="fp16,fp32,bf16")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--sweep", default="2048x128,2048x256,4096x128,4096x256,4096x512,8192x256,8192x512")
    parser.add_argument("--sweep-vocab", type=int, default=151936)
    parser.add_argument("--sweep-batches", default="1,2,4,8")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    state = {"state": "running", "stage": "minp-cuda"}
    try:
        import torch
        import triton.testing

        from kernels import min_p as triton_rel
        from kernels import min_p_cuda

        from vllm.v1.worker.gpu.sample.min_p import apply_min_p as upstream

        shutil.copyfile(Path(min_p_cuda.source_path()), out / "min_p_cuda.cu")
        shutil.copyfile(Path(triton_rel.__file__), out / "min_p_triton.py")
        module = min_p_cuda.build()
        state["build"] = min_p_cuda.build_info()
        if args.build_only:
            state["state"] = "passed"
            (out / "status.json").write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps(state["build"], indent=2), flush=True)
            return

        dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        vocab_list = [int(v) for v in args.vocabs.split(",")]
        batch_list = [int(b) for b in args.batches.split(",")]
        timing_vocabs = [int(v) for v in args.timing_vocabs.split(",")]
        timing_batches = [int(b) for b in args.timing_batches.split(",")]
        dtypes = [dtype_map[d.strip()] for d in args.dtypes.split(",")]

        def make_case(vocab: int, batch: int, dtype, kind: str):
            storage = torch.full((batch, vocab + 7), 777.0, device="cuda", dtype=dtype)
            values = torch.randn((batch, vocab), device="cuda", dtype=dtype)
            if kind == "special":
                values[0].fill_(float("-inf"))
                if batch > 1:
                    values[1, 0] = float("inf")
                    values[1, -1] = float("nan")
                if batch > 2:
                    values[2].zero_()
            storage[:, :vocab] = values
            mapping = torch.tensor([(i * 3 + 2) % 11 for i in range(batch)],
                                   device="cuda", dtype=torch.int32)
            probs = torch.tensor([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 0.2, 0.0, 0.7, 1.0, 0.9],
                                 device="cuda", dtype=torch.float32)
            return storage, mapping, probs

        cases = []
        nan_probe = []

        def run_case(vocab: int, batch: int, dtype, kind: str):
            storage, mapping, probs = make_case(vocab, batch, dtype, kind)
            logits = storage[:, :vocab]
            reference = logits.clone()
            upstream(reference, mapping, probs)
            row = {"vocab": vocab, "batch": batch, "dtype": str(dtype), "kind": kind}
            for nan_mode in (0, 1):
                actual = logits.clone()
                min_p_cuda.apply_min_p(actual, mapping, probs, nan_mode=nan_mode)
                torch.cuda.synchronize()
                equal_upstream = torch.equal(reference.view(torch.uint8),
                                             actual.view(torch.uint8))
                if nan_mode == 0:
                    row["byte_equal_upstream"] = equal_upstream
                    row["padding_untouched"] = torch.equal(
                        storage[:, vocab:].view(torch.uint8),
                        torch.full_like(storage[:, vocab:], 777.0).view(torch.uint8),
                    )
                    triton_out = logits.clone()
                    triton_rel.apply_min_p(triton_out, mapping, probs)
                    torch.cuda.synchronize()
                    row["byte_equal_triton"] = torch.equal(reference.view(torch.uint8),
                                                           triton_out.view(torch.uint8))
                else:
                    row["nan_mode1_equal_upstream"] = equal_upstream
            cases.append(row)
            (out / "correctness.json").write_text(json.dumps(cases, indent=2))
            return row

        for dtype in dtypes:
            for vocab in vocab_list:
                for batch in batch_list:
                    run_case(vocab, batch, dtype, "random")
            for vocab in (4097, 151936):
                run_case(vocab, 8, dtype, "special")
        # Which NaN semantics does the pinned upstream have?  Compare directly on
        # rows that contain NaN instead of relying on the random matrix.
        probe_storage, probe_map, probe_probs = make_case(151936, 8, torch.float16, "special")
        probe_logits = probe_storage[:, :151936]
        probe_ref = probe_logits.clone()
        upstream(probe_ref, probe_map, probe_probs)
        for nan_mode in (0, 1):
            probe = probe_logits.clone()
            min_p_cuda.apply_min_p(probe, probe_map, probe_probs, nan_mode=nan_mode)
            torch.cuda.synchronize()
            nan_probe.append({"nan_mode": nan_mode,
                              "equal_upstream": torch.equal(probe_ref.view(torch.uint8),
                                                            probe.view(torch.uint8))})
        (out / "nan_probe.json").write_text(json.dumps(nan_probe, indent=2))

        failures = [c for c in cases if not (c.get("byte_equal_upstream")
                                             and c.get("byte_equal_triton")
                                             and c.get("padding_untouched"))]
        state["correctness_cases"] = len(cases)
        state["correctness_failures"] = failures
        state["nan_probe"] = nan_probe
        if failures:
            raise AssertionError(f"{len(failures)} correctness cases failed: {failures[:3]}")

        # Launch-config sweep.  The chunked kernel is bound by how many CTAs stay
        # resident (26 SMs here), so chunk/threads trade CTA count against work
        # per CTA.  Measure on the shapes that matter in serving (fp16,
        # V151936, batch 1..8), require byte equality for every variant, and
        # pick the fixed default that the .cu then hard-codes.
        sweep_cfg = [tuple(int(v) for v in c.split("x"))
                     for c in args.sweep.split(",") if c.strip()]
        sweep_rows = []
        with (out / "sweep.jsonl").open("w") as stream:
            for batch in [int(b) for b in args.sweep_batches.split(",")]:
                vocab = args.sweep_vocab
                x = torch.randn((batch, vocab), device="cuda", dtype=torch.float16)
                mapping = torch.arange(batch, device="cuda", dtype=torch.int32)
                probs = torch.full((batch,), 0.05, device="cuda")
                reference = x.clone()
                upstream(reference, mapping, probs)
                best = None
                for chunk, threads in sweep_cfg:
                    trial = x.clone()
                    min_p_cuda.apply_min_p(trial, mapping, probs, chunk=chunk,
                                           threads=threads)
                    equal = torch.equal(reference.view(torch.uint8),
                                        trial.view(torch.uint8))
                    row = {"vocab": vocab, "batch": batch, "chunk": chunk,
                           "threads": threads, "byte_equal_upstream": bool(equal)}
                    if equal:
                        values = triton.testing.do_bench(
                            lambda c=chunk, t=threads: min_p_cuda.apply_min_p(
                                x, mapping, probs, chunk=c, threads=t),
                            warmup=5, rep=20, quantiles=[0.5])
                        # do_bench returns a bare float for a single quantile.
                        median = values if not isinstance(values, (list, tuple)) else values[0]
                        row["median_us"] = median * 1000
                    sweep_rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    if equal and (best is None or row["median_us"] < best["median_us"]):
                        best = row
                if best is not None:
                    print(json.dumps({"sweep_best": best}), flush=True)
        state["sweep_rows"] = len(sweep_rows)
        state["sweep_failures"] = [r for r in sweep_rows if not r["byte_equal_upstream"]]
        if state["sweep_failures"]:
            raise AssertionError(f"launch sweep changed results: {state['sweep_failures'][:3]}")

        timing = []
        with (out / "timing.jsonl").open("w") as stream:
            for dtype in (torch.float16, torch.float32):
                for vocab in timing_vocabs:
                    for batch in timing_batches:
                        x = torch.randn((batch, vocab), device="cuda", dtype=dtype)
                        mapping = torch.arange(batch, device="cuda", dtype=torch.int32)
                        probs = torch.full((batch,), 0.05, device="cuda")
                        arms = [("base", upstream), ("cuda", min_p_cuda.apply_min_p),
                                ("triton", triton_rel.apply_min_p), ("cuda", min_p_cuda.apply_min_p),
                                ("base", upstream)]
                        for arm, fn in arms:
                            values = triton.testing.do_bench(
                                lambda fn=fn: fn(x, mapping, probs),
                                warmup=5, rep=20, quantiles=[0.5, 0.2, 0.8])
                            row = {"dtype": str(dtype), "vocab": vocab, "batch": batch,
                                   "arm": arm, "median_us": values[0] * 1000,
                                   "p20_us": values[1] * 1000, "p80_us": values[2] * 1000}
                            timing.append(row)
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
                        recent = timing[-5:]
                        base = statistics.median(r["median_us"] for r in recent if r["arm"] == "base")
                        cuda = statistics.median(r["median_us"] for r in recent if r["arm"] == "cuda")
                        tri = statistics.median(r["median_us"] for r in recent if r["arm"] == "triton")
                        line = {"dtype": str(dtype), "vocab": vocab, "batch": batch,
                                "base_us": base, "cuda_us": cuda, "triton_us": tri,
                                "cuda_speedup": base / cuda, "cuda_vs_triton": tri / cuda}
                        print(json.dumps(line), flush=True)
        state.update({"state": "passed", "timing_rows": len(timing)})
        (out / "status.json").write_text(json.dumps(state, indent=2) + "\n")
    except BaseException:
        state["state"] = "failed"
        state["traceback"] = traceback.format_exc()
        (out / "status.json").write_text(json.dumps(state, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
