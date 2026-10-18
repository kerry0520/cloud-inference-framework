"""Does halving the weight bytes halve the decode step?  A same-process intervention test.

The model fitted in stage five says a batch-2 decode step costs about
(weight bytes) / (achievable bandwidth) plus a fixed tail.  Correlation is not
enough to defend that claim, so this script performs an *intervention*: run the
same engine, same model, same prompts twice -- once with fp16 weights, once with
fp8 W8A8 weights (half the bytes) -- and compare the per-token decode interval.

Conditional prediction from the stage-three trace: layer GEMMs are 63.4% of
kernel time (83.8% includes the unquantized lm_head). If only layer GEMM time
halves, total kernel time falls 31.7%. This is not a request-time prediction:
W8A8 changes kernels/fusion and adds activation quantization.

fp8 changes numerics, so token ids are recorded for reference and never asserted;
the numerical contract of a kernel is checked elsewhere, byte by byte.  Engines
are booted one at a time (a 8 GB card cannot hold two engines with graphs) and
rounds alternate A/B/B/A to cancel clock drift.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback

MODEL = str(Path(os.environ.get("CI_RUNTIME_DIR", "/opt/cloud-inference-framework")) / "models/Qwen3-0.6B")


def bootstrap_ci(values, iterations=10000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.mean(values[rng.randrange(n)] for _ in range(n))
             for _ in range(iterations)]
    means.sort()
    return [means[int(0.025 * iterations)], means[int(0.975 * iterations)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="fp16,fp8")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--prompt-len", type=int, default=128,
                        help="single prompt length (kept for compatibility)")
    parser.add_argument("--prompt-lens", default="",
                        help="comma list of prompt lengths for a context sweep")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-fraction", type=float, default=0.55)
    args = parser.parse_args()
    if args.max_tokens < 2 or args.rounds < 1:
        parser.error("--max-tokens must be >=2 and --rounds >=1")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    batches = [int(x) for x in args.batches.split(",")]
    prompt_lens = ([int(x) for x in args.prompt_lens.split(",") if x.strip()]
                   if args.prompt_lens else [args.prompt_len])

    try:
        import torch
        from vllm import LLM, SamplingParams

        longest = max(prompt_lens) + args.max_tokens

        def boot(arm):
            kwargs = dict(model=MODEL, dtype="float16", max_model_len=longest,
                          max_num_seqs=max(batches),
                          max_num_batched_tokens=max(prompt_lens) * max(batches),
                          gpu_memory_utilization=args.gpu_fraction,
                          enable_prefix_caching=False, seed=0,
                          disable_log_stats=False)
            if arm != "fp16":
                # On-the-fly W8A8 quantisation: vLLM quantises the checkpoint at
                # load time, no calibrated file needed.
                kwargs["quantization"] = arm
            started = time.perf_counter()
            llm = LLM(**kwargs)
            boot_seconds = time.perf_counter() - started
            applied = getattr(llm.llm_engine.model_config, "quantization", None)
            print(json.dumps({"booted": arm, "seconds": round(boot_seconds, 1),
                              "quantization": str(applied)}), flush=True)
            return llm, boot_seconds, str(applied)

        def free(llm):
            # The V1 engine runs its core in a child process; dropping the
            # handle is not enough, the child needs a moment to tear down its
            # weights and CUDA graphs.  Wait until the card really is free
            # again, otherwise the next arm boots into a half-empty card.
            llm.llm_engine.engine_core.shutdown()
            del llm
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            for _ in range(60):
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes >= 5 * 1024 ** 3:
                    break
                time.sleep(0.5)
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise RuntimeError("Engine shutdown did not release at least 5 GiB within 30 seconds")
            gc.collect()
            torch.cuda.empty_cache()

        def run(llm, batch, prompt_len):
            prompts = [{"prompt_token_ids": [1000 + i] * prompt_len}
                       for i in range(batch)]
            params = [SamplingParams(temperature=0.0, seed=0,
                                     max_tokens=args.max_tokens,
                                     min_tokens=args.max_tokens,
                                     ignore_eos=True, detokenize=False)
                      for _ in range(batch)]
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            wall = time.perf_counter() - started
            intervals = []
            for output in outputs:
                metrics = output.metrics
                if metrics is None or getattr(metrics, "first_token_ts", None) is None:
                    raise RuntimeError("Missing first-token metrics")
                if len(output.outputs[0].token_ids) != args.max_tokens:
                    raise RuntimeError("Unexpected output length")
                intervals.append(1000 * (metrics.last_token_ts - metrics.first_token_ts)
                                 / (args.max_tokens - 1))
            ids = [list(output.outputs[0].token_ids) for output in outputs]
            if len(intervals) != batch:
                raise RuntimeError("Incomplete request metrics")
            return {"wall": wall, "intervals": intervals, "ids": ids}

        evidence = {
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "model": MODEL, "arms": arms, "rounds": args.rounds,
            "batches": batches, "prompt_lens": prompt_lens,
            "max_tokens": args.max_tokens, "sampling": "greedy",
            "gpu_fraction": args.gpu_fraction,
            "quantization_applied": {}, "boot_seconds": {},
        }
        evidence["prediction"] = {
            "reference": "stage-three instrumented trace; conditional estimate only",
            "layer_gemm_share": 0.634, "lm_head_share": 0.204,
            "conditional_fp8_pct": -31.7,
            "assumptions": ("Only layer GEMM time halves; all other costs unchanged. "
                            "W8A8 also changes kernels/fusion and adds activation quantization."),
        }
        (out / "config.json").write_text(json.dumps(evidence, indent=2))

        rows = []
        with (out / "timing.jsonl").open("w") as stream:
            for round_index in range(args.rounds):
                order = arms if round_index % 2 == 0 else list(reversed(arms))
                for arm in order:
                    llm, boot_seconds, applied = boot(arm)
                    evidence["boot_seconds"].setdefault(arm, []).append(boot_seconds)
                    evidence["quantization_applied"][arm] = applied
                    try:
                        for batch in batches:
                            for prompt_len in prompt_lens:
                                try:
                                    run(llm, batch, prompt_len)   # warmup
                                    result = run(llm, batch, prompt_len)
                                except Exception as error:        # context too long
                                    row = {"round": round_index, "arm": arm,
                                           "batch": batch, "prompt_len": prompt_len,
                                           "skipped": repr(error)}
                                    stream.write(json.dumps(row) + "\n")
                                    stream.flush()
                                    print(json.dumps(row), flush=True)
                                    continue
                                row = {"round": round_index, "arm": arm,
                                       "batch": batch, "prompt_len": prompt_len,
                                       "wall_seconds": result["wall"],
                                       "decode_ms": result["intervals"],
                                       "output_ids": result["ids"]}
                                rows.append(row)
                                stream.write(json.dumps(row) + "\n")
                                stream.flush()
                                print(json.dumps({
                                    "round": round_index, "arm": arm,
                                    "batch": batch, "prompt_len": prompt_len,
                                    "wall_ms": round(result["wall"] * 1000, 1),
                                    "decode_ms": [round(v, 3)
                                                  for v in result["intervals"]]}),
                                    flush=True)
                    finally:
                        free(llm)

        summaries = []
        expected_rows = args.rounds * len(arms) * len(batches) * len(prompt_lens)
        if len(rows) != expected_rows:
            raise RuntimeError(f"Incomplete measurement matrix: {len(rows)} / {expected_rows}; inspect timing.jsonl")
        for prompt_len in prompt_lens:
            for batch in batches:
                usable = [r for r in rows if r.get("prompt_len") == prompt_len
                          and r["batch"] == batch and r.get("decode_ms")]
                per_arm = {}
                for arm in arms:
                    values = [v for r in usable if r["arm"] == arm
                              for v in r["decode_ms"]]
                    per_arm[arm] = {"median_decode_ms":
                                    statistics.median(values) if values else None,
                                    "values": values}
                round_pcts = []
                for round_index in range(args.rounds):
                    base = [v for r in usable if r["round"] == round_index
                            and r["arm"] == arms[0] for v in r["decode_ms"]]
                    other = [v for r in usable if r["round"] == round_index
                             and r["arm"] == arms[-1] for v in r["decode_ms"]]
                    if len(arms) > 1 and base and other:
                        round_pcts.append(100 * (statistics.mean(other)
                                                 - statistics.mean(base))
                                          / statistics.mean(base))
                summaries.append({
                    "prompt_len": prompt_len, "batch": batch,
                    "per_arm": {k: v["median_decode_ms"] for k, v in per_arm.items()},
                    "round_paired_pct": round_pcts,
                    "paired_pct_median": statistics.median(round_pcts) if round_pcts else None,
                    "paired_pct_ci95": (bootstrap_ci(round_pcts, seed=batch)
                                        if len(round_pcts) > 2 else None),
                })
                print(json.dumps({"prompt_len": prompt_len, "batch": batch,
                                  "median_decode_ms": summaries[-1]["per_arm"],
                                  "paired_pct": summaries[-1]["paired_pct_median"]}),
                      flush=True)

        evidence["summary"] = summaries
        evidence["timing_sha256"] = hashlib.sha256((out / "timing.jsonl").read_bytes()).hexdigest()
        evidence["timing_rows"] = len(rows)
        (out / "summary.json").write_text(json.dumps(evidence, indent=2))
        (out / "status.json").write_text(json.dumps({"state": "passed",
                                                     "rows": len(rows)}, indent=2))
    except BaseException:
        (out / "status.json").write_text(json.dumps({"state": "failed",
                                                     "traceback": traceback.format_exc()},
                                                    indent=2))
        raise


if __name__ == "__main__":
    main()
