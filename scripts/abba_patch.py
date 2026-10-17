"""Paired end-to-end comparison of the pinned upstream min-p kernel and the
two-stage patch, inside one engine process.

Both arms go through the same Python dispatch wrapper, so the only difference
is the kernel implementation.  Rounds alternate ABBA / BAAB to cancel drift;
each round contributes one paired difference, and the reported interval is a
bootstrap over rounds (not over individual requests, which are not independent
within a round).

Token-id agreement between arms is recorded, never asserted: vLLM sampling is
seeded per request, but the observed batch-4 runs show that the sampled ids can
drift between rounds for reasons unrelated to min-p.  Byte-level equality of
the filtered logits is established separately by the verify mode.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import traceback


def bootstrap_ci(values, iterations=10000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(statistics.mean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def median_ci(values, seed=0):
    return {"n": len(values), "median": statistics.median(values),
            "ci95": list(bootstrap_ci(values, seed=seed))}


def compare_ids(first, other):
    if other == first:
        return {"equal": True}
    limit = min(len(first), len(other))
    index = next((i for i in range(limit) if first[i] != other[i]), limit)
    return {"equal": False, "first_difference_at_token": index,
            "tokens_differing": sum(1 for a, b in zip(first, other) if a != b)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch-module", required=True)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-p", type=float, default=0.1)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    batches = [int(x) for x in args.batches.split(",")]
    module_path = Path(args.patch_module).resolve()
    try:
        import torch
        from vllm import LLM, SamplingParams

        llm = LLM(model="/opt/cloud-inference-framework/models/Qwen3-0.6B",
                  dtype="float16", max_model_len=2048, max_num_seqs=8,
                  max_num_batched_tokens=2048, gpu_memory_utilization=0.65,
                  enable_prefix_caching=False, seed=0, disable_log_stats=False,
                  worker_extension_cls="project_worker.ProjectWorkerExtension")
        evidence = {"patch_module": str(module_path),
                    "patch_module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                    "torch": torch.__version__, "rounds": args.rounds,
                    "batches": batches, "prompt_len": args.prompt_len,
                    "max_tokens": args.max_tokens, "min_p": args.min_p,
                    "sampling": {"temperature": 0.8, "seed": 123},
                    "seeded_ids_asserted": False}
        (out / "config.json").write_text(json.dumps(evidence, indent=2))

        def install(arm):
            if arm == "base":
                llm.collective_rpc("minp_install", args=("base",))
            else:
                llm.collective_rpc("minp_install_file",
                                   args=("candidate", str(module_path)))

        def select(arm):
            # Both arms are already resident; switching is a flag flip, so the
            # measurement never pays for a module import.
            llm.collective_rpc("minp_select",
                               args=("base" if arm == "base" else "candidate",))

        def run(batch):
            prompts = [{"prompt_token_ids": [1000 + i] * args.prompt_len}
                       for i in range(batch)]
            params = [SamplingParams(temperature=0.8, min_p=args.min_p, seed=123 + i,
                                     max_tokens=args.max_tokens,
                                     min_tokens=args.max_tokens,
                                     ignore_eos=True, detokenize=False)
                      for i in range(batch)]
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            wall = time.perf_counter() - started
            intervals = []
            for output in outputs:
                metrics = output.metrics
                if metrics is None or getattr(metrics, "first_token_ts", None) is None:
                    continue
                intervals.append(1000 * (metrics.last_token_ts - metrics.first_token_ts)
                                 / (args.max_tokens - 1))
            ids = [list(output.outputs[0].token_ids) for output in outputs]
            assert len(ids) == batch
            assert all(len(tokens) == args.max_tokens for tokens in ids)
            return {"wall": wall, "intervals": intervals, "ids": ids}

        # Warmup: page cache, CUDA graphs and Triton autotune for both arms.
        for arm in ("base", "patched"):
            install(arm)
            for batch in batches:
                run(batch)

        # Dedicated equivalence phase: base, patched, patched, base.  Any
        # divergence here is reported with its token index, not hidden.
        equivalence = []
        for batch in batches:
            install("patched")
            reference = None
            records = []
            for arm in ("base", "patched", "patched", "base"):
                select(arm)
                result = run(batch)
                entry = {"batch": batch, "arm": arm, "ids": result["ids"]}
                if reference is None:
                    reference = result["ids"]
                entry["against_reference"] = compare_ids(reference, result["ids"])
                records.append(entry)
            all_equal = all(r["against_reference"]["equal"] for r in records)
            equivalence.append({"batch": batch, "all_four_calls_equal": all_equal,
                                "calls": records})
            print(json.dumps({"equivalence_batch": batch, "all_equal": all_equal}),
                  flush=True)
        (out / "equivalence.json").write_text(json.dumps(equivalence, indent=2))

        rows = []
        with (out / "timing.jsonl").open("w") as stream:
            for batch in batches:
                install("patched")
                for round_index in range(args.rounds):
                    order = "ABBA" if round_index % 2 == 0 else "BAAB"
                    for position, letter in enumerate(order):
                        arm = "base" if letter == "A" else "patched"
                        select(arm)
                        result = run(batch)
                        row = {"batch": batch, "round": round_index, "order": order,
                               "position": position, "arm": arm,
                               "wall_seconds": result["wall"],
                               "decode_ms": result["intervals"],
                               "output_ids": result["ids"]}
                        rows.append(row)
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        print(json.dumps({"batch": batch, "round": round_index,
                                          "arm": arm,
                                          "wall_ms": round(result["wall"] * 1000, 2)}),
                              flush=True)
                for arm in ("base", "patched"):
                    stats = llm.collective_rpc("minp_stats")
                    evidence[f"worker_stats_batch{batch}_{arm}"] = stats
                    if arm == "patched":
                        kernels = stats[0].get("kernels", {})
                        assert stats[0]["calls"] > 0, stats
                        assert kernels.get("_min_p_partial_max", 0) > 0, stats
                        assert kernels.get("_min_p_apply_threshold", 0) > 0, stats
                        assert kernels.get("_min_p_kernel", 0) == 0, stats

        summaries = []
        for batch in batches:
            reference = next(r["output_ids"] for r in rows
                             if r["batch"] == batch and r["arm"] == "base")
            agreement = {}
            for arm in ("base", "patched"):
                calls = [r for r in rows if r["batch"] == batch and r["arm"] == arm]
                same = [compare_ids(reference, r["output_ids"]) for r in calls]
                agreement[arm] = {
                    "calls": len(calls),
                    "identical_to_first_base": sum(1 for item in same if item["equal"]),
                    "first_divergence_token": [item.get("first_difference_at_token")
                                               for item in same if not item["equal"]][:8]}
            rounds = []
            for round_index in range(args.rounds):
                samples = [r for r in rows
                           if r["batch"] == batch and r["round"] == round_index]
                base = [r for r in samples if r["arm"] == "base"]
                patched = [r for r in samples if r["arm"] == "patched"]
                base_wall = statistics.mean(r["wall_seconds"] for r in base)
                patched_wall = statistics.mean(r["wall_seconds"] for r in patched)
                base_decode_values = [v for r in base for v in r["decode_ms"]]
                patched_decode_values = [v for r in patched for v in r["decode_ms"]]
                base_decode = statistics.mean(base_decode_values) if base_decode_values else None
                patched_decode = (statistics.mean(patched_decode_values)
                                  if patched_decode_values else None)
                rounds.append({"round": round_index,
                               "base_wall_ms": base_wall * 1000,
                               "patched_wall_ms": patched_wall * 1000,
                               "wall_pct": 100 * (patched_wall - base_wall) / base_wall,
                               "base_decode_ms": base_decode,
                               "patched_decode_ms": patched_decode,
                               "decode_pct": None if base_decode is None or patched_decode is None
                               else 100 * (patched_decode - base_decode) / base_decode})
            wall_pct = [r["wall_pct"] for r in rounds]
            decode_pct = [r["decode_pct"] for r in rounds if r["decode_pct"] is not None]
            all_base = [r["wall_seconds"] for r in rows
                        if r["batch"] == batch and r["arm"] == "base"]
            all_patched = [r["wall_seconds"] for r in rows
                           if r["batch"] == batch and r["arm"] == "patched"]
            summaries.append({
                "batch": batch,
                "rounds": args.rounds,
                "base_median_wall_ms": statistics.median(all_base) * 1000,
                "patched_median_wall_ms": statistics.median(all_patched) * 1000,
                "wall_pct_of_paired_rounds": median_ci(wall_pct, seed=batch),
                "decode_pct_of_paired_rounds": median_ci(decode_pct, seed=batch + 100)
                if decode_pct else None,
                "id_agreement": agreement,
                "rounds_detail": rounds,
            })
        evidence["summary"] = summaries
        (out / "summary.json").write_text(json.dumps(evidence, indent=2))
        (out / "status.json").write_text(json.dumps(
            {"state": "passed", "timing_rows": len(rows),
             "timing_sha256": hashlib.sha256((out / "timing.jsonl").read_bytes()).hexdigest()},
            indent=2))
        print(json.dumps(summaries, indent=2))
    except BaseException:
        (out / "status.json").write_text(json.dumps(
            {"state": "failed", "traceback": traceback.format_exc()}, indent=2))
        raise


if __name__ == "__main__":
    main()
