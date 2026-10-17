"""Fixed-token offline baseline; wall time includes prefill and engine overhead."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="/opt/cloud-inference-framework/models/Qwen3-0.6B")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "runner.py")
    write_json(out / "status.json", {"state": "running", "pid": os.getpid()})
    try:
        import torch
        from vllm import LLM, SamplingParams

        packages = {d.metadata["Name"]: d.version
                    for d in importlib.metadata.distributions()}
        write_json(out / "environment.json", {
            "args": vars(args), "packages": packages,
            "python": platform.python_version(), "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch_cuda": torch.version.cuda,
            "environment_overrides": {k: os.environ[k] for k in (
                "VLLM_WSL2_ENABLE_PIN_MEMORY", "VLLM_GPU_SYNC_CHECK",
                "VLLM_USE_V2_MODEL_RUNNER", "VLLM_WORKER_MULTIPROC_METHOD",
                "CUDA_VISIBLE_DEVICES", "MAX_JOBS", "TORCHINDUCTOR_COMPILE_THREADS"
            ) if k in os.environ},
            "source_commit": subprocess.check_output(
                ["git", "-C", "/opt/cloud-inference-framework/vllm-src",
                 "rev-parse", "HEAD"], text=True).strip(),
            "vllm_import": __import__("vllm").__file__,
        })
        config = dict(
            model=args.model, dtype="float16", max_model_len=2048,
            max_num_seqs=8, max_num_batched_tokens=2048,
            gpu_memory_utilization=0.65, enable_prefix_caching=False,
            seed=0, enforce_eager=args.eager, disable_log_stats=False,
        )
        if args.profile:
            config["profiler_config"] = {
                "profiler": "torch", "torch_profiler_dir": str(out / "traces"),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": False,
                "torch_profiler_dump_cuda_time_total": False,
                "delay_iterations": 4, "max_iterations": 12,
                "ignore_frontend": True,
            }
        started = time.perf_counter()
        llm = LLM(**config)
        write_json(out / "requested-config.json", config)
        write_json(out / "effective-config.txt.json", {
            "config": str(llm.llm_engine.vllm_config),
            "initialization_seconds": time.perf_counter() - started,
        })
        params = SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                min_tokens=args.output_tokens, ignore_eos=True,
                                detokenize=False)
        batches = [int(b) for b in args.batches.split(",")]
        all_rows = []
        with (out / "requests.jsonl").open("w", encoding="utf-8") as records:
            for batch in batches:
                # Valid fixed token IDs; this is a synthetic latency workload.
                prompts = [{"prompt_token_ids": [1000 + i] * args.input_tokens}
                           for i in range(batch)]
                reference = None
                for rep in range(-1, args.repeats):
                    profiling = args.profile and rep == 0
                    if profiling:
                        llm.start_profile()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
                    if profiling:
                        llm.stop_profile()
                    assert len(outputs) == batch
                    tokens = [list(item.outputs[0].token_ids) for item in outputs]
                    assert all(item.finished for item in outputs)
                    assert all(len(t) == args.output_tokens for t in tokens)
                    if reference is None:
                        reference = tokens
                    else:
                        assert tokens == reference, "Same-shape greedy run changed tokens"
                    row = {
                        "batch": batch, "repeat": rep, "warmup": rep < 0,
                        "instrumented": args.profile,
                        "input_tokens_per_request": args.input_tokens,
                        "output_tokens_per_request": args.output_tokens,
                        "wall_seconds": wall,
                        "output_tokens_per_second": batch * args.output_tokens / wall,
                        "output_token_ids": tokens,
                        "finish_reasons": [x.outputs[0].finish_reason for x in outputs],
                        "metrics": [dataclasses.asdict(x.metrics)
                                    if dataclasses.is_dataclass(x.metrics)
                                    else str(x.metrics) for x in outputs],
                    }
                    row["mean_decode_intervals_ms"] = [
                        1000 * (x.metrics.last_token_ts - x.metrics.first_token_ts)
                        / (len(x.outputs[0].token_ids) - 1)
                        if x.metrics is not None and x.metrics.first_token_ts > 0
                        and len(x.outputs[0].token_ids) > 1 else None
                        for x in outputs
                    ]
                    records.write(json.dumps(row, default=str) + "\n")
                    records.flush()
                    all_rows.append(row)
                    print(json.dumps({k: row[k] for k in (
                        "batch", "repeat", "wall_seconds", "output_tokens_per_second"
                    )}), flush=True)
        digest = hashlib.sha256((out / "requests.jsonl").read_bytes()).hexdigest()
        write_json(out / "status.json", {"state": "passed", "rows": len(all_rows),
                                          "requests_sha256": digest})
    except BaseException:
        write_json(out / "status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
