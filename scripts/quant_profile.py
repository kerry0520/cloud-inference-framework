"""Profile a decode step with fp16 and fp8 weights in one process, for a fair split.

`quant_decode.py` observed 24--27% lower mean decode intervals with FP8. The
old ~40% forecast counted lm_head as quantized; it is withdrawn. Two relevant
costs to inspect are:

  1. `lm_head` (20.4% of the step) is not quantised, so its 311 MB still stream in
     fp16 and cannot benefit;
  2. the fp8 path adds 112 activation-quantisation kernels per step, and the fp8
     GEMM kernel does not necessarily reach the same effective bandwidth.

To separate them without cross-session clock drift, both traces are captured in
one process, back to back, with the same profiler configuration.  Kernels are then
grouped by family (`attribute_decode.py --by-family`), never by role position,
because the fp8 step has a different kernel sequence.

This stage claims nothing about uninstrumented speed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

MODEL = str(Path(os.environ.get("CI_RUNTIME_DIR", "/opt/cloud-inference-framework")) / "models/Qwen3-0.6B")


def write(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def capture(arm: str, out: Path, args, torch, LLM, SamplingParams):
    kwargs = dict(model=MODEL, dtype="float16", max_model_len=2048,
                  max_num_seqs=8, max_num_batched_tokens=2048,
                  gpu_memory_utilization=args.gpu_fraction,
                  enable_prefix_caching=False, seed=0, disable_log_stats=False,
                  profiler_config={"profiler": "torch",
                                   "torch_profiler_dir": str(out / arm / "traces"),
                                   "torch_profiler_with_stack": False,
                                   "torch_profiler_dump_cuda_time_total": False,
                                   "delay_iterations": 4, "max_iterations": 12,
                                   "ignore_frontend": True})
    if arm != "fp16":
        kwargs["quantization"] = arm
    llm = LLM(**kwargs)
    applied = getattr(llm.llm_engine.model_config, "quantization", None)
    prompts = [{"prompt_token_ids": [1000 + i] * args.prompt_len}
               for i in range(args.batch)]
    params = [SamplingParams(temperature=0.0, seed=0,
                             max_tokens=args.max_tokens,
                             min_tokens=args.max_tokens,
                             ignore_eos=True, detokenize=False)
              for _ in range(args.batch)]
    llm.generate(prompts, params, use_tqdm=False)      # warmup, outside profiler
    llm.start_profile(profile_prefix=arm)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    llm.stop_profile()
    ids = [list(o.outputs[0].token_ids) for o in outputs]
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    traces = list((out / arm / "traces").glob("**/*trace.json.gz"))
    return {"arm": arm, "quantization_applied": str(applied),
            "output_token_ids": ids,
            "traces": [str(t) for t in traces],
            "trace_sha256": {t.name: hashlib.sha256(t.read_bytes()).hexdigest()
                             for t in traces}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="fp16,fp8")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--gpu-fraction", type=float, default=0.6)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    try:
        import torch
        from vllm import LLM, SamplingParams

        config = {"model": MODEL, "arms": arms, "batch": args.batch,
                  "prompt_len": args.prompt_len, "max_tokens": args.max_tokens,
                  "sampling": "greedy", "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0),
                  "note": ("both traces captured in one process, back to back; "
                           "kernel families only, no role positions")}
        write(out / "config.json", config)
        captured = []
        for arm in arms:
            entry = capture(arm, out, args, torch, LLM, SamplingParams)
            captured.append(entry)
            print(json.dumps({k: entry[k] for k in
                              ("arm", "quantization_applied", "traces")}), flush=True)
        write(out / "status.json", {"state": "passed", "captured": captured})
    except BaseException:
        write(out / "status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    sys.exit(main())
