"""Compare the upstream min-p kernel with two candidate variants.

Variants
  upstream  pinned vLLM file (per-tile cross-lane reduction, BLOCK_SIZE 1024)
  knob      only launch parameters change (BLOCK_SIZE 16384, num_warps 16)
  pr58907   upstream PR #58907 at commit 95e9d21: per-lane maxima reduced once,
            tile 8192 / 16 warps for vocab > 16384 and batch <= 32, else 1024 / 4

Everything is measured on identical tensors with two protocols:

  interleaved  variants rotate in a shuffled round-robin, each measurement
               launches the kernel 100 times back to back and divides, so the
               number is steady-state throughput with launches queued
  cudagraph    Triton's do_bench_cudagraph (median), amortized time per call from
               a graph containing many unrolled calls, with warm-cache reuse

Neither protocol measures an isolated cold-cache launch. Every variant
is checked to produce byte-identical output before it is timed.
"""

import argparse
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton

REPO = Path(__file__).resolve().parent.parent

KERNEL_PATH = "vllm/v1/worker/gpu/sample/min_p.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def build_pr_module(upstream_path, diff_path, work_dir):
    """Apply the min_p.py hunk of a pull request diff to a copy of the file."""
    text = Path(diff_path).read_text(encoding="utf-8")
    parts = text.split("diff --git ")
    keep = [p for p in parts[1:] if KERNEL_PATH in p.split("\n")[0]]
    if not keep:
        raise SystemExit(f"no {KERNEL_PATH} hunk in {diff_path}")
    subset = Path(work_dir) / "min_p_pr.diff"
    subset.write_text("diff --git " + keep[0], encoding="utf-8")

    target = Path(work_dir) / KERNEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(Path(upstream_path).read_text(encoding="utf-8"), encoding="utf-8")

    subprocess.run(["git", "init", "-q", str(work_dir)], check=True)
    subprocess.run(
        ["git", "apply", str(subset)], cwd=str(work_dir), check=True
    )
    applied = target.read_text(encoding="utf-8")
    if "tl.full((BLOCK_SIZE,)" not in applied:
        raise SystemExit("PR diff applied but the expected kernel change is missing")
    return load_module(target, "mrv2_min_p_pr58907")


def timed_interleaved(calls, rounds=12, iters=100):
    """Rotate variants and keep the minimum of per-round averages."""
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


def timed_cudagraph(calls):
    out = {}
    bench = getattr(triton.testing, "do_bench_cudagraph", None)
    for name, fn in calls:
        if bench is not None:
            out[name] = float(bench(fn)) * 1e3
        else:
            out[name] = float(triton.testing.do_bench(fn, quantiles=[0.5])[0]) * 1e3
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--upstream", required=True, help="pinned min_p.py")
    parser.add_argument("--knob", required=True, help="patches/mrv2_min_p_knob.py")
    parser.add_argument("--pr-diff", required=True, help="pull request diff file")
    parser.add_argument("--vocabs", default="32000,151936")
    parser.add_argument("--batches", default="1,2,4,8,32")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "triton:", triton.__version__, "dtype:", args.dtype)

    with tempfile.TemporaryDirectory() as tmp:
        modules = {
            "upstream": load_module(args.upstream, "mrv2_min_p_upstream"),
            "knob": load_module(args.knob, "mrv2_min_p_knob"),
            "pr58907": build_pr_module(args.upstream, args.pr_diff, tmp),
        }

        rows = []
        for vocab in [int(v) for v in args.vocabs.split(",")]:
            for batch in [int(b) for b in args.batches.split(",")]:
                torch.manual_seed(0)
                base = torch.randn(batch, vocab, device="cuda", dtype=dtype) * 2
                mapping = torch.arange(batch, device="cuda", dtype=torch.int64)
                min_p = torch.full((batch,), 0.1, device="cuda", dtype=torch.float32)

                reference = base.clone()
                modules["upstream"].apply_min_p(reference, mapping, min_p)

                buffers = {}
                calls = []
                for name, module in modules.items():
                    work = base.clone()
                    module.apply_min_p(work, mapping, min_p)
                    if not torch.equal(
                        work.view(torch.uint8), reference.view(torch.uint8)
                    ):
                        raise SystemExit(f"{name} changed bytes at vocab={vocab} batch={batch}")
                    buffers[name] = work
                    calls.append(
                        (
                            name,
                            lambda w=work, n=name: modules[n].apply_min_p(
                                w, mapping, min_p
                            ),
                        )
                    )

                interleaved = timed_interleaved(calls)
                graph = timed_cudagraph(calls)
                for name in modules:
                    rows.append(
                        {
                            "vocab": vocab,
                            "batch": batch,
                            "dtype": args.dtype,
                            "variant": name,
                            "interleaved_us": round(interleaved[name], 3),
                            "cudagraph_us": round(graph[name], 3),
                            "speedup_interleaved": round(
                                interleaved["upstream"] / interleaved[name], 3
                            ),
                            "speedup_cudagraph": round(
                                graph["upstream"] / graph[name], 3
                            ),
                            "byte_identical": True,
                        }
                    )
                print(
                    f"vocab={vocab} batch={batch} "
                    f"upstream {interleaved['upstream']:.1f}/{graph['upstream']:.1f} us "
                    f"knob {interleaved['knob']:.1f}/{graph['knob']:.1f} us "
                    f"pr58907 {interleaved['pr58907']:.1f}/{graph['pr58907']:.1f} us "
                    f"(interleaved/cudagraph)"
                )

    (out / "compare.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("wrote", out / "compare.json")


if __name__ == "__main__":
    main()
