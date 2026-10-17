"""Measure achievable streaming bandwidth and the real decode GEMM shapes.

The trace attribution says every layer GEMM and lm_head move their whole weight
once per decode step at 334-402 GB/s.  This script tests that claim from the
other side: what can this device stream, and do the same GEMMs cost the same
when the weights are cold?

Measurements:
  * copy / sum / read: device-to-device copy, a read-only reduction, and a
    Triton read-and-reduce kernel swept over block size -- the read ceiling.
  * gemm: each decode shape alone, replayed from a CUDA graph that walks R
    distinct weight slices so the loop cannot be served from L2.
  * step: the whole decode step's GEMM stream -- 28 layers x (qkv, o, gate_up,
    down) plus lm_head, each reading its own weight copy -- compared with the
    trace's 3,177.84 us per step.

Timing is CUDA events around each call, no L2 flush between iterations: every
loop here reads 96 MB to 1.19 GB of distinct weights, so nothing can be served
from the 32 MB L2, while triton.testing.do_bench's 256 MB flush writes leak into
a short measurement window (reported alongside for one case to show the size of
that artifact).

Clocks are sampled per phase (SM, memory, power): a slower standalone number is
only interpretable next to the clock state it was taken in.  Profiler runs
capture the CUDA kernel names so standalone GEMMs can be matched against the
compiled-path kernels.

Instrumented timings are not uninstrumented performance, and nothing here is a
claim about a full request.

Usage:
    python3 scripts/measure_bandwidth.py --out DIR
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import threading

import torch
import triton
import triton.language as tl

ELEM_BYTES = 2
VOCAB = 151936
SHAPES = [
    ("qkv_proj", 2, 4096, 1024),
    ("o_proj", 2, 1024, 2048),
    ("gate_up", 2, 6144, 1024),
    ("down_proj", 2, 1024, 3072),
    ("lm_head", 2, VOCAB, 1024),
]
LAYERS = 28
LAYER_BYTES = 4096 * 1024 * 2 + 1024 * 2048 * 2 + 6144 * 1024 * 2 \
    + 1024 * 3072 * 2
TARGET_LOOP_BYTES = 96 << 20
TRACE_GEMM_US_PER_STEP = 3177.8353


@triton.jit
def _read_sum_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0)
    tl.store(out_ptr + pid, tl.sum(vals.to(tl.float32)))


def bench_events_us(fn, iters=40, warmup=8):
    """(median, min) of per-call CUDA event timings, no cache flush.

    The minimum is the closest estimate of the kernel's cost at the highest SM
    clock reached during the loop: these M=2 kernels are latency sensitive, so
    a median taken while clocks are still ramping overstates their cost.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]
    return statistics.median(times), min(times)


def bench_flush_us(fn, warmup=200, rep=400):
    """do_bench median in us; its 256 MB flush writes are outside the events."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep,
                                   return_mode="median") * 1e3


def graph_capture(body):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        body()
        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
    torch.cuda.synchronize()
    return graph


class ClockSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.marks = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,"
                     "power.draw,temperature.gpu", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                if out:
                    self.samples.append(out)
            except Exception:
                pass
            self.stop.wait(0.2)

    def mark(self, phase):
        self.marks.append((phase, len(self.samples)))

    def summary(self, start=0, end=None):
        rows = self.samples[start:end]
        buckets = {"sm_MHz": [], "mem_MHz": [], "power_W": [], "temp_C": []}
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            for key, raw in zip(buckets, parts):
                try:
                    buckets[key].append(float(raw.split()[0]))
                except ValueError:
                    pass
        out = {}
        for key, values in buckets.items():
            out[key] = None if not values else {
                "min": min(values), "max": max(values),
                "median": statistics.median(values)}
        out["samples"] = len(rows)
        return out

    def per_phase(self):
        out = {}
        bounds = self.marks + [("end", len(self.samples))]
        for i in range(len(self.marks)):
            out[self.marks[i][0]] = self.summary(bounds[i][1], bounds[i + 1][1])
        return out


def kernel_names(fn):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    names = []
    for event in prof.key_averages():
        low = event.key.lower()
        if event.self_device_time_total > 0 and ("gemm" in low or "cutlass" in low):
            names.append(event.key)
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    idle = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,clocks.sm,clocks.mem",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "device": props.name,
        "total_memory_MiB": props.total_memory / 2**20,
        "l2_cache_MiB": props.L2_cache_size / 2**20,
        "idle_state": idle,
    }
    print(json.dumps(env, indent=2))
    result = {"environment": env, "peak_stream": {}, "gemm": {}, "step": {}}

    # Raise clocks and power state before measuring anything.
    burn_a = torch.randn((8192, 8192), dtype=torch.float16, device=device)
    burn_b = torch.randn((8192, 8192), dtype=torch.float16, device=device)
    for _ in range(200):
        torch.mm(burn_a, burn_b)
    torch.cuda.synchronize()
    print("burn done")
    del burn_a, burn_b
    torch.cuda.empty_cache()

    sampler = ClockSampler()
    sampler.start()

    n = 512 << 20
    src = torch.ones(n // ELEM_BYTES, dtype=torch.float16, device=device)
    dst = torch.empty_like(src)
    sampler.mark("copy")
    copy_med_us, copy_min_us = bench_events_us(lambda: dst.copy_(src))
    sampler.mark("sum")
    sum_med_us, sum_min_us = bench_events_us(lambda: torch.sum(src))
    sampler.mark("copy_flush")
    copy_flush_us = bench_flush_us(lambda: dst.copy_(src))
    result["peak_stream"]["copy"] = {
        "bytes_moved": 2 * n, "median_us": copy_med_us, "min_us": copy_min_us,
        "GB_s_median": 2 * n / (copy_med_us * 1e3),
        "GB_s_min": 2 * n / (copy_min_us * 1e3)}
    result["peak_stream"]["sum"] = {
        "bytes_moved": n, "median_us": sum_med_us, "min_us": sum_min_us,
        "GB_s_median": n / (sum_med_us * 1e3),
        "GB_s_min": n / (sum_min_us * 1e3)}
    result["peak_stream"]["copy_with_flush"] = {
        "bytes_moved": 2 * n, "do_bench_us": copy_flush_us,
        "GB_s": 2 * n / (copy_flush_us * 1e3)}
    print(f"copy events med {copy_med_us:9.2f}us min {copy_min_us:9.2f}us "
          f"{result['peak_stream']['copy']['GB_s_min']:7.1f} GB/s | "
          f"do_bench {copy_flush_us:9.2f}us "
          f"{result['peak_stream']['copy_with_flush']['GB_s']:7.1f} GB/s")
    print(f"sum  events med {sum_med_us:9.2f}us min {sum_min_us:9.2f}us "
          f"{result['peak_stream']['sum']['GB_s_min']:7.1f} GB/s")
    del src, dst
    torch.cuda.empty_cache()

    # Triton read ceiling: one program reduces a contiguous block.
    read_n = 256 << 20  # 512 MB of fp16
    big = torch.randn(read_n, dtype=torch.float16, device=device)
    sampler.mark("read")
    read_sweep = []
    for block in (4096, 8192, 16384):
        out = torch.empty(triton.cdiv(read_n, block), dtype=torch.float32,
                          device=device)
        grid = (triton.cdiv(read_n, block),)
        us, us_min = bench_events_us(
            lambda: _read_sum_kernel[grid](big, out, read_n, BLOCK=block))
        gbs = read_n * ELEM_BYTES / (us_min * 1e3)
        read_sweep.append({"block": block, "median_us": us, "min_us": us_min,
                           "GB_s": gbs})
        print(f"read BLOCK={block:6d} med {us:9.2f}us min {us_min:9.2f}us "
              f"{gbs:7.1f} GB/s")
    result["peak_stream"]["triton_read"] = {
        "bytes_moved": read_n * ELEM_BYTES, "sweep": read_sweep,
        "best": max(read_sweep, key=lambda row: row["GB_s"])}
    del big
    torch.cuda.empty_cache()

    for name, m, n_out, k in SHAPES:
        weight_bytes = n_out * k * ELEM_BYTES
        repeats = max(1, min(32, TARGET_LOOP_BYTES // weight_bytes))
        weights = torch.randn((repeats, n_out, k), dtype=torch.float16,
                              device=device)
        x = torch.randn((m, k), dtype=torch.float16, device=device)
        outs = [torch.empty((m, n_out), dtype=torch.float16, device=device)
                for _ in range(repeats)]

        def body(weights=weights, x=x, outs=outs, repeats=repeats):
            for i in range(repeats):
                torch.mm(x, weights[i].t(), out=outs[i])

        graph = graph_capture(body)
        sampler.mark(name)
        replay_us, replay_min_us = bench_events_us(graph.replay, iters=40,
                                                   warmup=8)
        per_us = replay_min_us / repeats
        nbytes = weight_bytes * repeats
        names = kernel_names(lambda: torch.mm(x, weights[0].t(), out=outs[0]))
        result["gemm"][name] = {
            "M": m, "N": n_out, "K": k,
            "weight_bytes": weight_bytes, "repeats": repeats,
            "graph_replay_us": replay_us,
            "graph_replay_min_us": replay_min_us,
            "per_gemm_us": per_us,
            "per_gemm_median_us": replay_us / repeats,
            "effective_GB_s": nbytes / (replay_min_us * 1e3),
            "kernel_names": names,
        }
        print(f"{name:10s} M={m} N={n_out:6d} K={k:5d} R={repeats:2d} "
              f"{per_us:8.2f}us "
              f"{result['gemm'][name]['effective_GB_s']:7.1f} GB/s")
        del weights, outs, graph, x
        torch.cuda.empty_cache()

    # The whole decode step's GEMM stream, with one weight copy per layer so no
    # step can be served from L2.
    x_qkv = torch.randn((2, 1024), dtype=torch.float16, device=device)
    x_o = torch.randn((2, 2048), dtype=torch.float16, device=device)
    x_gu = torch.randn((2, 1024), dtype=torch.float16, device=device)
    x_down = torch.randn((2, 3072), dtype=torch.float16, device=device)
    w_qkv = torch.randn((LAYERS, 4096, 1024), dtype=torch.float16, device=device)
    w_o = torch.randn((LAYERS, 1024, 2048), dtype=torch.float16, device=device)
    w_gu = torch.randn((LAYERS, 6144, 1024), dtype=torch.float16, device=device)
    w_down = torch.randn((LAYERS, 1024, 3072), dtype=torch.float16, device=device)
    w_lm = torch.randn((VOCAB, 1024), dtype=torch.float16, device=device)
    out_qkv = torch.empty((2, 4096), dtype=torch.float16, device=device)
    out_o = torch.empty((2, 1024), dtype=torch.float16, device=device)
    out_gu = torch.empty((2, 6144), dtype=torch.float16, device=device)
    out_down = torch.empty((2, 1024), dtype=torch.float16, device=device)
    out_lm = torch.empty((2, VOCAB), dtype=torch.float16, device=device)

    def step_body():
        for layer in range(LAYERS):
            torch.mm(x_qkv, w_qkv[layer].t(), out=out_qkv)
            torch.mm(x_o, w_o[layer].t(), out=out_o)
            torch.mm(x_gu, w_gu[layer].t(), out=out_gu)
            torch.mm(x_down, w_down[layer].t(), out=out_down)
        torch.mm(x_qkv, w_lm.t(), out=out_lm)

    step_graph = graph_capture(step_body)
    sampler.mark("step")
    step_us, step_min_us = bench_events_us(step_graph.replay, iters=40,
                                           warmup=8)
    sampler.mark("step_flush")
    step_flush_us = bench_flush_us(step_graph.replay, warmup=300, rep=1200)
    step_bytes = (LAYER_BYTES * LAYERS + VOCAB * 1024 * ELEM_BYTES)
    result["step"] = {
        "kernels": LAYERS * 4 + 1,
        "weight_bytes": step_bytes,
        "median_us": step_us,
        "min_us": step_min_us,
        "do_bench_us": step_flush_us,
        "effective_GB_s": step_bytes / (step_min_us * 1e3),
        "trace_gemm_us_per_step": TRACE_GEMM_US_PER_STEP,
        "ratio_vs_trace": step_min_us / TRACE_GEMM_US_PER_STEP,
        "ratio_vs_trace_median": step_us / TRACE_GEMM_US_PER_STEP,
    }
    print(f"step      113 GEMMs med {step_us:9.2f}us min {step_min_us:9.2f}us "
          f"{result['step']['effective_GB_s']:7.1f} GB/s | "
          f"do_bench {step_flush_us:9.2f}us "
          f"(trace {TRACE_GEMM_US_PER_STEP:.2f}us, "
          f"ratio {result['step']['ratio_vs_trace']:.3f})")

    sampler.stop.set()
    sampler.join(timeout=5)
    result["clocks"] = {"overall": sampler.summary(),
                        "per_phase": sampler.per_phase()}
    print(json.dumps(result["clocks"]["per_phase"], indent=2))

    (args.out / "bandwidth.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
