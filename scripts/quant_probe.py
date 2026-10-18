"""Feasibility probe: can we halve weight bytes on this GPU and keep the same GEMM shapes?

Part of the fifth-stage follow-up (option A). The claim under test is that a decode
step costs about (weight bytes) / (achievable bandwidth). If we can run the *same*
shapes with 1-byte weights instead of 2-byte weights and the time drops about by half,
the byte-proportional model survives an intervention; if the time barely moves, the
model is wrong and the roofline fit was a coincidence.

This script only answers "is the machinery available and what does it cost":
  1. does this GPU (sm_120, RTX 5060 Laptop) run fp8 e4m3 GEMM through torch._scaled_mm?
  2. which kernel does the dispatcher actually launch (so we know it is not falling
     back to a dequant + fp16 matmul, which would read *more* bytes, not fewer)?
  3. how long do the five real decode shapes take in fp16 vs fp8, and what effective
     weight-read bandwidth does each imply?

Numerics are checked only to fp8 tolerance: this is a bandwidth probe, not a
correctness claim about an fp8 model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

# Same shapes as the traced batch-2 decode step (X[M,K] @ W[N,K]^T).
SHAPES = {
    "qkv_proj": (4096, 1024),
    "o_proj": (1024, 2048),
    "gate_up_proj": (6144, 1024),
    "down_proj": (1024, 3072),
    "lm_head": (151936, 1024),
}


def env_report() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_MiB": torch.cuda.get_device_properties(0).total_memory // (1 << 20),
        "has_scaled_mm": hasattr(torch, "_scaled_mm"),
        "has_float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
        "has_int_mm": hasattr(torch, "_int_mm"),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # pragma: no cover - diagnostics only
        info["triton"] = f"unavailable: {exc}"
    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

        info["vllm_quant_methods"] = sorted(QUANTIZATION_METHODS)
    except Exception as exc:
        info["vllm_quant_methods"] = f"unavailable: {exc}"
    return info


def quantize_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric quantisation to e4m3, returning (tensor, dequant scale)."""
    amax = x.abs().amax().clamp(min=1e-8)
    scale = (amax / 448.0).to(torch.float32)
    q = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def scaled_mm(a_q: torch.Tensor, b_q: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
              out_dtype=torch.float16) -> torch.Tensor:
    """Call torch._scaled_mm across the signature variants seen in 2.x releases."""
    attempts = (
        lambda: torch._scaled_mm(a_q, b_q, scale_a, scale_b, out_dtype=out_dtype),
        lambda: torch._scaled_mm(a_q, b_q, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype),
        lambda: torch._scaled_mm(a_q, b_q, scale_a, scale_b, bias=None, out_dtype=out_dtype),
    )
    last = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:  # pragma: no cover - version probing
            last = exc
    raise RuntimeError(f"_scaled_mm unavailable: {last}")


def time_us(fn, warmup: int = 20, iters: int = 100) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples), min(samples)


def kernel_names(fn) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for event in prof.key_averages():
        if event.device_time_total > 0 and event.key not in names:
            names.append(event.key)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", default="1,2,8", help="decode batch sizes to probe")
    parser.add_argument("--vocab-shapes", default="qkv_proj,o_proj,gate_up_proj,down_proj,lm_head",
                        help="comma list of shapes to include")
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = [int(v) for v in args.rows.split(",")]
    wanted = [s for s in args.vocab_shapes.split(",") if s in SHAPES]

    report: dict = {"env": env_report(), "shapes": {}, "kernels": {}, "errors": {}}
    print(f"torch {report['env']['torch']} cuda {report['env']['cuda']} "
          f"{report['env']['device']} sm_{report['env']['capability'][0]}{report['env']['capability'][1]}")
    print(f"scaled_mm={report['env']['has_scaled_mm']} fp8={report['env']['has_float8_e4m3fn']}")

    if not report["env"]["has_scaled_mm"]:
        report["errors"]["scaled_mm"] = "torch._scaled_mm missing"
        (out / "quant-probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    device = torch.device("cuda")
    for name in wanted:
        n, k = SHAPES[name]
        for m in rows:
            key = f"{name}_m{m}"
            x = torch.randn(m, k, device=device, dtype=torch.float16)
            w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.05
            ref = x @ w.t()
            try:
                w_q, s_w = quantize_e4m3(w)
                x_q, s_x = quantize_e4m3(x)
                y = scaled_mm(x_q, w_q.t(), s_x, s_w)
                finite = bool(torch.isfinite(y).all())
                rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean()
            except Exception as exc:
                report["errors"][key] = f"{type(exc).__name__}: {exc}"
                print(f"{key}: fp8 path failed: {type(exc).__name__}: {exc}")
                continue

            fp16_med, fp16_min = time_us(lambda: x @ w.t())
            fp8_med, fp8_min = time_us(lambda: scaled_mm(x_q, w_q.t(), s_x, s_w))
            weight_bytes = n * k * 2
            report["shapes"][key] = {
                "n": n, "k": k, "m": m,
                "fp8_finite": finite, "fp8_mean_rel_error": float(rel),
                "fp16_us_median": fp16_med, "fp16_us_min": fp16_min,
                "fp8_us_median": fp8_med, "fp8_us_min": fp8_min,
                "fp16_GB_s_median": weight_bytes / fp16_med / 1e3,
                "fp8_GB_s_median": weight_bytes / fp8_med / 1e3,
                "speedup_median": fp16_med / fp8_med,
            }
            if key not in report["kernels"]:
                report["kernels"][key] = {
                    "fp16": kernel_names(lambda: x @ w.t()),
                    "fp8": kernel_names(lambda: scaled_mm(x_q, w_q.t(), s_x, s_w)),
                }
            s = report["shapes"][key]
            print(f"{key:18s} fp16 {fp16_med:8.2f} us  fp8 {fp8_med:8.2f} us  "
                  f"speedup {s['speedup_median']:4.2f}x  rel_err {rel:.3f}  finite {finite}")

    (out / "quant-probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out / 'quant-probe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
