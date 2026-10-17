"""One-parameter roofline check for the batch-2 decode step.

Question: does "time = weight bytes / effective bandwidth" reproduce every
weight-streaming role of the traced step, including the one that was never
used to fit the constant?

The constant is fitted by leave-one-out: for each role the remaining roles
(on their own) determine the bandwidth, and the held-out role is predicted.
"""

import argparse
import json
from pathlib import Path

# Roles whose kernels stream a whole weight matrix exactly once per step.
STREAMING = ("qkv_proj", "o_proj", "gate_up", "down_proj", "lm_head")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--attribution", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    attribution = json.loads(Path(args.attribution).read_text())
    per_role = attribution["per_role"]

    rows = []
    for role in STREAMING:
        entry = per_role[role]
        steps = entry["kernels_per_step"]
        step_bytes = entry["weight_bytes"] * steps
        rows.append({"role": role, "kernels_per_step": steps,
                     "weight_bytes": entry["weight_bytes"],
                     "step_bytes": step_bytes,
                     "measured_us": entry["per_step_us"],
                     "implied_GB_s": step_bytes / entry["per_step_us"] / 1e3})

    for row in rows:
        others = [r for r in rows if r["role"] != row["role"]]
        other_bytes = sum(r["step_bytes"] for r in others)
        other_us = sum(r["measured_us"] for r in others)
        bandwidth = other_bytes / other_us / 1e3
        predicted = row["step_bytes"] / (bandwidth * 1e3)
        row["fit_without_self_GB_s"] = bandwidth
        row["predicted_us"] = predicted
        row["residual_pct"] = 100 * (predicted - row["measured_us"]) / row["measured_us"]

    total_bytes = sum(r["step_bytes"] for r in rows)
    total_us = sum(r["measured_us"] for r in rows)
    fitted = total_bytes / total_us / 1e3
    residuals = [r["residual_pct"] for r in rows]
    result = {
        "scope": "batch-2 instrumented trace, steady-state decode replay; "
                 "kernel times include profiler overhead",
        "attribution": str(Path(args.attribution).resolve()),
        "measured_gemm_us_per_step": attribution["gemm_us_per_step"],
        "measured_step_kernel_sum_us": attribution["median_step_kernel_sum_us"],
        "streaming_bytes_per_step": total_bytes,
        "fitted_effective_GB_s": fitted,
        "fitted_effective_GB_s_note": "per_step_bytes / sum(per_role per_step_us)",
        "max_abs_residual_pct": max(abs(r) for r in residuals),
        "roles": rows,
        "non_streaming_us_per_step": attribution["gemm_us_per_step"] - total_us
        if total_us < attribution["gemm_us_per_step"] else 0.0,
    }
    (out / "roofline.json").write_text(json.dumps(result, indent=2))
    print(f"{'role':10s} {'bytes/step':>12s} {'measured_us':>12s} "
          f"{'pred_us':>9s} {'residual':>9s} {'fit_B':>8s}")
    for row in rows:
        print(f"{row['role']:10s} {row['step_bytes']:12d} {row['measured_us']:12.2f} "
              f"{row['predicted_us']:9.2f} {row['residual_pct']:+8.1f}% "
              f"{row['fit_without_self_GB_s']:7.1f}")
    print(f"fitted effective bandwidth (all five): {fitted:.2f} GB/s")
    print(f"max |leave-one-out residual|: {result['max_abs_residual_pct']:.1f}%")


if __name__ == "__main__":
    main()
