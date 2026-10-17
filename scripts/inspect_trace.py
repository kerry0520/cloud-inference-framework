"""Describe the observed decode trace; no uninstrumented speed claims."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
paths = list((root / "traces").glob("**/*trace.json.gz"))
assert len(paths) == 1
with gzip.open(paths[0], "rt") as stream:
    events = json.load(stream)["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
cpu_steps = [e for e in events if e.get("cat") == "user_annotation"
             and e.get("ph") == "X"]
gpu_steps = [e for e in events if e.get("cat") == "gpu_user_annotation"
             and e.get("ph") == "X"]
cpu_steps.sort(key=lambda e: e["ts"])
gpu_steps.sort(key=lambda e: e["ts"])
total = sum(e["dur"] for e in kernels)
gemm = sum(e["dur"] for e in kernels if "gemm" in e["name"].lower())
runtime_names = Counter(e["name"] for e in events
                        if e.get("cat") == "cuda_runtime")
steps = [{"name": e["name"], "duration_us": e["dur"], "ts": e["ts"]}
         for e in gpu_steps]
window_start = gpu_steps[1]["ts"]
window_end = gpu_steps[-1]["ts"] + gpu_steps[-1]["dur"]
intervals = sorted((max(e["ts"], window_start),
                    min(e["ts"] + e["dur"], window_end)) for e in kernels
                   if e["ts"] < window_end and e["ts"] + e["dur"] > window_start)
merged = []
for start, end in intervals:
    if merged and start <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], end)
    else:
        merged.append([start, end])
busy = sum(end - start for start, end in merged)
result = {
    "scope": "Instrumented decode trace only",
    "cpu_annotations": len(cpu_steps), "gpu_annotations": len(gpu_steps),
    "gpu_steps": steps,
    "median_cpu_annotation_us_excluding_first": statistics.median(
        e["dur"] for e in cpu_steps[1:]),
    "median_gpu_annotation_us_excluding_first": statistics.median(
        e["dur"] for e in gpu_steps[1:]) if len(gpu_steps) > 1 else None,
    "kernel_sum_us": total, "gemm_kernel_sum_us": gemm,
    "gemm_fraction_of_kernel_sum": gemm / total,
    "stable_observed_window_us": window_end - window_start,
    "kernel_union_in_window_us": busy,
    "kernel_union_fraction_in_window": busy / (window_end - window_start),
    "median_gpu_annotation_start_interval_us": statistics.median(
        b["ts"] - a["ts"] for a, b in zip(gpu_steps[1:], gpu_steps[2:])),
    "runtime_api_counts": dict(runtime_names),
    "step_name_counts": dict(Counter(e["name"] for e in cpu_steps)),
}
(root / "trace-analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
