"""Dump launch arguments of the post-graph tail kernels of a decode step.

The tail (final norm, lm_head, sampler) runs eagerly outside the CUDA graph
captured for the layer stack, so its shape is not inferable from the per-layer
kernel sequence. This script prints and records the grid/block/register
occupancy reported in the trace, which is what shows that the lm_head kernel
covers the whole vocabulary instead of a pruned slice.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

TAIL_START = 371  # first kernel after the 28 layer blocks, 0-based per step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="run directory holding traces/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=TAIL_START)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    trace_path = sorted((args.root / "traces").glob("**/*trace.json.gz"))[0]
    with gzip.open(trace_path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    windows = sorted((e for e in events if e.get("cat") == "gpu_user_annotation"),
                     key=lambda e: e["ts"])

    hist = Counter()
    aligned = []
    for i, start in enumerate(windows):
        end = windows[i + 1]["ts"] if i + 1 < len(windows) else start["ts"] + start["dur"]
        win = sorted((k for k in kernels if start["ts"] <= k["ts"] < end),
                     key=lambda k: k["ts"])
        hist[len(win)] += 1
        if len(win) == hist.most_common(1)[0][0]:
            aligned.append(win)
    if not aligned:
        raise SystemExit("no step with the modal kernel count")

    step = aligned[len(aligned) // 2]
    rows = []
    for index in range(args.start, len(step)):
        kernel = step[index]
        info = kernel.get("args", {})
        rows.append({
            "index": index,
            "name": kernel["name"],
            "dur_us": kernel["dur"],
            "grid": info.get("grid"),
            "block": info.get("block"),
            "registers_per_thread": info.get("registers per thread"),
            "occupancy": info.get("est. achieved occupancy %"),
            "graph_node_id": info.get("graph node id"),
            "shared_mem": info.get("shared memory"),
        })
    payload = {
        "trace": str(trace_path),
        "window_kernel_histogram": dict(hist),
        "aligned_steps": len(aligned),
        "step_index": aligned.index(step),
        "tail_start": args.start,
        "kernels": rows,
    }
    (args.out / "tail-args.json").write_text(json.dumps(payload, indent=2),
                                             encoding="utf-8")
    print(f"trace {trace_path.name} | windows {len(windows)} | "
          f"aligned {len(aligned)} | step {payload['step_index']}")
    for row in rows:
        print(f"{row['index']:3d} {row['name'][:58]:58s} "
              f"{row['dur_us']:8.2f}us grid={row['grid']} block={row['block']} "
              f"regs={row['registers_per_thread']} occ={row['occupancy']} "
              f"node={row['graph_node_id']}")


if __name__ == "__main__":
    main()
