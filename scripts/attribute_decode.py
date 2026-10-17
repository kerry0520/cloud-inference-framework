"""Attribute compiled-path decode kernels to roles from an existing trace.

Reads a torch-profiler trace produced by the pinned decode profile stage.  The
steady-state decode steps are CUDA-graph replays, so each kernel carries a
stable graph node id; alignment uses the modal step's kernel count (a partially
captured step is dropped, and reported).

The layer block is 13 kernels and is verified position by position:

    input_norm, qkv_proj, rotary, kv_cache_write, attn_prep, attention,
    attn_combine, o_prep, o_proj, post_norm, gate_up, silu_mul, down_proj

followed by a tail (final norm, lm_head, sampler).  Roles are assigned from the
verified position, never from the kernel name alone.  Weight bytes come from the
Qwen3-0.6B config (hidden 1024, intermediate 3072, vocab 151936, 16 q / 8 kv
heads, head_dim 128, 28 layers) at 2 bytes per element.

Makes no claim about uninstrumented speed.

Usage:
    python3 scripts/attribute_decode.py <trace-root> --out DIR [--dump]
"""

import argparse
import collections
import gzip
import json
from pathlib import Path
import statistics

HIDDEN = 1024
INTERMEDIATE = 3072
VOCAB = 151936
HEAD_DIM = 128
NUM_LAYERS = 28
ELEM_BYTES = 2
QKV_N = 16 * HEAD_DIM + 2 * 8 * HEAD_DIM  # 4096
O_K = 16 * HEAD_DIM                       # 2048

WEIGHT_BYTES = {
    "qkv_proj": QKV_N * HIDDEN * ELEM_BYTES,
    "o_proj": HIDDEN * O_K * ELEM_BYTES,
    "gate_up": 2 * INTERMEDIATE * HIDDEN * ELEM_BYTES,
    "down_proj": HIDDEN * INTERMEDIATE * ELEM_BYTES,
    "lm_head": VOCAB * HIDDEN * ELEM_BYTES,
}

LAYER_BLOCK = [
    "input_norm", "qkv_proj", "rotary", "kv_cache_write", "attn_prep",
    "attention", "attn_combine", "o_prep", "o_proj", "post_norm",
    "gate_up", "silu_mul", "down_proj",
]
EXPECTED_CLASS = {
    "input_norm": "rms_norm", "post_norm": "rms_norm", "final_norm": "rms_norm",
    "qkv_proj": "gemm_128x1", "o_proj": "gemm_128x2", "gate_up": "gemm_128x2",
    "down_proj": "gemm_128x2", "lm_head": "gemm_128x2",
    "rotary": "rotary", "kv_cache_write": "kv_cache_write",
    "attn_prep": "other", "attn_combine": "attention_combine",
    "o_prep": "other", "silu_mul": "silu_mul", "attention": "attention",
}


def load_events(root):
    paths = list((root / "traces").glob("**/*trace.json.gz"))
    assert len(paths) == 1, paths
    with gzip.open(paths[0], "rt") as stream:
        return json.load(stream)["traceEvents"]


def classify(name):
    low = name.lower()
    if "cutlass" in low and "gemm" in low:
        if "128x2" in low:
            return "gemm_128x2"
        if "128x1" in low:
            return "gemm_128x1"
        return "gemm_other"
    if "flash_fwd_splitkv_kernel" in name:
        return "attention"
    if "flash_fwd_splitkv_combine" in name:
        return "attention_combine"
    if "reshape_and_cache" in name:
        return "kv_cache_write"
    if "add_rms_norm" in name or "embedding_rms_norm" in name:
        return "rms_norm"
    if "silu" in low:
        return "silu_mul"
    if "red_fused_1" in name or "red_fused_3" in name:
        return "rotary"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump", action="store_true",
                        help="print every kernel of the modal step")
    args = parser.parse_args()

    events = load_events(args.root)
    kernels = [e for e in events if e.get("cat") == "kernel"]
    steps = sorted((e for e in events if e.get("cat") == "gpu_user_annotation"),
                   key=lambda e: e["ts"])
    assert len(steps) >= 3

    windows = []
    for i in range(1, len(steps)):
        end = (steps[i + 1]["ts"] if i + 1 < len(steps)
               else steps[i]["ts"] + steps[i]["dur"])
        windows.append((steps[i]["ts"], end))

    per_step = []
    for start, end in windows:
        win = [k for k in kernels if start <= k["ts"] < end]
        win.sort(key=lambda k: k["ts"])
        per_step.append(win)

    modal = collections.Counter(len(w) for w in per_step).most_common(1)[0][0]
    kept = [w for w in per_step if len(w) == modal]
    dropped = [len(w) for w in per_step if len(w) != modal]
    same_seq = all([classify(k["name"]) for k in w] ==
                   [classify(k["name"]) for k in kept[0]] for w in kept)
    assert same_seq, "kept steps do not share a kernel sequence"

    reference = kept[0]
    classes = [classify(k["name"]) for k in reference]
    first_qkv = classes.index("gemm_128x1")
    block_start = first_qkv - 1
    assert block_start >= 0

    # Verify the 28 layer blocks position by position.
    for layer in range(NUM_LAYERS):
        base = block_start + len(LAYER_BLOCK) * layer
        for offset, role in enumerate(LAYER_BLOCK):
            idx = base + offset
            assert idx < len(reference), (layer, offset)
            got = classes[idx]
            assert got == EXPECTED_CLASS[role], (
                f"layer {layer} offset {offset} role {role}: "
                f"class {got}, name {reference[idx]['name']}")

    tail_start = block_start + len(LAYER_BLOCK) * NUM_LAYERS
    tail = list(range(tail_start, len(reference)))
    tail_classes = [classes[i] for i in tail]
    assert tail_classes[0] == "rms_norm", tail_classes
    lm_head_idx = next(i for i in tail if classes[i] == "gemm_128x2")
    tail_roles = {}
    for i in tail:
        if i == tail_start:
            tail_roles[i] = "final_norm"
        elif i == lm_head_idx:
            tail_roles[i] = "lm_head"
        elif i < lm_head_idx:
            tail_roles[i] = "pre_head"
        else:
            tail_roles[i] = "sampler"

    role_positions = collections.defaultdict(list)
    for layer in range(NUM_LAYERS):
        base = block_start + len(LAYER_BLOCK) * layer
        for offset, role in enumerate(LAYER_BLOCK):
            role_positions[role].append((layer, base + offset))
    for idx, role in tail_roles.items():
        role_positions[role].append((None, idx))

    per_role = {}
    for role, entries in role_positions.items():
        durs = [w[idx]["dur"] for _, idx in entries for w in kept]
        per_role[role] = {
            "kernels_per_step": len(entries),
            "mean_us": statistics.mean(durs),
            "median_us": statistics.median(durs),
            "stdev_us": statistics.stdev(durs),
            "per_step_us": statistics.mean(durs) * len(entries),
        }
        if role in WEIGHT_BYTES:
            per_role[role]["weight_bytes"] = WEIGHT_BYTES[role]
            per_role[role]["effective_read_GB_s"] = (
                WEIGHT_BYTES[role] / per_role[role]["mean_us"] / 1e3)

    per_layer = []
    for layer in range(NUM_LAYERS):
        base = block_start + len(LAYER_BLOCK) * layer
        row = {"layer": layer + 1}
        for offset, role in enumerate(LAYER_BLOCK):
            if role in WEIGHT_BYTES:
                row[role + "_us"] = statistics.mean(
                    w[base + offset]["dur"] for w in kept)
        per_layer.append(row)

    step_totals = [sum(k["dur"] for k in w) for w in kept]
    step_graph = [sum(k["dur"] for k in w
                      if k["args"].get("graph node id")) for w in kept]
    step_eager = [sum(k["dur"] for k in w
                      if not k["args"].get("graph node id")) for w in kept]

    gemm_roles = [r for r in per_role if r in WEIGHT_BYTES]
    gemm_us = sum(per_role[r]["mean_us"] * per_role[r]["kernels_per_step"]
                  for r in gemm_roles)
    gemm_bytes = sum(WEIGHT_BYTES[r] * per_role[r]["kernels_per_step"]
                     for r in gemm_roles)

    result = {
        "scope": ("Steady-state batch-2 decode replays of one instrumented "
                  "trace only; not uninstrumented performance"),
        "trace_root": str(args.root),
        "annotation_steps": len(steps),
        "steady_windows": len(windows),
        "modal_kernels_per_step": modal,
        "kept_steps": len(kept),
        "dropped_step_kernel_counts": dropped,
        "sequence_identical_across_kept_steps": same_seq,
        "block_start_index": block_start,
        "tail_roles": {i: tail_roles[i] for i in tail},
        "unique_graph_nodes_in_step": len({k["args"].get("graph node id")
                                           for k in reference}),
        "median_step_kernel_sum_us": statistics.median(step_totals),
        "median_graph_replay_kernel_us": statistics.median(step_graph),
        "median_eager_tail_kernel_us": statistics.median(step_eager),
        "gemm_us_per_step": gemm_us,
        "gemm_bytes_per_step": gemm_bytes,
        "gemm_aggregate_GB_s": gemm_bytes / gemm_us / 1e3,
        "non_gemm_us_per_step": statistics.median(step_totals) - gemm_us,
        "per_role": per_role,
        "per_layer": per_layer,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "attribution.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({k: result[k] for k in
                      ("steady_windows", "modal_kernels_per_step", "kept_steps",
                       "dropped_step_kernel_counts", "block_start_index",
                       "median_step_kernel_sum_us",
                       "median_graph_replay_kernel_us",
                       "median_eager_tail_kernel_us",
                       "gemm_us_per_step", "gemm_aggregate_GB_s")}, indent=2))
    print(f"{'role':18s} {'n/step':>6s} {'mean_us':>9s} {'per_step_us':>12s} "
          f"{'MB':>8s} {'GB/s':>8s}")
    for role, v in sorted(per_role.items(), key=lambda kv: -kv[1]["per_step_us"]):
        mb = v.get("weight_bytes", 0) / 1e6
        bw = v.get("effective_read_GB_s", 0)
        print(f"{role:18s} {v['kernels_per_step']:6d} {v['mean_us']:9.2f} "
              f"{v['per_step_us']:12.2f} {mb:8.2f} {bw:8.1f}")
    print("per-layer GEMM mean us (qkv / o / gate_up / down):")
    for row in per_layer:
        print(f"  L{row['layer']:02d} {row['qkv_proj_us']:6.2f} "
              f"{row['o_proj_us']:6.2f} {row['gate_up_us']:6.2f} "
              f"{row['down_proj_us']:6.2f}")

    if args.dump:
        for idx, ref in enumerate(reference):
            print(f"{idx:4d} {classes[idx]:16s} "
                  f"{statistics.mean(w[idx]['dur'] for w in kept):8.2f}us "
                  f"node={ref['args'].get('graph node id')} "
                  f"{ref['name'][:70]}")


if __name__ == "__main__":
    main()
