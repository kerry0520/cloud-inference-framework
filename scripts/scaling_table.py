"""Turn an abba-scale summary.json into the batch-scaling table used in the docs.

The engine writes one paired difference per ABBA round, so the percentage here
is a median over rounds with a bootstrap interval, never over requests.
"""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    rounds = summary["rounds"]
    steps_per_run = summary["max_tokens"] - 1
    rows = []
    for entry in summary["summary"]:
        batch = entry["batch"]
        stats = summary[f"worker_stats_batch{batch}_patched"][0]
        calls = sum(stats["shapes"].values())
        calls_per_step = calls / (rounds * 4 * steps_per_run)
        base_step = statistics.median(r["base_decode_ms"] for r in entry["rounds_detail"])
        patched_step = statistics.median(
            r["patched_decode_ms"] for r in entry["rounds_detail"])
        # Paired per-round difference, matching the percentage above.
        saving_us = statistics.median(
            (r["base_decode_ms"] - r["patched_decode_ms"]) * 1000
            for r in entry["rounds_detail"])
        routing = summary[f"parallel_path_batch{batch}"]
        rows.append({
            "batch": batch,
            "base_decode_ms": base_step,
            "patched_decode_ms": patched_step,
            "saving_us_per_step": saving_us,
            "decode_pct": entry["decode_pct_of_paired_rounds"],
            "wall_pct": entry["wall_pct_of_paired_rounds"],
            "minp_calls_per_step": calls_per_step,
            "parallel_launches": routing["parallel_launches"],
            "upstream_launches": routing["upstream_launches"],
            "used_parallel_path": routing["rows_above_window"] is False,
        })
    table = ["| batch | decode ms/step | 每步省 µs | decode Δ% (median, 95% CI) "
             "| wall Δ% | min-p 调用/步 | 实际内核 |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        pct = row["decode_pct"]
        ci = f"{pct['median']:+.2f} ({pct['ci95'][0]:+.2f}…{pct['ci95'][1]:+.2f})"
        kernel = "parallel" if row["used_parallel_path"] else "upstream fallback"
        wall = row["wall_pct"]
        table.append(
            f"| {row['batch']} | {row['base_decode_ms']:.2f} → "
            f"{row['patched_decode_ms']:.2f} | {row['saving_us_per_step']:+.0f} | "
            f"{ci} | {wall['median']:+.2f} | {row['minp_calls_per_step']:.2f} | {kernel} |")
    args.out.write_text(json.dumps({
        "results": str(args.results),
        "rounds": rounds,
        "max_tokens": summary["max_tokens"],
        "prompt_len": summary["prompt_len"],
        "max_num_seqs": summary.get("max_num_seqs"),
        "rows": rows,
        "table": table,
    }, indent=2), encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
