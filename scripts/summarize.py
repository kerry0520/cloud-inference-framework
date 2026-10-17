"""Validate saved rows and describe traces without treating profiled time as speed."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    root = Path(args.run)
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "passed", status
    raw = (root / "requests.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == status["requests_sha256"]
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    assert len(rows) == status["rows"] > 0
    groups = defaultdict(list)
    for row in rows:
        assert len(row["output_token_ids"]) == row["batch"]
        expected = row.get("expected_output_lengths",
                           [row["output_tokens_per_request"]] * row["batch"])
        assert [len(ids) for ids in row["output_token_ids"]] == expected
        assert row["wall_seconds"] > 0
        if not row["warmup"]:
            groups[row["batch"]].append(row)
    summary = {"validated_rows": len(rows), "groups": [], "traces": []}
    for batch, measured in sorted(groups.items()):
        intervals = [v for row in measured
                     for v in row.get("mean_decode_intervals_ms", []) if v is not None]
        summary["groups"].append({
            "batch": batch, "measured_batches": len(measured),
            "profiled": measured[0]["instrumented"],
            "median_batch_wall_ms": statistics.median(
                row["wall_seconds"] * 1000 for row in measured),
            "median_aggregate_output_tokens_per_second": statistics.median(
                row["output_tokens_per_second"] for row in measured),
            "median_request_mean_decode_interval_ms": (
                statistics.median(intervals) if intervals else None),
        })
    for path in sorted((root / "traces").glob("**/*trace.json*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            trace = json.load(stream)
        events = trace["traceEvents"]
        categories = Counter(e.get("cat", "") for e in events)
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        assert kernels, f"No GPU kernel records in {path}"
        names = defaultdict(lambda: [0, 0.0])
        for event in kernels:
            names[event["name"]][0] += 1
            names[event["name"]][1] += event["dur"]
        annotations = [e for e in events if e.get("cat") == "user_annotation"
                       and e.get("ph") == "X"]
        summary["traces"].append({
            "file": str(path), "categories": dict(categories),
            "gpu_kernel_events": len(kernels),
            "annotations": [{"name": e["name"], "duration_us": e["dur"]}
                            for e in annotations[:40]],
            "top_kernels_by_sum_duration_us": sorted(
                [{"name": name, "count": v[0], "sum_duration_us": v[1]}
                 for name, v in names.items()],
                key=lambda x: x["sum_duration_us"], reverse=True)[:12],
        })
    if any(row["instrumented"] for row in rows):
        assert summary["traces"], "Profile run did not produce valid GPU traces"
    target = root / "summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
