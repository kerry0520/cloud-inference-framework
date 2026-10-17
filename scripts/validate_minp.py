"""Recheck saved engine evidence and identify real kernel dispatch in traces."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


def main():
    import torch
    from kernels.min_p import apply_min_p_optimized
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p as original
    root = Path(sys.argv[1])
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == "passed"
    raw = (root / "timing.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == status["timing_sha256"]
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    assert len(rows) == status["timing_rows"] == 24
    for batch in (1, 2, 4):
        group = [r for r in rows if r["batch"] == batch]
        assert len(group) == 8
        assert Counter(r["arm"] for r in group) == {"base": 4, "candidate": 4}
        assert all(r["output_token_ids"] == group[0]["output_token_ids"] for r in group)
        assert all(r["worker_stats"][0]["calls"] > 0 for r in group)
    replays = []
    for path in sorted((root / "real-tensors").glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(data["reference"].view(torch.uint8), data["candidate"].view(torch.uint8))
        mapping, probs = data["mapping"].cuda(), data["min_p"].cuda()
        reference, candidate = data["before"].cuda(), data["before"].cuda()
        original(reference, mapping, probs)
        apply_min_p_optimized(candidate, mapping, probs)
        assert torch.equal(reference.cpu().view(torch.uint8), data["reference"].view(torch.uint8))
        assert torch.equal(candidate.cpu().view(torch.uint8), data["candidate"].view(torch.uint8))
        replays.append({"file": path.name, "values": candidate.numel(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    traces = []
    for path in sorted((root / "traces").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        names = Counter(e["name"] for e in kernels)
        base_count = sum(n for name, n in names.items() if name == "_min_p_kernel")
        max_count = sum(n for name, n in names.items() if name == "_min_p_partial_max")
        mask_count = sum(n for name, n in names.items() if name == "_min_p_apply_threshold")
        if max_count:
            arm = "candidate"
            assert max_count == mask_count == 12 and base_count == 0
        else:
            arm = "base"
            assert base_count == 12 and mask_count == 0
        total = sum(e["dur"] for e in kernels if e["name"] in (
            "_min_p_kernel", "_min_p_partial_max", "_min_p_apply_threshold"))
        traces.append({"file": path.name, "identified_arm": arm,
                       "base_kernel_calls": base_count, "max_calls": max_count,
                       "mask_calls": mask_count, "min_p_kernel_sum_us": total,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not status.get("timing_only"):
        assert len(replays) == 4
        assert Counter(t["identified_arm"] for t in traces) == {"base": 1, "candidate": 1}
    result = {"state": "passed", "timing_rows_verified": len(rows),
              "real_replays": replays, "trace_dispatch": traces}
    (root / "independent-check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
