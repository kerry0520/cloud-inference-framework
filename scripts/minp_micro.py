import argparse
import json
from pathlib import Path
import shutil
import statistics
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        import triton.testing
        from vllm.v1.worker.gpu.sample.min_p import apply_min_p as original
        from kernels.min_p import apply_min_p as candidate
        shutil.copyfile(__import__("kernels.min_p", fromlist=["__file__"]).__file__,
                        out / "candidate.py")
        torch.manual_seed(17)
        cases = []

        def check(vocab, batch, dtype, kind="random"):
            storage = torch.full((batch, vocab+7), 777., device="cuda", dtype=dtype)
            values = torch.randn((batch, vocab), device="cuda", dtype=dtype)
            if kind == "special":
                values[0].fill_(float("-inf"))
                if batch > 1:
                    values[1, 0] = float("inf")
                    values[1, -1] = float("nan")
                if batch > 2:
                    values[2].zero_()
            storage[:, :vocab] = values
            mapping = torch.tensor([(i*3+2) % 11 for i in range(batch)],
                                   device="cuda", dtype=torch.int32)
            probs = torch.tensor([0., .01, .05, .1, .5, 1., .2, 0., .7, 1., .9],
                                 device="cuda", dtype=torch.float32)
            reference, actual = storage.clone(), storage.clone()
            original(reference[:, :vocab], mapping, probs)
            candidate(actual[:, :vocab], mapping, probs)
            torch.cuda.synchronize()
            equal = torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
            row = {"vocab": vocab, "batch": batch, "dtype": str(dtype),
                   "kind": kind, "byte_equal_including_padding": equal}
            cases.append(row)
            (out / "correctness.json").write_text(json.dumps(cases, indent=2))
            assert equal, row

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for vocab in (1, 31, 1023, 1024, 1025, 4095, 4096, 4097, 32000, 151936):
                for batch in (1, 2, 8):
                    check(vocab, batch, dtype)
            for vocab in (4097, 151936):
                check(vocab, 8, dtype, "special")
        print(f"Correctness passed: {len(cases)} cases", flush=True)
        timing = []
        with (out / "timing.jsonl").open("w") as stream:
            for vocab in (4096, 32000, 151936):
                for batch in (1, 2, 4, 8, 32):
                    x = torch.randn((batch, vocab), device="cuda", dtype=torch.float32)
                    mapping = torch.arange(batch, device="cuda", dtype=torch.int32)
                    probs = torch.full((batch,), .05, device="cuda")
                    for arm in ("base", "candidate", "candidate", "base"):
                        fn = original if arm == "base" else candidate
                        values = triton.testing.do_bench(
                            lambda: fn(x, mapping, probs), warmup=5, rep=20,
                            quantiles=[.5, .2, .8])
                        row = {"vocab": vocab, "batch": batch, "arm": arm,
                               "median_us": values[0]*1000,
                               "p20_us": values[1]*1000, "p80_us": values[2]*1000}
                        timing.append(row)
                        stream.write(json.dumps(row)+"\n")
                        stream.flush()
                    latest = timing[-4:]
                    base = statistics.median(r["median_us"] for r in latest if r["arm"] == "base")
                    cand = statistics.median(r["median_us"] for r in latest if r["arm"] == "candidate")
                    print(json.dumps({"vocab": vocab, "batch": batch,
                                      "base_us": base, "candidate_us": cand,
                                      "ratio": base/cand}), flush=True)
        (out / "status.json").write_text(json.dumps({"state": "passed",
                                                     "correctness_cases": len(cases),
                                                     "timing_rows": len(timing)}))
    except BaseException:
        (out / "status.json").write_text(json.dumps(
            {"state": "failed", "traceback": traceback.format_exc()}, indent=2))
        raise


if __name__ == "__main__":
    main()
