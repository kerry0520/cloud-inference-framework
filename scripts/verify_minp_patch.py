"""Verify the pinned-file patch for the MRV2 min-p parallel path.

Steps:
1. apply patches/0001-mrv2-min-p-two-stage.patch to a pristine copy of the
   pinned upstream file and require the result to equal patches/mrv2_min_p.py
   byte for byte;
2. require the installed wheel copy of the module to match the pinned source;
3. run the same 96-case acceptance matrix used for the exploratory candidate,
   recording for every case which Triton launcher actually ran;
4. probe the dispatch boundary (inclusive edges, column stride, empty batch)
   and the predicate's dtype guard;
5. time upstream vs patched through the patched entry point and check that the
   fast window gains and the fallback window does not regress.

Everything observed is written to patch-verification.json.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

PINNED_SOURCE = (
    "/opt/cloud-inference-framework/vllm-src/vllm/v1/worker/gpu/sample/min_p.py"
)
KERNELS = ("_min_p_kernel", "_min_p_partial_max", "_min_p_apply_threshold")
FAST_WINDOW = {"max_rows": 8, "min_vocab": 32000}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class LaunchCounter:
    """Count launch requests for one Triton kernel without altering it."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = 0

    def __getitem__(self, grid):
        self.calls += 1
        return self.kernel[grid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--patched", required=True)
    parser.add_argument("--source", default=PINNED_SOURCE)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    patch_path = str(Path(args.patch).resolve())
    result = {"patch": args.patch, "patched_source": args.patched,
              "pinned_source": args.source, "fast_window": FAST_WINDOW}
    try:
        import torch
        import triton.testing
        import vllm
        import vllm.v1.worker.gpu.sample.min_p as installed

        result["versions"] = {"torch": torch.__version__,
                              "vllm": vllm.__version__}
        # 1. apply the patch to a pristine copy of the pinned file
        apply_dir = out / "apply"
        target_dir = apply_dir / "vllm/v1/worker/gpu/sample"
        target_dir.mkdir(parents=True)
        target = target_dir / "min_p.py"
        shutil.copyfile(args.source, target)
        stat = subprocess.run(["git", "apply", "--stat", "-p1", patch_path],
                              cwd=apply_dir, check=True, capture_output=True,
                              text=True)
        subprocess.run(["git", "apply", "--check", "-p1", patch_path],
                       cwd=apply_dir, check=True, capture_output=True,
                       text=True)
        subprocess.run(["git", "apply", "-p1", patch_path], cwd=apply_dir,
                       check=True, capture_output=True, text=True)
        hashes = {"pinned": sha256(args.source),
                  "patched_expected": sha256(args.patched),
                  "applied": sha256(target)}
        assert hashes["applied"] == hashes["patched_expected"], (
            "applied patch differs from patches/mrv2_min_p.py", hashes)
        result["apply"] = {"stat": stat.stdout.strip(), "sha256": hashes,
                           "applied_equals_patched_source": True}

        # 2. the wheel ships the same file as the pinned checkout
        result["installed_matches_pinned"] = (
            sha256(installed.__file__) == hashes["pinned"])
        assert result["installed_matches_pinned"], (
            "installed module differs from pinned source", installed.__file__)

        upstream = installed.apply_min_p
        patched = load_module("mrv2_min_p_patched", target)
        counters = {name: LaunchCounter(getattr(patched, name))
                    for name in KERNELS}

        def dispatch(fn, logits, mapping, probs):
            for counter in counters.values():
                counter.calls = 0
            fn(logits, mapping, probs)
            torch.cuda.synchronize()
            return {name: counter.calls for name, counter in counters.items()}

        def in_fast_window(vocab, batch):
            return (1 <= batch <= FAST_WINDOW["max_rows"]
                    and vocab >= FAST_WINDOW["min_vocab"])

        # 3. acceptance matrix: same 96 cases as the exploratory candidate
        torch.manual_seed(17)
        cases = []
        for name, counter in counters.items():
            setattr(patched, name, counter)

        def check(vocab, batch, dtype, kind="random"):
            storage = torch.full((batch, vocab + 7), 777., device="cuda",
                                 dtype=dtype)
            values = torch.randn((batch, vocab), device="cuda", dtype=dtype)
            if kind == "special":
                values[0].fill_(float("-inf"))
                if batch > 1:
                    values[1, 0] = float("inf")
                    values[1, -1] = float("nan")
                if batch > 2:
                    values[2].zero_()
            storage[:, :vocab] = values
            mapping = torch.tensor([(i * 3 + 2) % 11 for i in range(batch)],
                                   device="cuda", dtype=torch.int32)
            probs = torch.tensor(
                [0., .01, .05, .1, .5, 1., .2, 0., .7, 1., .9],
                device="cuda", dtype=torch.float32)
            reference, actual = storage.clone(), storage.clone()
            upstream(reference[:, :vocab], mapping, probs)
            counts = dispatch(patched.apply_min_p, actual[:, :vocab], mapping,
                              probs)
            equal = torch.equal(reference.view(torch.uint8),
                                actual.view(torch.uint8))
            parallel = counters["_min_p_kernel"].calls == 0
            expected_parallel = in_fast_window(vocab, batch)
            predicate = patched._use_parallel_path(actual[:, :vocab], batch,
                                                   vocab)
            assert equal, {"vocab": vocab, "batch": batch, "dtype": str(dtype),
                           "kind": kind, "counts": counts}
            assert predicate == expected_parallel, {
                "vocab": vocab, "batch": batch, "predicate": predicate}
            assert parallel == expected_parallel, {"vocab": vocab,
                                                   "batch": batch,
                                                   "counts": counts}
            if parallel:
                assert counts["_min_p_partial_max"] == 1
                assert counts["_min_p_apply_threshold"] == 1
            else:
                assert counts["_min_p_kernel"] == 1
                assert counts["_min_p_partial_max"] == 0
                assert counts["_min_p_apply_threshold"] == 0
            cases.append({"vocab": vocab, "batch": batch, "dtype": str(dtype),
                          "kind": kind, "parallel_path": parallel,
                          "launch_calls": counts})

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for vocab in (1, 31, 1023, 1024, 1025, 4095, 4096, 4097, 32000,
                          151936):
                for batch in (1, 2, 8):
                    check(vocab, batch, dtype)
            for vocab in (4097, 151936):
                check(vocab, 8, dtype, "special")

        # 4. dispatch boundary, column stride and empty batch
        probes = []

        def probe(label, pair, expect_parallel):
            reference, actual = pair()
            batch = actual.shape[0]
            mapping = torch.arange(batch, device="cuda", dtype=torch.int32)
            probs = torch.full((batch,), .05, device="cuda")
            if batch:
                upstream(reference, mapping, probs)
            counts = dispatch(patched.apply_min_p, actual, mapping, probs)
            parallel = counters["_min_p_kernel"].calls == 0
            parity = torch.equal(reference.view(torch.uint8),
                                 actual.view(torch.uint8))
            predicate = patched._use_parallel_path(actual, batch,
                                                   actual.shape[1])
            assert parallel == expect_parallel, {"probe": label,
                                                 "counts": counts}
            assert predicate == expect_parallel, {"probe": label,
                                                  "predicate": predicate}
            assert parity, {"probe": label, "counts": counts}
            probes.append({"label": label, "shape": list(actual.shape),
                           "stride": list(actual.stride()),
                           "parallel_path": parallel, "parity": parity,
                           "launch_calls": counts})

        def rand(rows, cols):
            def make():
                base = torch.randn((rows, cols), device="cuda")
                return base.clone(), base.clone()
            return make

        probe("rows=8 vocab=32000 (both inclusive edges)", rand(8, 32000), True)
        probe("rows=1 vocab=32000 (lower row edge)", rand(1, 32000), True)
        probe("rows=8 vocab=31999 (one column under)", rand(8, 31999), False)
        probe("rows=9 vocab=151936 (one row over)", rand(9, 151936), False)
        probe("rows=0 empty batch", rand(0, 151936), False)

        strided = torch.randn((2, 151936), device="cuda")[:, ::2]
        assert patched._use_parallel_path(strided, 2, strided.shape[1]) is False
        probes.append({"label": "column stride 2 rejected by predicate",
                       "shape": list(strided.shape),
                       "stride": list(strided.stride()),
                       "parallel_path": False, "parity": None,
                       "launch_calls": None})
        for dtype in (torch.float64, torch.float8_e4m3fn):
            other = torch.empty((2, 40960), device="cuda", dtype=dtype)
            assert patched._use_parallel_path(other, 2, 40960) is False
        probes.append({"label": "float64 and float8 rejected by dtype guard",
                       "parallel_path": False, "parity": None,
                       "launch_calls": None})

        for name, counter in counters.items():
            setattr(patched, name, counter.kernel)

        # 5. local timing through the patched entry point
        timing = []
        for vocab, batch in ((151936, 1), (151936, 2), (151936, 4),
                             (151936, 8), (32000, 2), (32000, 32), (4096, 8)):
            mapping = torch.arange(batch, device="cuda", dtype=torch.int32)
            probs = torch.full((batch,), .05, device="cuda")
            values = {}
            for arm, fn in (("upstream", upstream),
                            ("patched", patched.apply_min_p)):
                x = torch.randn((batch, vocab), device="cuda",
                                dtype=torch.float32)
                median_ms = triton.testing.do_bench(
                    lambda: fn(x, mapping, probs), warmup=5, rep=20,
                    quantiles=[.5, .2, .8])[0]
                values[arm] = median_ms * 1000.0
            row = {"vocab": vocab, "batch": batch,
                   "parallel_path": in_fast_window(vocab, batch),
                   "upstream_us": round(values["upstream"], 2),
                   "patched_us": round(values["patched"], 2),
                   "speedup": round(values["upstream"] / values["patched"], 3)}
            timing.append(row)
            print(json.dumps(row), flush=True)

        fast = [r for r in timing if r["parallel_path"]]
        fallback = [r for r in timing if not r["parallel_path"]]
        assert all(r["speedup"] >= 2.0 for r in fast if r["vocab"] == 151936), timing
        assert all(r["speedup"] >= 1.5 for r in fast if r["vocab"] == 32000), timing
        assert all(0.7 <= r["speedup"] <= 1.4 for r in fallback), timing

        result.update({
            "state": "passed",
            "correctness_cases": len(cases),
            "parallel_cases": sum(c["parallel_path"] for c in cases),
            "fallback_cases": sum(not c["parallel_path"] for c in cases),
            "cases": cases,
            "dispatch_probes": probes,
            "timing": timing,
        })
        (out / "patch-verification.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({key: result[key] for key in
                          ("state", "correctness_cases", "parallel_cases",
                           "fallback_cases")}, indent=2))
    except BaseException:
        result["state"] = "failed"
        result["traceback"] = traceback.format_exc()
        (out / "patch-verification.json").write_text(json.dumps(result, indent=2))
        raise


if __name__ == "__main__":
    main()
