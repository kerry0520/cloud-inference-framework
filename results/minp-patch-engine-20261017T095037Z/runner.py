import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time
import traceback


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--patch-module")
    parser.add_argument("--patch")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only and not args.patch_module:
        parser.error("--check-only requires --patch-module")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        from vllm import LLM, SamplingParams
        from kernels import min_p
        shutil.copyfile(min_p.__file__, out / "candidate.py")
        shutil.copyfile(__file__, out / "runner.py")
        shutil.copyfile(Path(__file__).with_name("project_worker.py"), out / "worker.py")
        llm = LLM(model="/opt/cloud-inference-framework/models/Qwen3-0.6B",
                  dtype="float16", max_model_len=2048, max_num_seqs=8,
                  max_num_batched_tokens=2048, gpu_memory_utilization=0.65,
                  enable_prefix_caching=False, seed=0, disable_log_stats=False,
                  worker_extension_cls="project_worker.ProjectWorkerExtension",
                  profiler_config={"profiler": "torch",
                                   "torch_profiler_dir": str(out / "traces"),
                                   "torch_profiler_with_stack": False,
                                   "torch_profiler_dump_cuda_time_total": False,
                                   "delay_iterations": 4, "max_iterations": 12,
                                   "ignore_frontend": True})
        write(out / "config.json", {"config": str(llm.llm_engine.vllm_config),
                                    "async_scheduling": llm.llm_engine.vllm_config.scheduler_config.async_scheduling,
                                    "torch": torch.__version__})

        def run(batch, length=128, p=.05):
            prompts = [{"prompt_token_ids": [1000+i]*128} for i in range(batch)]
            params = [SamplingParams(temperature=.8, min_p=p, seed=123+i,
                                     max_tokens=length, min_tokens=length,
                                     ignore_eos=True, detokenize=False)
                      for i in range(batch)]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            assert len(outputs) == batch
            assert all(r.finished and len(r.outputs[0].token_ids) == length for r in outputs)
            return outputs

        if args.patch_module:
            module_path = Path(args.patch_module).resolve()
            llm.collective_rpc("minp_install_file",
                               args=("verify", str(module_path), str(out / "real-tensors")))
            run(2, 4)
            patch_stats = llm.collective_rpc("minp_stats")
            assert patch_stats[0]["checked"] >= 4, patch_stats
            assert patch_stats[0]["mismatches"] == 0, patch_stats
            kernels = patch_stats[0].get("kernels", {})
            assert kernels.get("_min_p_partial_max", 0) >= 1, patch_stats
            assert kernels.get("_min_p_apply_threshold", 0) >= 1, patch_stats
            assert kernels.get("_min_p_kernel", 0) == 0, patch_stats
            evidence = {"stats": patch_stats, "module": str(module_path),
                        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest()}
            if args.patch:
                patch_path = Path(args.patch).resolve()
                evidence["patch"] = str(patch_path)
                evidence["patch_sha256"] = hashlib.sha256(patch_path.read_bytes()).hexdigest()
            write(out / "patch-engine-verification.json", evidence)
            print(json.dumps(evidence, indent=2, default=str), flush=True)
            if args.check_only:
                write(out / "status.json", {"state": "passed", "check_only": True,
                                            "patch_engine": patch_stats})
                return

        if not args.timing_only:
            llm.collective_rpc("minp_install", args=("verify", str(out / "real-tensors")))
            run(2, 4)
            verification = llm.collective_rpc("minp_stats")
            assert verification[0]["checked"] >= 4
            assert verification[0]["mismatches"] == 0
            write(out / "real-verification.json", verification)

            disabled_ids = []
            disabled_stats = []
            for arm in ("base", "candidate"):
                llm.collective_rpc("minp_install", args=(arm,))
                result = run(2, 4, p=0.)
                disabled_ids.append([list(x.outputs[0].token_ids) for x in result])
                stats = llm.collective_rpc("minp_stats")
                disabled_stats.append(stats)
                assert stats[0]["calls"] == 0
            assert disabled_ids[0] == disabled_ids[1]
            write(out / "disabled-path.json", {"stats": disabled_stats, "ids": disabled_ids})

        rows = []
        with (out / "timing.jsonl").open("w") as stream:
            for batch in (1, 2, 4):
                for arm in ("base", "candidate"):
                    llm.collective_rpc("minp_install", args=(arm,))
                    run(batch)
                reference = None
                for block, order in enumerate(("ABBA", "BAAB")):
                    for position, letter in enumerate(order):
                        arm = "base" if letter == "A" else "candidate"
                        llm.collective_rpc("minp_install", args=(arm,))
                        start = time.perf_counter()
                        outputs = run(batch)
                        wall = time.perf_counter() - start
                        ids = [list(x.outputs[0].token_ids) for x in outputs]
                        if reference is None:
                            reference = ids
                        assert ids == reference, "Seeded model output changed across arms"
                        stats = llm.collective_rpc("minp_stats")
                        assert stats[0]["calls"] > 0
                        intervals = [1000*(x.metrics.last_token_ts-x.metrics.first_token_ts)
                                     /127 for x in outputs]
                        row = {"batch": batch, "block": block, "position": position,
                               "arm": arm, "wall_seconds": wall,
                               "mean_decode_intervals_ms": intervals,
                               "output_token_ids": ids, "worker_stats": stats}
                        rows.append(row)
                        stream.write(json.dumps(row)+"\n")
                        stream.flush()
                        print(json.dumps({k: row[k] for k in
                                          ("batch", "block", "arm", "wall_seconds")}), flush=True)

        if not args.timing_only:
            profile_ids = []
            for arm in ("base", "candidate"):
                llm.collective_rpc("minp_install", args=(arm,))
                llm.start_profile(profile_prefix=arm)
                outputs = run(2, 32)
                llm.stop_profile()
                profile_ids.append([list(x.outputs[0].token_ids) for x in outputs])
            assert profile_ids[0] == profile_ids[1]
        summaries = []
        for batch in (1, 2, 4):
            values = {}
            for arm in ("base", "candidate"):
                samples = [r for r in rows if r["batch"] == batch and r["arm"] == arm]
                values[arm] = {"median_wall_ms": statistics.median(r["wall_seconds"]*1000 for r in samples),
                               "wall_ms": [r["wall_seconds"]*1000 for r in samples],
                               "median_request_mean_decode_ms": statistics.median(
                                   v for r in samples for v in r["mean_decode_intervals_ms"])}
            summaries.append({"batch": batch, **values,
                              "wall_reduction_fraction": 1-values["candidate"]["median_wall_ms"]/values["base"]["median_wall_ms"]})
        write(out / "summary.json", summaries)
        write(out / "status.json", {"state": "passed", "timing_rows": len(rows),
                                     "timing_only": args.timing_only,
                                     "timing_sha256": hashlib.sha256((out / "timing.jsonl").read_bytes()).hexdigest()})
        print(json.dumps(summaries, indent=2))
    except BaseException:
        write(out / "status.json", {"state": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
