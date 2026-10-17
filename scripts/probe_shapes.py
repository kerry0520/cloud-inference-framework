import argparse
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model="/opt/cloud-inference-framework/models/Qwen3-0.6B",
                  dtype="float16", max_model_len=2048, max_num_seqs=8,
                  max_num_batched_tokens=2048, gpu_memory_utilization=0.65,
                  enable_prefix_caching=False, enforce_eager=True, seed=0,
                  worker_extension_cls="project_worker.ProjectWorkerExtension")
        metadata = llm.collective_rpc("probe_install")
        result = llm.generate([{"prompt_token_ids": [1000+i]*32} for i in range(2)],
                              SamplingParams(temperature=0, max_tokens=4,
                                             min_tokens=4, ignore_eos=True),
                              use_tqdm=False)
        shapes = llm.collective_rpc("probe_collect")
        assert len(metadata[0]["linears"]) == 4
        assert len(shapes[0]) == 8
        assert all(len(r.outputs[0].token_ids) == 4 for r in result)
        (out / "shapes.json").write_text(json.dumps({
            "metadata": metadata, "observed": shapes,
            "diagnostic_eager_only": True,
            "resolved_async_scheduling": llm.llm_engine.vllm_config.scheduler_config.async_scheduling,
        }, indent=2))
        (out / "status.json").write_text(json.dumps({"state": "passed"}))
        print(json.dumps(shapes, indent=2))
    except BaseException:
        (out / "status.json").write_text(json.dumps(
            {"state": "failed", "traceback": traceback.format_exc()}, indent=2))
        raise


if __name__ == "__main__":
    main()
