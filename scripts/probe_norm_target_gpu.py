"""Fresh-process GPU reproduction for FlashInfer #5355's norm follow-up.

Uses the installed wheel without changing it. Records expected failure alongside
normal and explicit-CUDA controls; does not claim current-main GPU validation.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def child(out):
    result = {"target_env": os.environ.get("CUTE_DSL_ARCH"),
              "force_cuda_env": os.environ.get("FLASHINFER_USE_CUDA_NORM")}
    try:
        import torch
        import flashinfer
        import flashinfer.norm as norm
        from cutlass.cutlass_dsl import CuTeDSL
        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      flashinfer=flashinfer.__version__, import_path=flashinfer.__file__,
                      cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
                      gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      captured=str(CuTeDSL._get_dsl().envar.arch),
                      cuda_fallback=norm._use_cuda_norm(torch.device("cuda:0")))
        source = inspect.getsource(norm._use_cuda_norm)
        (out / "dispatch.py").write_text(source)
        result["dispatch_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        torch.manual_seed(0)
        x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
        y = norm.rmsnorm(x, weight)
        torch.cuda.synchronize()
        ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        torch.testing.assert_close(y, ref, atol=0.02, rtol=0.02)
        result.update(state="passed", max_abs_error=float((y.float() - ref.float()).abs().max()))
    except Exception as error:
        result.update(state="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["state"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        raise SystemExit(child(args.out))
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, target, force in (("native", "sm_120a", "0"),
                                ("mismatch", "sm_90a", "0"),
                                ("mismatch-cuda", "sm_90a", "1")):
        case = args.out / name
        case.mkdir()
        env = dict(os.environ, CUTE_DSL_ARCH=target, FLASHINFER_USE_CUDA_NORM=force)
        with (case / "run.log").open("w") as log:
            try:
                process = subprocess.run([sys.executable, __file__, "--child", "--out", str(case)],
                                         env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
        (case / "exitcode").write_text(str(code) + "\n")
        payload = json.loads((case / "result.json").read_text()) if (case / "result.json").exists() else {}
        rows.append(dict(case=name, exitcode=code, result=payload))
    reproduced = (rows[0]["exitcode"] == 0 and rows[1]["exitcode"] != 0
                  and rows[1]["result"].get("cuda_fallback") is False
                  and "cudaErrorNoKernelImageForDevice" in rows[1]["result"].get("error", "")
                  and rows[2]["exitcode"] == 0)
    (args.out / "summary.json").write_text(json.dumps(
        dict(scope="installed wheel only; isolated CUTE_DSL_ARCH mismatch, not a mixed-GPU host",
             reproduced=reproduced, cases=rows), indent=2))
    print(json.dumps(dict(reproduced=reproduced, cases=[dict(case=r["case"], exitcode=r["exitcode"])
                                                     for r in rows]), indent=2))
    raise SystemExit(0 if reproduced else 1)


if __name__ == "__main__":
    main()
