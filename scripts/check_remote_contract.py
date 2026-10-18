"""Offline integration checks for paid-run entry points; no installs or GPU work."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def run(args, env, expected=0):
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    assert result.returncode == expected, (args, result.returncode, result.stdout, result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    checks = []
    for name in ("remote_bootstrap", "preflight", "run_stage", "repro_l4", "setup_cuda", "check_remote_local"):
        run(["bash", "-n", str(ROOT / f"scripts/{name}.sh")], os.environ)
    checks.append("shell syntax for all six entry-point scripts")
    with tempfile.TemporaryDirectory(prefix="ci-remote-contract-") as tmp:
        base = Path(tmp)
        # Use a disposable project so fake evidence never enters real results/.
        project = base / "project"
        import shutil
        for directory in ("scripts", "kernels"):
            shutil.copytree(ROOT / directory, project / directory)
        runtime = base / "custom-runtime"
        bin_dir = runtime / ".venv/bin"
        bin_dir.mkdir(parents=True)
        fake = bin_dir / "python"
        fake.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
if "--out" in sys.argv:
    out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(exist_ok=False)
    (out / "contract.json").write_text(json.dumps({"runtime": os.environ["CI_RUNTIME_DIR"]}))
sys.exit(int(os.environ.get("CONTRACT_EXIT", "0")))
''')
        fake.chmod(0o755)
        env = dict(os.environ, CI_RUNTIME_DIR=str(runtime))
        stage = str(project / "scripts/run_stage.sh")
        for _ in range(2):
            run(["bash", stage, "minp-compare-f32"], env)
        directories = list((runtime / "results").glob("minp-compare-f32-*/contract.json"))
        assert len(directories) == 2
        assert all(json.loads(p.read_text())["runtime"] == str(runtime) for p in directories)
        checks.append("custom runtime, fresh result creation, unique names and copying")
        run(["bash", stage, "minp-compare-f32"], dict(env, CONTRACT_EXIT="7"), 7)
        assert any(p.read_text().strip() == "7" for p in (runtime / "results").glob("*.exitcode"))
        checks.append("stage failure returns and preserves exact exit code")
        (project / ".venv").symlink_to(runtime / ".venv", target_is_directory=True)
        run(["bash", stage, "minp-compare-f32"], dict(env, CI_RUNTIME_DIR=str(project)))
        checks.append("runtime equal to checkout avoids copying files onto themselves")
        repro = str(project / "scripts/repro_l4.sh")
        before = set((runtime / "results").iterdir())
        preview = run(["bash", repro, "--dry-run"], env)
        assert "minp-compare-f32" in preview.stdout
        assert set((runtime / "results").iterdir()) == before
        run(["bash", repro, "--dry-run\r"], env, 2)
        run(["bash", repro, "--dry-run"], dict(env, STAGES="misspelled"), 2)
        checks.append("dry-run does not execute; malformed flags and stages rejected")
        run(["bash", repro], dict(env, STAGES="minp-compare-f32 abba-knob", CONTRACT_EXIT="7"), 1)
        assert not list((runtime / "results").glob("abba-knob-*"))
        checks.append("replication stops on first failed stage")
        bootstrap = str(project / "scripts/remote_bootstrap.sh")
        run(["bash", bootstrap, "--torch-only"], env, 2)
        fake_smi = bin_dir / "nvidia-smi"
        fake_smi.write_text('#!/bin/sh\necho 570.00\n')
        fake_smi.chmod(0o755)
        facts_env = dict(env, PATH=str(bin_dir) + ":" + env["PATH"])
        run(["bash", bootstrap, "--facts-only"], facts_env)
        run(["bash", bootstrap], facts_env, 1)
        checks.append("facts-only and old-driver gate run without installing packages")

        # Verify the model contract with tiny local files, including a missing
        # weight and a corrupt existing file. No network or real checkpoint needed.
        model = base / "model"
        model.mkdir()
        entries = []
        for name, content in {"config.json": b"{}", "model.safetensors": b"weights"}.items():
            (model / name).write_bytes(content)
            entries.append({"rfilename": name, "blobId": hashlib.sha1(
                f"blob {len(content)}\0".encode() + content).hexdigest()})
        metadata = base / "metadata.json"
        metadata.write_text(json.dumps({"sha": "c1899de289a04d12100db370d81485cdf75e47ca", "siblings": entries}))
        model_args = [sys.executable, str(ROOT / "scripts/prepare_model.py"), "--dest", str(model),
                "--metadata", str(metadata), "--verify-only"]
        run(model_args, env)
        (model / "model.safetensors").unlink()
        run(model_args, env, 1)
        (model / "model.safetensors").write_bytes(b"corrupt")
        run(model_args, env, 1)
        checks.append("model verifier rejects missing weights and existing corrupt files")
        server = base / "file-server"
        origin = server / "Qwen/Qwen3-0.6B/raw/c1899de289a04d12100db370d81485cdf75e47ca"
        origin.mkdir(parents=True)
        (origin / "config.json").write_bytes(b"{}")
        (origin / "model.safetensors").write_bytes(b"weights")
        fresh = base / "downloaded"
        run([sys.executable, str(ROOT / "scripts/prepare_model.py"), "--dest", str(fresh),
             "--metadata", str(metadata), "--endpoint", server.as_uri()], env)
        assert (fresh / "model.safetensors").read_bytes() == b"weights"
        checks.append("fresh model download uses configured endpoint and verifies blob hashes")
        run([sys.executable, str(ROOT / "scripts/quant_decode.py"), "--out", str(base / "invalid"),
             "--max-tokens", "1"], env, 2)
        checks.append("invalid decode length fails before importing GPU dependencies")
    payload = json.dumps({"state": "passed", "checks": checks}, indent=2)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
