"""Materialize the kernel from an upstream pull request as a loadable module.

Why: the engine-level paired test (scripts/abba_patch.py) installs a module by
path, so the PR's `min_p.py` has to exist as a file with provenance attached.
Reconstructing it from the diff (rather than copying from GitHub's raw view)
keeps the evidence honest: what we measured is exactly the diff applied to the
pinned file, and the hashes say so.

The diff is filtered to the kernel file; the benchmark and the test in the PR
are not used here.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

KERNEL_PATH = "vllm/v1/worker/gpu/sample/min_p.py"

# Markers that must be present after applying the diff.  If a future revision of
# the PR moves them, this fails loudly instead of silently measuring something
# else.
EXPECTED = (
    "tl.full((BLOCK_SIZE,), float(\"-inf\"), tl.float32)",
    "max_val = tl.max(max_val, 0)",
    "8192 if vocab_size > 16384 and num_tokens <= 32 else 1024",
    "num_warps=16 if BLOCK_SIZE == 8192 else 4",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="evidence directory")
    parser.add_argument("--target", required=True, help="stable directory for the module")
    parser.add_argument("--upstream", required=True, help="pinned min_p.py")
    parser.add_argument("--diff", required=True, help="pull request diff file")
    parser.add_argument("--commit", required=True, help="PR head commit this diff came from")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    text = Path(args.diff).read_text(encoding="utf-8")
    parts = text.split("diff --git ")
    keep = [p for p in parts[1:] if KERNEL_PATH in p.split("\n")[0]]
    if not keep:
        raise SystemExit(f"no {KERNEL_PATH} hunk in {args.diff}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        subset = work / "min_p_pr.diff"
        subset.write_text("diff --git " + keep[0], encoding="utf-8")
        staged = work / KERNEL_PATH
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(Path(args.upstream).read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(work)], check=True)
        subprocess.run(["git", "apply", str(subset)], cwd=str(work), check=True)
        applied = staged.read_text(encoding="utf-8")
        for marker in EXPECTED:
            if marker not in applied:
                raise SystemExit(f"marker missing after apply: {marker!r}")
        module_path = target / "min_p.py"
        shutil.copyfile(staged, module_path)

    (target / "provenance.json").write_text(
        json.dumps(
            {
                "purpose": "engine-level cross-check of an upstream PR kernel",
                "pr_commit": args.commit,
                "diff": str(args.diff),
                "diff_sha256": sha256(args.diff),
                "diff_bytes": Path(args.diff).stat().st_size,
                "upstream": str(args.upstream),
                "upstream_sha256": sha256(args.upstream),
                "module": str(module_path),
                "module_sha256": sha256(module_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if out.resolve() != target.resolve():
        shutil.copyfile(target / "provenance.json", out / "provenance.json")
    print(f"module  {module_path}")
    print(f"sha256  {sha256(module_path)}")
    print(f"commit  {args.commit}")
    print(f"diff    {sha256(args.diff)}")


if __name__ == "__main__":
    main()
