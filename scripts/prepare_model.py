"""Download only required files at a pinned public model revision."""

import argparse
import hashlib
import os
import json
from pathlib import Path
import urllib.request

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEST = Path(os.environ.get("CI_RUNTIME_DIR", "/opt/cloud-inference-framework")) / "models/Qwen3-0.6B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--metadata", type=Path,
                        default=Path(__file__).resolve().parent.parent / "configs/model-metadata.json")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    dest_dir = args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    endpoint_base = args.endpoint.rstrip("/")
    url = f"{endpoint_base}/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    cached_metadata = args.metadata
    if cached_metadata.exists():
        metadata = json.loads(cached_metadata.read_text(encoding="utf-8-sig"))
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
    assert metadata["sha"] == REVISION
    expected = {}
    git_blobs = {}
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        if Path(name).name != name:
            continue
        if not (name.endswith((".json", ".txt", ".safetensors")) or name == "LICENSE"):
            continue
        expected[name] = entry.get("lfs", {}).get("sha256")
        git_blobs[name] = entry.get("blobId")
        dest = dest_dir / name
        # Existing files are verified below; partial downloads are never accepted.
        if not dest.exists():
            if args.verify_only:
                raise FileNotFoundError(f"Required model file missing: {dest}")
            print(f"Downloading {name}", flush=True)
            endpoint = "resolve" if entry.get("lfs") else "raw"
            with urllib.request.urlopen(
                f"{endpoint_base}/{MODEL}/{endpoint}/{REVISION}/{name}",
                timeout=120,
            ) as source, dest.with_suffix(dest.suffix + ".partial").open("wb") as target:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(chunk)
            dest.with_suffix(dest.suffix + ".partial").replace(dest)
    files = []
    assert "config.json" in expected and any(n.endswith(".safetensors") for n in expected)
    for path in sorted(dest_dir.iterdir()):
        if path.name in expected:
            digest = hashlib.sha256()
            git_digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    git_digest.update(chunk)
            sha = digest.hexdigest()
            if expected[path.name]:
                assert sha == expected[path.name], f"SHA mismatch: {path.name}"
            elif git_blobs[path.name]:
                assert git_digest.hexdigest() == git_blobs[path.name], f"Git blob mismatch: {path.name}"
            files.append({"name": path.name, "bytes": path.stat().st_size,
                          "sha256": sha, "upstream_lfs_sha256": expected[path.name]})
    result = {"model": MODEL, "revision": REVISION, "files": files}
    (dest_dir / "download-manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
