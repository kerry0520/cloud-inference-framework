"""Download only required files at a pinned public model revision."""

import hashlib
import json
from pathlib import Path
import urllib.request

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEST = Path("/opt/cloud-inference-framework/models/Qwen3-0.6B")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    cached_metadata = Path("/opt/cloud-inference-framework/results/bootstrap-20261017/model-metadata.json")
    if cached_metadata.exists():
        metadata = json.loads(cached_metadata.read_text(encoding="utf-8-sig"))
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
        cached_metadata.write_text(json.dumps(metadata), encoding="utf-8")
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
        dest = DEST / name
        # Existing files are verified below; partial downloads are never accepted.
        if not dest.exists():
            print(f"Downloading {name}", flush=True)
            endpoint = "resolve" if entry.get("lfs") else "raw"
            with urllib.request.urlopen(
                f"https://huggingface.co/{MODEL}/{endpoint}/{REVISION}/{name}",
                timeout=120,
            ) as source, dest.with_suffix(dest.suffix + ".partial").open("wb") as target:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(chunk)
            dest.with_suffix(dest.suffix + ".partial").replace(dest)
    files = []
    for path in sorted(DEST.iterdir()):
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
    (DEST / "download-manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
