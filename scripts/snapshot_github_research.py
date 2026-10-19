"""Save public read-only GitHub research snapshots; never uses account tokens."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


QUERIES = {
    "vllm-sampling": "repo:vllm-project/vllm is:issue is:open sampling",
    "sglang-sampling": "repo:sgl-project/sglang is:issue is:open sampling",
    "flashinfer-sampling": "repo:flashinfer-ai/flashinfer is:issue is:open sampling",
    "lmdeploy-bugs": "repo:InternLM/lmdeploy is:issue is:open label:bug",
    "vllm-first": 'repo:vllm-project/vllm is:issue is:open label:"good first issue"',
    "sglang-first": 'repo:sgl-project/sglang is:issue is:open label:"good first issue"',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--url", action="append", default=[], help="name=https://api.github.com/... or raw.githubusercontent.com/...")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    urls = {}
    if args.url:
        for item in args.url:
            name, url = item.split("=", 1)
            assert Path(name).name == name and name not in (".", "..")
            urls[name] = url
    else:
        urls = {name: "https://api.github.com/search/issues?" + urllib.parse.urlencode(
            {"q": q, "sort": "updated", "order": "desc", "per_page": 25})
                for name, q in QUERIES.items()}

    def fetch(item):
        name, url = item
        assert urllib.parse.urlparse(url).hostname in ("api.github.com", "raw.githubusercontent.com")
        record = {"name": name, "url": url, "fetched_utc": datetime.now(timezone.utc).isoformat()}
        request = urllib.request.Request(url, headers={"User-Agent": "cloud-inference-project-readonly-research"})
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                data = response.read()
                record.update(status=response.status, sha256=hashlib.sha256(data).hexdigest(),
                              bytes=len(data), etag=response.headers.get("ETag"))
            (args.out / (name + ".snapshot")).write_bytes(data)
            try:
                payload = json.loads(data)
            except ValueError:
                return record, {"name": name, "bytes": len(data)}
            if isinstance(payload, dict) and "items" in payload:
                compact = [{"number": x["number"], "title": x["title"], "state": x["state"],
                            "updated": x["updated_at"], "comments": x["comments"],
                            "assignees": [a["login"] for a in x["assignees"]],
                            "url": x["html_url"]} for x in payload["items"]]
                return record, {"name": name, "total": payload["total_count"], "items": compact}
            return record, {"name": name, "saved": True, "bytes": len(data)}
        except (urllib.error.URLError, TimeoutError) as error:
            record["error"] = str(error)
            record["http_status"] = getattr(error, "code", None)
            return record, record

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, urls.items()))
    # Use a new batch directory per invocation; do not overwrite earlier evidence.
    (args.out / "manifest.json").write_text(json.dumps([r[0] for r in results], indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps([r[1] for r in results], indent=2), encoding="utf-8")
    print(json.dumps([r[1] for r in results], indent=2))
    if any("error" in r[0] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
