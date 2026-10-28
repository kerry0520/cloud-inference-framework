"""Concurrency / output-length scan over the same vLLM serve harness.

One server, stock kernels (no A/B here), ``min_p > 0`` so the min-p path is on
the critical path.  For each cell we run at least four waves of requests with
fresh prompts (the prefix cache cannot carry work across cells), record the
client-side TTFT/ITL/TPOT/total, and take Prometheus ``/metrics`` snapshots
before and after the cell so the engine-side queue/prefill/decode split can be
compared with the client-side numbers.

Grid: ``--concurrency-list`` at ``--max-tokens``, then ``--length-list`` at
``--length-concurrency``.  ``requests = max(--requests-min, 4 * concurrency)``
keeps every cell at >= 4 waves.

Outputs: config.json, cells.jsonl (one row per cell), requests.jsonl,
summary.json, status.json, metrics/<cell>-{before,after}.txt, logs/server.log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from serve_stage import (  # noqa: E402  (same directory, same harness)
    DEFAULT_MODEL,
    describe,
    gpu_free_mib,
    one_request,
    start_server,
    stop_server,
    utc_now,
    wait_health,
    wait_port_free,
)

METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+]?[0-9.eE+]+)$")


def parse_metrics(text: str) -> dict[str, float]:
    """Sums every sample of a metric family across label sets."""
    series: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name.endswith("_bucket"):
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        series[name] = series.get(name, 0.0) + value
    return series


def metric(series: dict, name: str):
    for key in (name, name + "_total"):
        if key in series:
            return series[key]
    return None


def mean_delta(before: dict, after: dict, name: str):
    """Mean of a histogram family over one cell, from sum/count deltas."""
    s0, c0 = metric(before, name + "_sum"), metric(before, name + "_count")
    s1, c1 = metric(after, name + "_sum"), metric(after, name + "_count")
    if None in (s0, c0, s1, c1) or c1 - c0 <= 0:
        return None
    return (s1 - s0) / (c1 - c0)


def counter_delta(before: dict, after: dict, name: str):
    a, b = metric(before, name), metric(after, name)
    if a is None or b is None:
        return None
    return b - a


async def snapshot(cfg, path: Path):
    """GET /metrics; returns {} when the endpoint is disabled/unavailable."""
    url = "http://127.0.0.1:%d/metrics" % cfg["port"]
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
        resp.raise_for_status()
        text = resp.text
        path.write_text(text, encoding="utf-8")
        return parse_metrics(text)
    except Exception as exc:  # noqa: BLE001 - metrics are evidence, not a gate
        path.write_text("unavailable: %r\n" % (exc,), encoding="utf-8")
        return {}


def prompts_for(cfg, seed_offset: int, count: int):
    generator = np.random.default_rng(cfg["seed_base"] + seed_offset)
    return generator.integers(100, 15000, size=(count, cfg["prompt_len"])).tolist()


async def run_cell(cfg, cell, prompts, out_dir):
    local = dict(cfg, concurrency=cell["concurrency"], max_tokens=cell["max_tokens"])
    concurrency = local["concurrency"]
    requests = max(local["requests_min"], 4 * concurrency)
    before_path = out_dir / "metrics" / ("%s-before.txt" % cell["name"])
    after_path = out_dir / "metrics" / ("%s-after.txt" % cell["name"])
    before = await snapshot(local, before_path)
    rows = []
    limits = httpx.Limits(max_connections=concurrency + 16,
                          max_keepalive_connections=concurrency + 16)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0),
                                 limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)

        async def one(index):
            async with sem:
                row = await one_request(client, local, prompts[index],
                                        min_p=local["min_p"],
                                        seed=local["seed_base"] + index)
            row.update({"cell": cell["name"], "index": index,
                        "concurrency": concurrency, "max_tokens": cell["max_tokens"]})
            rows.append(row)
            return row

        waves = math.ceil(requests / concurrency)
        for wave in range(waves):
            begin = wave * concurrency
            seeds = list(range(begin, min(requests, begin + concurrency)))
            await asyncio.gather(*(one(i) for i in seeds))
    wall = time.perf_counter() - started
    after = await snapshot(local, after_path)
    with (out_dir / "requests.jsonl").open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    tokens = sum(r["tokens"] for r in rows)
    engine = {
        "ttft_ms": mean_delta(before, after, "vllm:time_to_first_token_seconds"),
        "queue_ms": mean_delta(before, after, "vllm:request_queue_time_seconds"),
        "prefill_ms": mean_delta(before, after, "vllm:request_prefill_time_seconds"),
        "decode_ms": mean_delta(before, after, "vllm:request_decode_time_seconds"),
        "tpot_ms": mean_delta(before, after,
                              "vllm:request_time_per_output_token_seconds"),
        "e2e_ms": mean_delta(before, after, "vllm:e2e_request_latency_seconds"),
    }
    engine = {k: (None if v is None else v * 1e3) for k, v in engine.items()}
    counted = {
        "preemptions": counter_delta(before, after, "vllm:num_preemptions"),
        "prefix_hits": counter_delta(before, after, "vllm:prefix_cache_hits"),
        "prefix_queries": counter_delta(before, after, "vllm:prefix_cache_queries"),
        "generation_tokens": counter_delta(before, after, "vllm:generation_tokens"),
        "request_success": counter_delta(before, after, "vllm:request_success"),
    }
    generated = counted["generation_tokens"]
    server_tps = generated / wall if generated and wall else None
    hit_rate = None
    if counted["prefix_queries"]:
        hit_rate = counted["prefix_hits"] / counted["prefix_queries"]
    return {
        "name": cell["name"],
        "concurrency": concurrency,
        "max_tokens": cell["max_tokens"],
        "requests": len(rows),
        "wall_seconds": wall,
        "requests_per_second": len(rows) / wall if wall else None,
        "tokens_per_second": tokens / wall if wall else None,
        "server_tokens_per_second": server_tps,
        "engine_ms": engine,
        "queue_share": (engine["queue_ms"] / engine["ttft_ms"]
                        if engine["queue_ms"] is not None and engine["ttft_ms"] else None),
        "counters": counted,
        "prefix_hit_rate": hit_rate,
        "kv_usage_perc": after.get("vllm:kv_cache_usage_perc"),
        "ttft_ms": describe([r["ttft_ms"] for r in rows]),
        "tpot_ms": describe([r["tpot_ms"] for r in rows]),
        "itl_median_ms": describe([r["itl_median_ms"] for r in rows]),
        "total_ms": describe([r["total_ms"] for r in rows]),
        "token_mismatch": sum(1 for r in rows if r["tokens"] != cell["max_tokens"]),
        "metrics": {"before": str(before_path), "after": str(after_path),
                    "available": bool(before) and bool(after)},
    }


async def run_all(cfg, cells, out_dir):
    results = []
    log_path = out_dir / "logs" / "server.log"
    wait_port_free(cfg["port"])
    print("[scan] gpu_free_mib=%s" % gpu_free_mib(), flush=True)
    proc, log = start_server(cfg, "base", log_path)
    try:
        clock = time.perf_counter()
        await wait_health(cfg, proc)
        print("[scan] healthy in %.1fs" % (time.perf_counter() - clock), flush=True)

        max_concurrency = max(cell["concurrency"] for cell in cells)
        warm = prompts_for(cfg, 90000, max_concurrency)
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
            for _ in range(cfg["warmup_rounds"]):
                await asyncio.gather(*(
                    one_request(client, dict(cfg, max_tokens=cfg["warmup_tokens"]),
                                warm[i], min_p=cfg["min_p"],
                                seed=cfg["seed_base"] + 90000 + i)
                    for i in range(max_concurrency)))
        print("[scan] warmup done at concurrency %d" % max_concurrency, flush=True)

        for repeat in range(cfg["repeats"]):
            for index, cell in enumerate(cells):
                count = max(cfg["requests_min"], 4 * cell["concurrency"])
                # Fresh prompts per repeat: a repeated cell must not be served
                # from the prefix cache of its previous repetition.
                prompts = prompts_for(cfg, 1000 * (index + 1) + 100000 * repeat, count)
                cell_started = utc_now()
                repeated = dict(cell, name="%s-r%d" % (cell["name"], repeat),
                                base=cell["name"], repeat=repeat)
                row = await run_cell(cfg, repeated, prompts, out_dir)
                row["started_at"] = cell_started
                row["ok"] = row["token_mismatch"] == 0 and row["requests"] > 0
                results.append(row)
                with (out_dir / "cells.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps({k: row[k] for k in
                                  ("name", "concurrency", "max_tokens", "requests",
                                   "ttft_ms", "tpot_ms", "tokens_per_second")},
                                 sort_keys=True), flush=True)
    finally:
        stop_server(proc, log)
        wait_port_free(cfg["port"], timeout=90)
    return results


def aggregate(results):
    """Per base cell: median across repeats plus the min/max of per-repeat p99."""
    groups: dict[str, list[dict]] = {}
    for row in results:
        groups.setdefault(row["name"].rsplit("-r", 1)[0], []).append(row)
    out = []
    for name, rows in groups.items():
        def med(key, sub="median"):
            values = [r[key][sub] for r in rows if r[key] and r[key].get(sub) is not None]
            return statistics.median(values) if values else None

        def spread(key, sub="p99"):
            values = [r[key][sub] for r in rows if r[key] and r[key].get(sub) is not None]
            return (min(values), max(values)) if values else (None, None)

        tps = [r["tokens_per_second"] for r in rows if r["tokens_per_second"]]
        rps = [r["requests_per_second"] for r in rows if r["requests_per_second"]]
        out.append({
            "name": name,
            "concurrency": rows[0]["concurrency"],
            "max_tokens": rows[0]["max_tokens"],
            "repeats": len(rows),
            "ttft_p50_ms": med("ttft_ms"),
            "ttft_p99_ms": spread("ttft_ms"),
            "tpot_p50_ms": med("tpot_ms"),
            "tpot_p99_ms": spread("tpot_ms"),
            "tokens_per_second": statistics.median(tps) if tps else None,
            "requests_per_second": statistics.median(rps) if rps else None,
        })
    return sorted(out, key=lambda item: (item["max_tokens"], item["concurrency"]))


def table(results, aggregated):
    def num(value, digits=1):
        return "-" if value is None else ("%.*f" % (digits, value))

    lines = [
        "| cell | conc | out tok | n | TTFT p50 | TTFT p99 | TPOT p50 | ITL p99 "
        "| total p50 | req/s | tok/s | engine TTFT | queue | prefill | preempt |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        engine = row["engine_ms"]
        lines.append("| %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["name"], row["concurrency"], row["max_tokens"], row["requests"],
            num(row["ttft_ms"].get("median")), num(row["ttft_ms"].get("p99")),
            num(row["tpot_ms"].get("median")), num(row["itl_median_ms"].get("p99")),
            num(row["total_ms"].get("median")), num(row["requests_per_second"], 2),
            num(row["tokens_per_second"], 0), num(engine["ttft_ms"]),
            num(engine["queue_ms"]), num(engine["prefill_ms"]),
            num(row["counters"].get("preemptions"), 0)))
    if aggregated and aggregated[0]["repeats"] > 1:
        lines += ["",
                  "| cell | conc | out tok | reps | TTFT p50 (med of reps) "
                  "| TTFT p99 (min..max) | TPOT p50 (med) | TPOT p99 (min..max) "
                  "| req/s (med) | tok/s (med) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for item in aggregated:
            lo, hi = item["ttft_p99_ms"]
            tlo, thi = item["tpot_p99_ms"]
            lines.append("| %s | %d | %d | %d | %s | %s..%s | %s | %s..%s | %s | %s |" % (
                item["name"], item["concurrency"], item["max_tokens"], item["repeats"],
                num(item["ttft_p50_ms"]), num(lo), num(hi), num(item["tpot_p50_ms"]),
                num(tlo), num(thi), num(item["requests_per_second"], 2),
                num(item["tokens_per_second"], 0)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency-list", default="1,2,4,8,16,32")
    parser.add_argument("--length-list", default="16,64,256")
    parser.add_argument("--length-concurrency", type=int, default=8)
    parser.add_argument("--requests-min", type=int, default=24)
    parser.add_argument("--min-p", type=float, default=0.1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="qwen3")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(args).items() if k != "out"}
    cfg["model"] = str(Path(args.model).resolve()) if args.model == DEFAULT_MODEL else args.model
    cfg["created_at"] = utc_now()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    cells = [{"name": "c%d" % c, "concurrency": c, "max_tokens": args.max_tokens}
             for c in (int(v) for v in args.concurrency_list.split(",") if v.strip())]
    cells += [{"name": "len%d" % n, "concurrency": args.length_concurrency, "max_tokens": n}
              for n in (int(v) for v in args.length_list.split(",") if v.strip())]

    results = asyncio.run(run_all(cfg, cells, out_dir))
    aggregated = aggregate(results)
    ok = all(row["ok"] for row in results) and len(results) == len(cells) * args.repeats
    summary = {"created_at": utc_now(), "config": cfg, "cells": results,
               "aggregated": aggregated,
               "verdict": "passed" if ok else "failed"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps(
        {"passed": ok, "cells": len(results), "failed_cells":
         [row["name"] for row in results if not row["ok"]]}, indent=2), encoding="utf-8")
    print("\n" + table(results, aggregated), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
