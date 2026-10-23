"""Serving-level A/B of the parallel min-p kernel over the OpenAI endpoint.

Each session starts ``vllm serve`` on a fixed port, waits for ``/health``, runs
two load phases with identical request sets (``min_p=0`` then ``min_p=0.1``)
and stops the server.  The patched arm exports ``CI_MINP_PATCH`` so
``scripts/sitecustomize.py`` installs ``kernels/min_p.py`` inside the engine
workers; the base arm runs the stock vLLM kernel.

Sessions are scheduled in ABBA order (base, patched, patched, base) per block so
slow drift (thermal, clocks, compile cache) hits both arms evenly.  Requests are
launched wave by wave at fixed concurrency; every request asks for exactly
``--max-tokens`` output tokens with ``ignore_eos`` so token counts match.

Outputs: config.json, sessions.jsonl (one row per session, appended as soon as it
finishes), requests.jsonl, summary.json, status.json and logs/.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = Path(os.environ.get("CI_RUNTIME_DIR", "/opt/cloud-inference-framework"))
DEFAULT_MODEL = str(RUNTIME / "models/Qwen3-0.6B")


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pct(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(0, math.ceil(q / 100.0 * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": pct(values, 90),
        "p99": pct(values, 99),
        "min": min(values),
        "max": max(values),
    }


def gpu_free_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - diagnostics only
        return -1


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_port_free(port, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_open(port):
            return True
        time.sleep(0.5)
    return False


def start_server(cfg, arm, log_path):
    env = os.environ.copy()
    env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
    env.pop("CI_MINP_PATCH", None)
    env.pop("CI_MINP_MARKER_FILE", None)
    env.pop("CI_MINP_VERIFY_FIRST", None)
    if arm == "patched":
        env["CI_MINP_PATCH"] = str(cfg["patch_module"])
        env["CI_MINP_MARKER_FILE"] = str(log_path) + ".markers.txt"
        env["CI_MINP_STATS_EVERY"] = str(cfg["stats_every"])
        if cfg["verify_first"]:
            env["CI_MINP_VERIFY_FIRST"] = str(cfg["verify_first"])
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", cfg["model"],
        "--host", "127.0.0.1", "--port", str(cfg["port"]),
        "--served-model-name", cfg["served_model_name"],
        "--dtype", "float16", "--seed", "0",
        "--max-model-len", str(cfg["max_model_len"]),
        "--max-num-seqs", str(cfg["max_num_seqs"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
    ]
    log = log_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    return proc, log


def stop_server(proc, log):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)
    finally:
        log.close()


async def wait_health(cfg, proc, timeout=420.0):
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("server exited with code %s" % proc.returncode)
            try:
                resp = await client.get("http://127.0.0.1:%d/health" % cfg["port"])
                if resp.status_code == 200:
                    return time.monotonic()
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("server did not become healthy in %.0fs" % timeout)


async def one_request(client, cfg, prompt, *, min_p, seed):
    body = {
        "model": cfg["served_model_name"],
        "prompt": prompt,
        "max_tokens": cfg["max_tokens"],
        "min_tokens": cfg["max_tokens"],
        "temperature": 0.8,
        "min_p": min_p,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "skip_special_tokens": True,
        "stream_options": {"include_usage": True},
    }
    url = "http://127.0.0.1:%d/v1/completions" % cfg["port"]
    sent = time.perf_counter()
    first = None
    marks = []
    text = []
    usage = None
    async with client.stream("POST", url, json=body) as resp:
        if resp.status_code != 200:
            payload = (await resp.aread()).decode("utf-8", "replace")
            raise RuntimeError("HTTP %d: %s" % (resp.status_code, payload[:400]))
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            now = time.perf_counter()
            marks.append(now - sent)
            if first is None:
                first = now - sent
            piece = choices[0].get("text") or ""
            if piece:
                text.append(piece)
    done = time.perf_counter()
    n_tokens = (usage or {}).get("completion_tokens") or len(marks)
    itls = [b - a for a, b in zip(marks, marks[1:])] if len(marks) > 1 else []
    tpot = ((marks[-1] - marks[0]) / (n_tokens - 1)) if n_tokens > 1 and len(marks) > 1 else None
    digest = hashlib.sha1("".join(text).encode("utf-8")).hexdigest()
    return {
        "seed": seed,
        "min_p": min_p,
        "ttft_ms": None if first is None else first * 1e3,
        "tpot_ms": None if tpot is None else tpot * 1e3,
        "itl_median_ms": None if not itls else statistics.median(itls) * 1e3,
        "itl_max_ms": None if not itls else max(itls) * 1e3,
        "total_ms": (done - sent) * 1e3,
        "tokens": n_tokens,
        "chunks": len(marks),
        "text_sha1": digest,
        "text_chars": sum(len(t) for t in text),
    }


async def run_phase(cfg, phase_name, min_p, prompts, out_rows, requests, session_id):
    sem = asyncio.Semaphore(cfg["concurrency"])
    limits = httpx.Limits(max_connections=cfg["concurrency"] + 16,
                          max_keepalive_connections=cfg["concurrency"] + 16)
    started = time.perf_counter()

    async def one(index, seed):
        async with sem:
            row = await one_request(client, cfg, prompts[index], min_p=min_p, seed=seed)
        row.update({"session": session_id, "phase": phase_name, "index": index,
                    "started_at": started})
        out_rows.append(row)
        return row

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0),
                                 limits=limits) as client:
        waves = math.ceil(requests / cfg["concurrency"])
        for wave in range(waves):
            seeds = list(range(wave * cfg["concurrency"],
                                min(requests, (wave + 1) * cfg["concurrency"])))
            await asyncio.gather(*(one(i, cfg["seed_base"] + i) for i in seeds))
            if cfg["pause"]:
                await asyncio.sleep(cfg["pause"])
    elapsed = time.perf_counter() - started
    tokens = sum(r["tokens"] for r in out_rows)
    return {
        "requests": len(out_rows),
        "wall_seconds": elapsed,
        "requests_per_second": len(out_rows) / elapsed if elapsed else None,
        "tokens_per_second": tokens / elapsed if elapsed else None,
        "ttft_ms": describe([r["ttft_ms"] for r in out_rows]),
        "tpot_ms": describe([r["tpot_ms"] for r in out_rows]),
        "itl_median_ms": describe([r["itl_median_ms"] for r in out_rows]),
        "total_ms": describe([r["total_ms"] for r in out_rows]),
    }


def parse_log(log_path):
    installed = 0
    stats = []
    verify = {"calls": 0, "checked": 0, "mismatches": 0}
    marker_path = Path(str(log_path) + ".markers.txt")
    source = marker_path if marker_path.exists() and marker_path.stat().st_size else log_path
    if not source.exists():
        return {"installed": 0, "stats": stats, "verify": verify}
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("CI_MINP_INSTALLED"):
            installed += 1
        elif line.startswith("CI_MINP_STATS"):
            stats.append(line)
        elif line.startswith("CI_MINP_VERIFY"):
            parts = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
            for key in ("calls", "checked", "mismatches"):
                try:
                    verify[key] = int(parts.get(key, verify[key]))
                except ValueError:
                    pass
    return {"installed": installed, "stats": stats[-4:], "verify": verify}


async def run_session(cfg, arm, block, index, out_dir):
    session_id = "b%d-%s-%02d" % (block, arm, index)
    log_path = out_dir / "logs" / ("session-%s.log" % session_id)
    print("[serve] start %s" % session_id, flush=True)
    wait_port_free(cfg["port"])
    free = gpu_free_mib()
    print("[serve] %s gpu_free_mib=%s" % (session_id, free), flush=True)
    started = utc_now()
    clock = time.perf_counter()
    proc, log = start_server(cfg, arm, log_path)
    phases = {}
    rows = []
    try:
        await wait_health(cfg, proc)
        startup = time.perf_counter() - clock
        print("[serve] %s healthy in %.1fs" % (session_id, startup), flush=True)

        def prompt_set(offset, count):
            generator = np.random.default_rng(cfg["seed_base"] + offset)
            return generator.integers(100, 15000,
                                      size=(count, cfg["prompt_len"])).tolist()

        # Disjoint prompt pools keep the prefix cache from transferring work
        # between the two phases: every prompt appears exactly once, so a hit
        # could only come from the warmup pool.
        phase_prompts = {"nop": prompt_set(0, cfg["requests"]),
                         "minp": prompt_set(1, cfg["requests"])}
        warmup_prompts = prompt_set(2, max(cfg["warmup"], cfg["concurrency"]))
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as warm:
            for round_index in range(cfg["warmup_rounds"]):
                for warm_min_p in (0.0, cfg["min_p"]):
                    await asyncio.gather(*(
                        one_request(warm, cfg, warmup_prompts[i], min_p=warm_min_p,
                                    seed=cfg["seed_base"] + 20000 + i)
                        for i in range(cfg["concurrency"])))
            print("[serve] %s warmup done" % session_id, flush=True)
        phase_order = [("nop", 0.0), ("minp", cfg["min_p"])]
        if cfg["first_phase"] == "minp":
            phase_order.reverse()
        for phase_name, min_p in phase_order:
            rows = []  # flushed per phase, so each jsonl block holds one phase
            phases[phase_name] = await run_phase(cfg, phase_name, min_p,
                                                 phase_prompts[phase_name],
                                                 rows, cfg["requests"], session_id)
            phases[phase_name]["wall_clock"] = utc_now()
            with (out_dir / "requests.jsonl").open("a", encoding="utf-8") as handle:
                for row in rows:
                    row["session_id"] = session_id
                    row["arm"] = arm
                    row["block"] = block
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        ok, reason = True, "ok"
    except Exception as exc:  # noqa: BLE001 - evidence of failure matters
        startup = None
        ok, reason = False, "%s: %s" % (type(exc).__name__, exc)
        traceback.print_exc()
        print("[serve] %s FAILED: %s" % (session_id, reason), flush=True)
    finally:
        stop_server(proc, log)
        wait_port_free(cfg["port"], timeout=90)
    parsed = parse_log(log_path)
    session = {
        "session_id": session_id,
        "arm": arm,
        "block": block,
        "index": index,
        "ok": ok,
        "reason": reason,
        "started_at": started,
        "startup_seconds": startup,
        "phases": {name: {k: v for k, v in phase.items() if k != "wall_clock"}
                  for name, phase in phases.items()},
        "installed_markers": parsed["installed"],
        "stats_tail": parsed["stats"],
        "verify": parsed["verify"],
        "gpu_free_mib_before": free,
        "log": str(log_path),
    }
    with (out_dir / "sessions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(session, sort_keys=True) + "\n")
    print("[serve] done %s ok=%s startup=%s phases=%s" %
          (session_id, ok, startup, sorted(phases)), flush=True)
    return session


def load_requests(out_dir):
    rows = []
    path = out_dir / "requests.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def session_metric(phase_metrics, key):
    value = phase_metrics.get(key)
    if isinstance(value, dict):
        return value.get("median")
    return value


def summarize(sessions, requests, cfg):
    arms = {}
    for arm in ("base", "patched"):
        arm_sessions = [s for s in sessions if s["arm"] == arm and s["ok"]]
        phases = {}
        for phase in ("nop", "minp"):
            phase_sessions = [s["phases"].get(phase) for s in arm_sessions]
            phase_sessions = [p for p in phase_sessions if p]
            pooled = [r for r in requests if r["arm"] == arm and r["phase"] == phase]
            metrics = {}
            for key in ("ttft_ms", "tpot_ms", "itl_median_ms", "total_ms"):
                metrics[key] = describe([r[key] for r in pooled])
            metrics["session_medians"] = {
                key: [p[key]["median"] for p in phase_sessions if p[key]["n"]]
                for key in ("ttft_ms", "tpot_ms", "total_ms")
            }
            metrics["requests_per_second"] = {
                "per_session": [p["requests_per_second"] for p in phase_sessions],
                "median": statistics.median([p["requests_per_second"] for p in phase_sessions])
                if phase_sessions else None,
            }
            metrics["tokens_per_second"] = {
                "per_session": [p["tokens_per_second"] for p in phase_sessions],
                "median": statistics.median([p["tokens_per_second"] for p in phase_sessions])
                if phase_sessions else None,
            }
            phases[phase] = metrics
        arms[arm] = {
            "sessions": len(arm_sessions),
            "startup_seconds": describe([s["startup_seconds"] for s in arm_sessions
                                         if s["startup_seconds"]]),
            "phases": phases,
        }
    paired = []
    blocks = sorted({s["block"] for s in sessions if s["ok"]})
    for block in blocks:
        entry = {"block": block}
        for phase in ("nop", "minp"):
            deltas = {}
            for key in ("ttft_ms", "tpot_ms", "total_ms", "tokens_per_second"):
                base_values = [session_metric(s["phases"][phase], key) for s in sessions
                               if s["ok"] and s["block"] == block and s["arm"] == "base"
                               and phase in s["phases"]]
                patched_values = [session_metric(s["phases"][phase], key) for s in sessions
                                  if s["ok"] and s["block"] == block and s["arm"] == "patched"
                                  and phase in s["phases"]]
                base_values = [v for v in base_values if v is not None]
                patched_values = [v for v in patched_values if v is not None]
                if base_values and patched_values:
                    b = statistics.median(base_values)
                    p = statistics.median(patched_values)
                    deltas[key] = {"base": b, "patched": p,
                                   "delta_pct": (p - b) / b * 100.0 if b else None}
            entry[phase] = deltas
        paired.append(entry)
    verify_mismatches = sum(s["verify"]["mismatches"] for s in sessions)
    checks = {
        "sessions_total": len(sessions),
        "sessions_ok": sum(1 for s in sessions if s["ok"]),
        "patched_installs": sum(s["installed_markers"] for s in sessions if s["arm"] == "patched"),
        "base_installs": sum(s["installed_markers"] for s in sessions if s["arm"] == "base"),
        "verify_mismatches": verify_mismatches,
        "paired_blocks": len(blocks),
    }
    passed = (checks["sessions_ok"] == checks["sessions_total"]
              and checks["sessions_total"] > 0
              and checks["patched_installs"] > 0
              and checks["base_installs"] == 0
              and verify_mismatches == 0)
    summary = {
        "created_at": utc_now(),
        "config": cfg,
        "checks": checks,
        "arms": arms,
        "paired": paired,
        "verdict": "passed" if passed else "failed",
    }
    return passed, summary


def token_agreement(requests):
    seen = {}
    agree = total = 0
    for row in requests:
        key = (row.get("block"), row["phase"], row["index"])
        seen.setdefault(key, {})[row["arm"]] = row["text_sha1"]
    for arms in seen.values():
        if "base" in arms and "patched" in arms:
            total += 1
            agree += int(arms["base"] == arms["patched"])
    return {"agree": agree, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-p", type=float, default=0.1)
    # nop-first is the default; minp-first is the control for phase-order drift
    # (thermal/clock ramp), because the second phase always runs on a hotter GPU.
    parser.add_argument("--first-phase", choices=("nop", "minp"), default="nop")
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="qwen3")
    parser.add_argument("--patch-module", required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--stats-every", type=int, default=256)
    parser.add_argument("--verify-first", type=int, default=48)
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(args).items() if k != "out"}
    cfg["model"] = str(Path(args.model).resolve()) if args.model == DEFAULT_MODEL else args.model
    cfg["patch_module"] = str(Path(args.patch_module).resolve())
    cfg["created_at"] = utc_now()
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    sessions = []
    order = ("base", "patched", "patched", "base")
    for block in range(args.blocks):
        for index, arm in enumerate(order):
            session = asyncio.run(run_session(cfg, arm, block, index, out_dir))
            sessions.append(session)
            if not session["ok"]:
                print("[serve] aborting after failure", flush=True)
                (out_dir / "summary.json").write_text(
                    json.dumps({"verdict": "failed", "config": cfg}, indent=2))
                (out_dir / "status.json").write_text(json.dumps(
                    {"passed": False, "reason": session["reason"]}, indent=2))
                return 1
    passed, summary = summarize(sessions, load_requests(out_dir), cfg)
    summary["token_agreement"] = token_agreement(load_requests(out_dir))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps(
        {"passed": passed, "checks": summary["checks"]}, indent=2), encoding="utf-8")
    print(json.dumps(summary["checks"], indent=2), flush=True)
    print(json.dumps(summary["paired"], indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
