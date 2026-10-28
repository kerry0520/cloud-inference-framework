"""Diagnostic probe for the "min_p=0.1 costs ~1.2 ms/step" observation.

This module is loaded by ``sitecustomize.py`` into a worker process and replaces
``vllm.v1.worker.gpu.sample.states.apply_min_p`` (the module global the sampler
calls).  It keeps the upstream kernel as the implementation and only measures
or perturbs the call:

- ``CI_PROBE_MODE=time``   call the upstream kernel once, record CUDA-event
  duration and host-side launch time per call.
- ``CI_PROBE_MODE=skip``   do not call the kernel at all (logits are NOT
  filtered). The resulting text is wrong on purpose; only the timing is used.
- ``CI_PROBE_MODE=double`` call the kernel twice (idempotent: the second call
  sees the already-thresholded logits) to measure the marginal cost of one
  extra call.

``CI_PROBE_OUT`` names the summary file; the pid is appended so several server
processes can run in the same experiment.  The summary is rewritten every
``CI_PROBE_EVERY`` calls (default 64) and on SIGTERM, because the harness stops
servers with SIGTERM (atexit alone would never run).  Nothing here is used by the
project patches (``patches/mrv2_min_p*.py``); it is evidence only.
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import time

import torch

from vllm.v1.worker.gpu.sample.min_p import apply_min_p as _upstream

MODE = os.environ.get("CI_PROBE_MODE", "time")
OUT = os.environ.get("CI_PROBE_OUT", "")
EVERY = int(os.environ.get("CI_PROBE_EVERY", "64"))
_CALLS: list[tuple[torch.cuda.Event, torch.cuda.Event, float]] = []
_PATH = "%s.%d.json" % (OUT, os.getpid()) if OUT else ""


def _write(force: bool = False) -> None:
    if not _PATH:
        return
    gpu_us = []
    for start, end, _ in _CALLS:
        if force or end.query():
            gpu_us.append(start.elapsed_time(end) * 1e3)
    host_us = [h for _, _, h in _CALLS]
    summary = {
        "mode": MODE,
        "pid": os.getpid(),
        "calls": len(_CALLS),
        "timed": len(gpu_us),
        "gpu_us_mean": statistics.fmean(gpu_us) if gpu_us else None,
        "gpu_us_p50": statistics.median(gpu_us) if gpu_us else None,
        "gpu_us_total": sum(gpu_us) if gpu_us else None,
        "host_us_mean": statistics.fmean(host_us) if host_us else None,
        "host_us_total": sum(host_us) if host_us else None,
    }
    with open(_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def apply_min_p(logits, expanded_idx_mapping, min_p):
    if MODE == "skip":
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.perf_counter()
    _upstream(logits, expanded_idx_mapping, min_p)
    if MODE == "double":
        _upstream(logits, expanded_idx_mapping, min_p)
    t1 = time.perf_counter()
    end.record()
    _CALLS.append((start, end, (t1 - t0) * 1e6))
    if len(_CALLS) % EVERY == 0:
        _write()
    return None


def _on_term(signum, frame):  # noqa: ARG001 - signal signature
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - teardown races are not interesting
        pass
    _write(force=True)
    print("CI_PROBE %s" % json.dumps({"mode": MODE, "calls": len(_CALLS)}), flush=True)
    raise SystemExit(0)


for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _on_term)
    except Exception:  # noqa: BLE001 - not the main thread (unlikely here)
        pass
