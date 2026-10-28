"""Step-level probe: where does the extra ~1 ms/step with ``min_p > 0`` go?

The call-level probe (``mrv3_minp_probe.py``) showed the min-p kernel itself
costs ~0.08 ms per call in the served process, yet turning ``min_p`` on adds
~1.1 ms per decode step.  This probe brackets ``Sampler.sample`` (the per-step
sampling entry point) with CUDA events and tags each step by whether the min-p
kernel ran, so the two phases can be compared step by step.

``Sampler`` cannot be imported at probe-load time (``states`` is imported from
inside ``sampler``, so the class does not exist yet), hence the lazy install on
the first ``apply_min_p`` call - which the warmup already triggers.  Run with
``--first-phase minp`` so both phases are recorded after installation.

Outputs ``<CI_PROBE_OUT>.<pid>.json``: step counts and GPU/host time per step,
split by tag.  Evidence only; not used by the project patches.
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import sys
import time

import torch

from vllm.v1.worker.gpu.sample.min_p import apply_min_p as _upstream

OUT = os.environ.get("CI_PROBE_OUT", "")
EVERY = int(os.environ.get("CI_PROBE_EVERY", "64"))
_PATH = "%s.%d.json" % (OUT, os.getpid()) if OUT else ""

_CALLS: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_STEPS: list[tuple[torch.cuda.Event, torch.cuda.Event, float, bool]] = []
_TAG = {"minp": False}
_INSTALLED = {"done": False}


def _install() -> None:
    if _INSTALLED["done"]:
        return
    module = sys.modules.get("vllm.v1.worker.gpu.sample.sampler")
    sampler_cls = getattr(module, "Sampler", None) if module else None
    if sampler_cls is None or getattr(sampler_cls.sample, "_ci_probe", False):
        return
    original = sampler_cls.sample

    def sample(self, *args, **kwargs):
        had = _TAG["minp"]
        _TAG["minp"] = False
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        out = original(self, *args, **kwargs)
        t1 = time.perf_counter()
        end.record()
        _STEPS.append((start, end, (t1 - t0) * 1e6, had))
        if len(_STEPS) % EVERY == 0:
            _write()
        return out

    sample._ci_probe = True
    sampler_cls.sample = sample
    _INSTALLED["done"] = True
    print("CI_STEP_PROBE installed", flush=True)


def apply_min_p(logits, expanded_idx_mapping, min_p):
    _install()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _upstream(logits, expanded_idx_mapping, min_p)
    end.record()
    _CALLS.append((start, end))
    _TAG["minp"] = True
    return None


def _write() -> None:
    if not _PATH:
        return
    groups = {}
    for tag, wanted in (("minp", True), ("nop", False)):
        gpu = []
        host = []
        for start, end, host_us, had in _STEPS:
            if had is not wanted or not end.query():
                continue
            gpu.append(start.elapsed_time(end))
            host.append(host_us)
        if not gpu:
            continue
        groups[tag] = {
            "steps": len(gpu),
            "gpu_ms_mean": statistics.fmean(gpu),
            "gpu_ms_p50": statistics.median(gpu),
            "host_us_mean": statistics.fmean(host),
        }
    call_us = [s.elapsed_time(e) * 1e3 for s, e in _CALLS if e.query()]
    summary = {
        "pid": os.getpid(),
        "installed": _INSTALLED["done"],
        "steps_total": len(_STEPS),
        "minp_calls": len(_CALLS),
        "minp_call_us_mean": statistics.fmean(call_us) if call_us else None,
        "groups": groups,
    }
    with open(_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _on_term(signum, frame):  # noqa: ARG001
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    _write()
    raise SystemExit(0)


for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _on_term)
    except Exception:  # noqa: BLE001
        pass
