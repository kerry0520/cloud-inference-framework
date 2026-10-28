"""Per-op probe inside one sampling step.

The step-level probe showed ``Sampler.sample`` burns ~1.3 ms of GPU time per
step when ``min_p=0.1`` is set versus ~0.4 ms when it is 0, while the min-p
kernel itself measures only ~0.08 ms.  This probe brackets the four ops that
run inside a step (temperature, min-p, top-k/top-p, gumbel sampling) so the
missing millisecond can be attributed to one of them.

Installed lazily on the first ``apply_min_p`` call because ``sampler`` imports
``states`` and its class is not defined yet at probe-load time.  Run with
``--first-phase minp`` so both phases are measured after installation.

Outputs ``<CI_PROBE_OUT>.<pid>.json``.  Evidence only.
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import sys
import time

import torch

from vllm.v1.worker.gpu.sample import states as _states
from vllm.v1.worker.gpu.sample.min_p import apply_min_p as _min_p_impl

OUT = os.environ.get("CI_PROBE_OUT", "")
EVERY = int(os.environ.get("CI_PROBE_EVERY", "64"))
_PATH = "%s.%d.json" % (OUT, os.getpid()) if OUT else ""

_RECORDS: list[tuple[str, torch.cuda.Event, torch.cuda.Event, float, bool]] = []
_CUR = {"had": False, "steps": 0}
_STEP_TAG = {"minp": False}
_INSTALLED = {"done": False}


def _wrap(op_name: str, original):
    def wrapped(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        t1 = time.perf_counter()
        end.record()
        _RECORDS.append((op_name, start, end, (t1 - t0) * 1e6, _CUR["had"]))
        return out

    return wrapped


def _install() -> None:
    if _INSTALLED["done"]:
        return
    module = sys.modules.get("vllm.v1.worker.gpu.sample.sampler")
    sampler_cls = getattr(module, "Sampler", None) if module else None
    if sampler_cls is None:
        return
    for name in ("gumbel_sample", "apply_top_k_top_p"):
        if hasattr(module, name) and not getattr(getattr(module, name), "_ci_ops", False):
            fn = _wrap("sampler." + name, getattr(module, name))
            fn._ci_ops = True
            setattr(module, name, fn)
    if not getattr(_states.SamplingStates.apply_temperature, "_ci_ops", False):
        meth = _wrap("states.apply_temperature", _states.SamplingStates.apply_temperature)
        meth._ci_ops = True
        _states.SamplingStates.apply_temperature = meth
    if not getattr(sampler_cls.sample, "_ci_ops", False):
        original = sampler_cls.sample

        def sample(self, *args, **kwargs):
            _CUR["had"] = _STEP_TAG["minp"]
            _STEP_TAG["minp"] = False
            _CUR["steps"] += 1
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            t0 = time.perf_counter()
            out = original(self, *args, **kwargs)
            t1 = time.perf_counter()
            end.record()
            _RECORDS.append(("sample", start, end, (t1 - t0) * 1e6, _CUR["had"]))
            if _CUR["steps"] % EVERY == 0:
                _write()
            return out

        sample._ci_ops = True
        sampler_cls.sample = sample
    _INSTALLED["done"] = True
    print("CI_OPS_PROBE installed", flush=True)


def apply_min_p(logits, expanded_idx_mapping, min_p):
    _install()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.perf_counter()
    _min_p_impl(logits, expanded_idx_mapping, min_p)
    t1 = time.perf_counter()
    end.record()
    _RECORDS.append(("minp", start, end, (t1 - t0) * 1e6, _CUR["had"]))
    _STEP_TAG["minp"] = True
    return None


def _write() -> None:
    if not _PATH:
        return
    groups: dict[str, dict[str, dict[str, float]]] = {}
    for op, start, end, host_us, had in _RECORDS:
        if not end.query():
            continue
        bucket = groups.setdefault(op, {"minp": {}, "nop": {}})
        tag = "minp" if had else "nop"
        entry = bucket[tag]
        entry.setdefault("_gpu", []).append(start.elapsed_time(end))
        entry.setdefault("_host", []).append(host_us)
    out = {}
    for op, bucket in groups.items():
        out[op] = {}
        for tag, entry in bucket.items():
            gpu = entry.get("_gpu", [])
            if not gpu:
                continue
            out[op][tag] = {
                "ops": len(gpu),
                "gpu_ms_mean": statistics.fmean(gpu),
                "gpu_ms_p50": statistics.median(gpu),
                "host_us_mean": statistics.fmean(entry["_host"]),
            }
    with open(_PATH, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "installed": _INSTALLED["done"], "ops": out}, handle, indent=2)


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
