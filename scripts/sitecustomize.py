"""Install an alternative min-p kernel into a serving vLLM process at import time.

Python imports ``sitecustomize`` in every interpreter that inherits this
directory on ``PYTHONPATH``.  When ``CI_MINP_PATCH`` names a module that exposes
``apply_min_p``, this hook replaces the module-level function that
``SamplingStates.apply_min_p`` calls once ``vllm.v1.worker.gpu.sample.states``
has been imported.  ``vllm-src`` is never edited; unsetting the variable gives
the pure upstream kernel.

Diagnostics are printed to the process stdout (captured by the server log) and,
when ``CI_MINP_MARKER_FILE`` is set, appended to that file as well, because
worker output does not always reach the parent's stdout:

``CI_MINP_INSTALLED pid=... module=...``
    the candidate was installed in this process.
``CI_MINP_VERIFY calls=... checked=... mismatches=... shape=...``
    with ``CI_MINP_VERIFY_FIRST`` set, the first N calls run both kernels and
    compare the filtered logits byte for byte.
``CI_MINP_STATS calls=... kernels={...}``
    launch counts, printed every ``CI_MINP_STATS_EVERY`` calls (default 256).
"""

import builtins
import os
import sys

_TARGET = "vllm.v1.worker.gpu.sample.states"
_PATCH = os.environ.get("CI_MINP_PATCH")
_MARKER = os.environ.get("CI_MINP_MARKER_FILE")
_VERIFY_FIRST = int(os.environ.get("CI_MINP_VERIFY_FIRST", "0"))
_STATS_EVERY = int(os.environ.get("CI_MINP_STATS_EVERY", "256"))

_state = {"installed": False, "installing": False, "calls": 0, "checked": 0,
          "mismatches": 0, "kernels": {}, "marker": None}


def _emit(message):
    line = "%s pid=%d" % (message, os.getpid())
    print(line, flush=True)
    if not _MARKER:
        return
    try:
        if _state["marker"] is None:
            _state["marker"] = open(_MARKER, "a", encoding="utf-8", buffering=1)
        _state["marker"].write(line + "\n")
    except OSError:
        _state["marker"] = None


class _LaunchCounter:
    """Wraps a triton JIT kernel to count launches without changing behaviour."""

    def __init__(self, kernel, name, counts):
        self.kernel = kernel
        self.name = name
        self.counts = counts

    def __getitem__(self, grid):
        self.counts[self.name] += 1
        return self.kernel[grid]


def _load_candidate(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ci_minp_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install(states):
    candidate_module = _load_candidate(_PATCH)
    for name in ("_min_p_kernel", "_min_p_partial_max", "_min_p_apply_threshold"):
        kernel = getattr(candidate_module, name, None)
        if kernel is None:
            continue
        _state["kernels"][name] = 0
        setattr(candidate_module, name,
                _LaunchCounter(kernel, name, _state["kernels"]))
    candidate = candidate_module.apply_min_p
    original = states.apply_min_p

    def dispatch(logits, mapping, min_p):
        _state["calls"] += 1
        if _VERIFY_FIRST and _state["checked"] < _VERIFY_FIRST:
            import torch

            reference = logits.clone()
            original(reference, mapping, min_p)
            candidate(logits, mapping, min_p)
            equal = torch.equal(reference.view(torch.uint8), logits.view(torch.uint8))
            _state["checked"] += 1
            _state["mismatches"] += int(not equal)
            _emit("CI_MINP_VERIFY calls=%d checked=%d mismatches=%d shape=%s"
                  % (_state["calls"], _state["checked"], _state["mismatches"],
                     tuple(logits.shape)))
        else:
            candidate(logits, mapping, min_p)
        if _STATS_EVERY and _state["calls"] % _STATS_EVERY == 0:
            _emit("CI_MINP_STATS calls=%d kernels=%s"
                  % (_state["calls"], _state["kernels"]))

    states.apply_min_p = dispatch
    _state["installed"] = True
    _emit("CI_MINP_INSTALLED module=%s" % _PATCH)


if _PATCH:
    _original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _original_import(name, globals, locals, fromlist, level)
        if not _state["installed"] and not _state["installing"]:
            target = sys.modules.get(_TARGET)
            # The module is registered before its body finishes executing, so
            # only patch once the attribute it should expose actually exists.
            if target is not None and callable(getattr(target, "apply_min_p", None)):
                _state["installing"] = True
                try:
                    _install(target)
                finally:
                    _state["installing"] = False
        return module

    builtins.__import__ = _import
