"""CPU source-level counterexamples, not GPU or full-framework reproductions.

Run only reviewed, saved source snapshots. Optional CuTe metadata and watchdog
construction are mocked to inspect native-arch selection and scheduler wiring.
No upstream code is edited and no packages are installed.
"""

import argparse
import ast
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def function_from_source(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    snippet = "from __future__ import annotations\n" + ast.unparse(node)
    exec(compile(snippet, str(path), "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    loader = importlib.machinery.SourceFileLoader(
        "reviewed_availability", str(args.source / "flashinfer-availability.snapshot"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    availability = importlib.util.module_from_spec(spec)
    loader.exec_module(availability)
    for name in ("cutlass", "cutlass.base_dsl", "cutlass.base_dsl.arch", "cutlass.cutlass_dsl"):
        sys.modules[name] = ModuleType(name)
    sys.modules["cutlass.base_dsl.arch"].Arch = {"sm_120a": object(), "sm_90a": object()}
    target = SimpleNamespace(arch="sm_90a")
    sys.modules["cutlass.cutlass_dsl"].CuTeDSL = SimpleNamespace(
        _get_dsl=lambda: SimpleNamespace(envar=target))
    availability.is_cute_dsl_available = lambda: True
    native_rows = []
    for captured in ("sm_120a", "sm_90a"):
        target.arch = captured
        availability.is_cute_dsl_arch_supported.cache_clear()
        supported = availability.is_cute_dsl_arch_supported(12, 0)
        native_rows.append(dict(device="sm_120", captured=availability._dsl_captured_arch(),
                                reported_supported=supported,
                                cross_major_mismatch=captured == "sm_90a"))
    assert all(row["reported_supported"] for row in native_rows)
    # Exercise the unmodified norm dispatcher body with the native helper.
    namespace = dict(_USE_CUDA_NORM=False,
                     _cute_dsl_supports_arch=availability.is_cute_dsl_arch_supported,
                     torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda device: (12, 0))))
    choose_cuda = function_from_source(args.source / "flashinfer-norm.snapshot", "_use_cuda_norm", namespace)
    fallback = choose_cuda("mock cuda:0")
    assert fallback is False

    namespace = dict(WatchdogRaw=lambda **kwargs: SimpleNamespace(**kwargs))
    create = function_from_source(args.source / "sglang-invariant.snapshot", "create_scheduler_watchdog", namespace)
    paused_rows = []
    for paused, busy in ((False, False), (False, True), (True, True)):
        scheduler = SimpleNamespace(is_initializing=False, cur_batch_for_debug=object() if busy else None,
                                    forward_ct=0, _engine_paused=paused)
        watchdog = create(scheduler, watchdog_timeout=1)
        paused_rows.append(dict(paused=paused, busy=busy, active=watchdog.is_active()))
    assert [row["active"] for row in paused_rows] == [False, True, True]
    report = dict(scope="CPU source wiring only; mocks do not establish a live CUDA crash or watchdog timeout",
                  native_arch_probe=native_rows, norm_cuda_fallback_with_mismatch=fallback,
                  scheduler_watchdog=paused_rows)
    (args.out / "probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
