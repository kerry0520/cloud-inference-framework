"""Local worker extension for controlled diagnostics and candidate selection."""

from pathlib import Path


class ProjectWorkerExtension:
    def probe_install(self):
        model = self.model_runner.model
        self._probe_rows = {}
        self._probe_handles = []
        metadata = []
        for name, module in model.named_modules():
            if not (name.startswith("model.layers.0.") and name.endswith(
                    ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"))):
                continue
            metadata.append({"name": name, "class": type(module).__name__,
                             "weight_shape": list(module.weight.shape),
                             "weight_dtype": str(module.weight.dtype)})

            def capture(mod, inputs, label=name):
                x = inputs[0]
                key = (label, tuple(x.shape))
                self._probe_rows[key] = {
                    "name": label, "input_shape": list(x.shape),
                    "input_stride": list(x.stride()),
                    "weight_shape": list(mod.weight.shape),
                    "dtype": str(x.dtype), "calls":
                    self._probe_rows.get(key, {}).get("calls", 0) + 1,
                }
            self._probe_handles.append(module.register_forward_pre_hook(capture))
        return {"linears": metadata, "lm_head_weight": list(model.lm_head.weight.shape),
                "lm_head_class": type(model.lm_head).__name__}

    def probe_collect(self):
        for handle in self._probe_handles:
            handle.remove()
        return list(self._probe_rows.values())

    def minp_install(self, mode, evidence_dir=None):
        from kernels.min_p import apply_min_p_optimized as candidate

        return self._install_minp(candidate, mode, evidence_dir)

    def minp_install_file(self, mode, module_path, evidence_dir=None):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("mrv2_min_p_patched",
                                                      module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        launched = {}

        class _LaunchCounter:
            def __init__(self, kernel, name, counts):
                self.kernel = kernel
                self.name = name
                self.counts = counts

            def __getitem__(self, grid):
                self.counts[self.name] += 1
                return self.kernel[grid]

        for name in ("_min_p_kernel", "_min_p_partial_max", "_min_p_apply_threshold"):
            kernel = getattr(module, name, None)
            if kernel is None:
                continue
            launched[name] = 0
            setattr(module, name, _LaunchCounter(kernel, name, launched))
        result = self._install_minp(module.apply_min_p, mode, evidence_dir)
        self._minp_stats["kernels"] = launched
        result["kernels"] = launched
        return result

    def _install_minp(self, candidate, mode, evidence_dir=None):
        import torch
        import vllm.v1.worker.gpu.sample.states as states

        if not hasattr(self, "_original_minp"):
            self._original_minp = states.apply_min_p
        original = self._original_minp
        self._minp_stats = {"mode": mode, "calls": 0, "checked": 0,
                            "mismatches": 0, "shapes": {}}
        assert mode in ("base", "candidate", "verify")
        self._minp_candidate = candidate
        self._minp_evidence_dir = evidence_dir
        self._minp_mode = mode

        def dispatch(logits, mapping, min_p):
            stats = self._minp_stats
            mode_now = self._minp_mode
            stats["mode"] = mode_now
            stats["calls"] += 1
            key = str(tuple(logits.shape))
            stats["shapes"][key] = stats["shapes"].get(key, 0) + 1
            if mode_now == "base":
                original(logits, mapping, min_p)
                return
            if mode_now == "verify":
                evidence_dir_now = self._minp_evidence_dir
                before = logits.clone()
                reference = logits.clone()
                original(reference, mapping, min_p)
                self._minp_candidate(logits, mapping, min_p)
                equal = torch.equal(reference.view(torch.uint8), logits.view(torch.uint8))
                stats["checked"] += 1
                stats["mismatches"] += int(not equal)
                if evidence_dir_now is not None and stats["checked"] <= 4:
                    folder = Path(evidence_dir_now)
                    folder.mkdir(parents=True, exist_ok=True)
                    torch.save({"before": before.cpu(), "reference": reference.cpu(),
                                "candidate": logits.cpu(), "mapping": mapping.cpu(),
                                "min_p": min_p.cpu()},
                               folder / f"real-call-{stats['checked']}.pt")
                assert equal, "Real min-p invocation disagreed with upstream"
            else:
                self._minp_candidate(logits, mapping, min_p)

        states.apply_min_p = dispatch
        return {"mode": mode, "installed_in": states.__name__}

    def minp_select(self, mode):
        """Switch between the two already-installed arms without re-importing."""
        assert mode in ("base", "candidate"), mode
        assert hasattr(self, "_minp_candidate"), "no candidate installed"
        self._minp_mode = mode
        self._minp_stats["mode"] = mode
        return {"selected": mode}

    def minp_stats(self):
        return self._minp_stats
