"""Local worker extension for controlled diagnostics and candidate selection."""

import json
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
        import torch
        import vllm.v1.worker.gpu.sample.states as states
        from kernels.min_p import apply_min_p_optimized as candidate

        if not hasattr(self, "_original_minp"):
            self._original_minp = states.apply_min_p
        original = self._original_minp
        self._minp_stats = {"mode": mode, "calls": 0, "checked": 0,
                            "mismatches": 0, "shapes": {}}
        assert mode in ("base", "candidate", "verify")

        def dispatch(logits, mapping, min_p):
            stats = self._minp_stats
            stats["calls"] += 1
            key = str(tuple(logits.shape))
            stats["shapes"][key] = stats["shapes"].get(key, 0) + 1
            if mode == "base":
                original(logits, mapping, min_p)
                return
            if mode == "verify":
                before = logits.clone()
                reference = logits.clone()
                original(reference, mapping, min_p)
                candidate(logits, mapping, min_p)
                equal = torch.equal(reference.view(torch.uint8), logits.view(torch.uint8))
                stats["checked"] += 1
                stats["mismatches"] += int(not equal)
                if evidence_dir is not None and stats["checked"] <= 4:
                    folder = Path(evidence_dir)
                    folder.mkdir(parents=True, exist_ok=True)
                    torch.save({"before": before.cpu(), "reference": reference.cpu(),
                                "candidate": logits.cpu(), "mapping": mapping.cpu(),
                                "min_p": min_p.cpu()},
                               folder / f"real-call-{stats['checked']}.pt")
                assert equal, "Real min-p invocation disagreed with upstream"
            else:
                candidate(logits, mapping, min_p)

        states.apply_min_p = dispatch
        return {"mode": mode, "installed_in": states.__name__}

    def minp_stats(self):
        return self._minp_stats
