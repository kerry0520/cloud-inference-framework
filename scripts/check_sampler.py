import json
import torch
import flashinfer

logits = torch.arange(512, device="cuda", dtype=torch.float32).reshape(2, 256)
result = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, 10, 0.9)
torch.cuda.synchronize()
assert result.shape == (2,)
assert ((result >= 0) & (result < 256)).all().item()
print(json.dumps({"flashinfer_sampling_check": "passed", "tokens": result.tolist()}))
