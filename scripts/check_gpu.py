import json
import torch
import vllm

a = torch.arange(256, device="cuda", dtype=torch.float32).reshape(16, 16)
actual = (a @ a.T).cpu()
expected = a.cpu() @ a.cpu().T
torch.testing.assert_close(actual, expected)
print(json.dumps({"torch": torch.__version__, "vllm": vllm.__version__,
                  "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(),
                  "capability": torch.cuda.get_device_capability(),
                  "matmul_check": "passed"}, indent=2))
