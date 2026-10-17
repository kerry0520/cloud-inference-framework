import json
import torch
from vllm.platforms import current_platform

host = torch.arange(4096, dtype=torch.int32).pin_memory()
device = host.to("cuda", non_blocking=True)
torch.cuda.synchronize()
torch.testing.assert_close(device.cpu(), host)
print(json.dumps({"host_is_pinned": host.is_pinned(),
                  "framework_pin_memory_available": current_platform.is_pin_memory_available(),
                  "async_copy_check": "passed"}, indent=2))
