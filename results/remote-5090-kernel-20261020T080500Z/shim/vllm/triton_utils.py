# Minimal stand-in for vllm.triton_utils on hosts without vLLM.
import triton
from triton import language as tl

__all__ = ["tl", "triton"]
