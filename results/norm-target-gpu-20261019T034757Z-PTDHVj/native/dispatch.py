def _use_cuda_norm(device: torch.device) -> bool:
    """Return ``True`` when the CUDA JIT norm kernels should be used instead
    of the CuTe DSL ones.

    Besides the explicit ``FLASHINFER_USE_CUDA_NORM`` opt-in, this covers
    devices whose architecture the installed CuTe DSL cannot target (e.g.
    Rubin/sm_107 against a DSL without ``sm_107a``, unless
    ``CUTE_DSL_ARCH=sm_100f`` was exported before the process started).
    The CUDA JIT kernels are functionally equivalent, so norm falls back
    instead of raising like the DSL-only kernels do."""
    if _USE_CUDA_NORM:
        return True
    return not _cute_dsl_supports_arch(*torch.cuda.get_device_capability(device))
