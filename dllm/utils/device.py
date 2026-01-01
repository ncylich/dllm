"""
Device detection utilities for TPU/GPU/CPU compatibility.
"""

import torch

# Lazy import for torch_xla to avoid import errors when not installed
_xla_available = None


def is_xla_available() -> bool:
    """Check if torch_xla is available."""
    global _xla_available
    if _xla_available is None:
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401

            _xla_available = True
        except ImportError:
            _xla_available = False
    return _xla_available


def is_tpu_available() -> bool:
    """Check if running on TPU (XLA device available)."""
    if not is_xla_available():
        return False
    try:
        import torch_xla.core.xla_model as xm

        # Try to get an XLA device - this will succeed on TPU
        device = xm.xla_device()
        return device is not None
    except Exception:
        return False


def get_device(local_rank: int = 0) -> torch.device:
    """
    Get the appropriate device for the current environment.

    Priority: TPU > CUDA > CPU

    Args:
        local_rank: Local process index for multi-device setups.

    Returns:
        torch.device for the current environment.
    """
    if is_tpu_available():
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")


def get_device_string(local_rank: int = 0) -> str:
    """
    Get device string for the current environment.

    Returns:
        Device string like "xla", "cuda:0", or "cpu".
    """
    if is_tpu_available():
        return "xla"
    elif torch.cuda.is_available():
        return f"cuda:{local_rank}"
    else:
        return "cpu"


def empty_cache() -> None:
    """Clear device memory cache (works for both CUDA and TPU)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm

            xm.mark_step()  # TPU equivalent - sync and potentially free memory
        except Exception:
            pass


def synchronize() -> None:
    """Synchronize device operations."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm

            xm.mark_step()
        except Exception:
            pass
