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
    """Check if running on TPU (XLA device available).

    Detects TPU without initializing the XLA runtime, which is required for
    compatibility with xmp.spawn(). Uses multiple detection methods:
    1. PJRT_DEVICE env var (set after xmp.spawn or by user)
    2. libtpu.so presence (hardware detection without runtime init)
    3. /dev/accel* devices (TPU v2-v5 hardware device nodes)
    4. /dev/vfio/* devices (TPU v6 PJRT hardware device nodes)
    """
    import os

    # Method 1: Check PJRT_DEVICE env var (set after spawn or manually)
    if os.environ.get("PJRT_DEVICE") == "TPU":
        return True

    # Method 2: Check for libtpu.so (TPU hardware) without initializing runtime
    # This mirrors what torch_xla does internally for auto-detection
    try:
        import ctypes

        ctypes.CDLL("libtpu.so")
        return True
    except OSError:
        pass

    # Method 3: Check for /dev/accel* devices (TPU v2-v5)
    try:
        dev_path = "/dev"
        if os.path.isdir(dev_path):
            for entry in os.listdir(dev_path):
                if entry.startswith("accel"):
                    return True
    except (OSError, PermissionError):
        pass

    # Method 4: Check for /dev/vfio/* devices (TPU v6 uses VFIO-based PJRT)
    # TPU v6 devices appear as /dev/vfio/0, /dev/vfio/1, etc.
    try:
        vfio_path = "/dev/vfio"
        if os.path.isdir(vfio_path):
            vfio_devices = [f for f in os.listdir(vfio_path) if f.isdigit()]
            if vfio_devices:
                return True
    except (OSError, PermissionError):
        pass

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


def wrap_model_xla_fsdp(
    model: torch.nn.Module,
    transformer_layer_cls: type | None = None,
    auto_wrap_policy: str = "transformer",
    reshard_after_forward: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = False,
) -> torch.nn.Module:
    """
    Wrap a model with XLA Fully Sharded Data Parallel (FSDP).

    This enables ZeRO-3 style sharding of parameters, gradients, and optimizer
    states across TPU cores, allowing training of larger models.

    Args:
        model: The model to wrap.
        transformer_layer_cls: The transformer layer class to wrap (e.g., ModernBertEncoderLayer).
            If None, wraps the entire model.
        auto_wrap_policy: "transformer" for transformer-based wrapping, "size" for size-based.
        reshard_after_forward: If True, enables ZeRO-3 (reshards params after forward).
            If False, uses ZeRO-2 (only shards grads/optimizer states).
        compute_dtype: Dtype for computation (typically bfloat16 on TPU).
        gradient_checkpointing: If True, wrap each FSDP unit with checkpoint_module
            to trade compute for memory (enables larger batch sizes).

    Returns:
        The FSDP-wrapped model.

    Example:
        >>> from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer
        >>> model = wrap_model_xla_fsdp(model, transformer_layer_cls=ModernBertEncoderLayer)
    """
    import os

    if not is_tpu_available():
        # Not on TPU, return model unchanged
        return model

    # Check if XLA FSDP is enabled via environment variable
    xla_fsdp_enabled = os.environ.get("DLLM_XLA_FSDP", "").lower() in ("1", "true", "yes")
    if not xla_fsdp_enabled:
        print("[XLA FSDP] Not enabled (set DLLM_XLA_FSDP=1 to enable)")
        return model

    # Check if gradient checkpointing is enabled via env var
    gradient_checkpointing = gradient_checkpointing or os.environ.get(
        "DLLM_XLA_GRADIENT_CHECKPOINTING", ""
    ).lower() in ("1", "true", "yes")

    try:
        from functools import partial

        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Import checkpoint_module for gradient checkpointing
        checkpoint_module = None
        if gradient_checkpointing:
            try:
                from torch_xla.distributed.fsdp import checkpoint_module
                print("[XLA FSDP] Gradient checkpointing enabled")
            except ImportError:
                print("[XLA FSDP] Warning: checkpoint_module not available, disabling gradient checkpointing")
                gradient_checkpointing = False

        print(f"[XLA FSDP] Wrapping model with FSDP (reshard_after_forward={reshard_after_forward}, gradient_checkpointing={gradient_checkpointing})")

        # Build auto-wrap policy if transformer layer class is provided
        if transformer_layer_cls is not None and auto_wrap_policy == "transformer":
            auto_wrap = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_layer_cls},
            )

            if gradient_checkpointing and checkpoint_module is not None:
                # Wrap each FSDP unit with checkpoint_module for activation checkpointing
                auto_wrapper_callable = lambda m, *args, **kwargs: FSDP(
                    checkpoint_module(m), *args, **kwargs
                )
                wrapped_model = FSDP(
                    model,
                    auto_wrap_policy=auto_wrap,
                    auto_wrapper_callable=auto_wrapper_callable,
                    reshard_after_forward=reshard_after_forward,
                    compute_dtype=compute_dtype,
                )
            else:
                wrapped_model = FSDP(
                    model,
                    auto_wrap_policy=auto_wrap,
                    reshard_after_forward=reshard_after_forward,
                    compute_dtype=compute_dtype,
                )
        else:
            # Wrap entire model
            if gradient_checkpointing and checkpoint_module is not None:
                model = checkpoint_module(model)
            wrapped_model = FSDP(
                model,
                reshard_after_forward=reshard_after_forward,
                compute_dtype=compute_dtype,
            )

        print(f"[XLA FSDP] Model wrapped successfully")
        return wrapped_model

    except ImportError as e:
        print(f"[XLA FSDP] Warning: Could not import XLA FSDP: {e}")
        return model
    except Exception as e:
        print(f"[XLA FSDP] Warning: Failed to wrap model: {e}")
        return model


def patch_torch_checkpoint_for_xla() -> None:
    """
    Patch torch.utils.checkpoint to use XLA's checkpoint implementation on TPU.

    Standard PyTorch checkpoint doesn't work correctly with XLA - it doesn't
    reduce memory. This patches the checkpoint function to use torch_xla's
    implementation which properly handles XLA's lazy execution model.

    Call this once at the start of training on TPU before enabling gradient
    checkpointing on the model.
    """
    if not is_tpu_available():
        return

    try:
        import torch.utils.checkpoint as torch_ckpt
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint

        # Store original for potential restoration
        if not hasattr(torch_ckpt, "_original_checkpoint"):
            torch_ckpt._original_checkpoint = torch_ckpt.checkpoint

        # Replace with XLA version
        torch_ckpt.checkpoint = xla_checkpoint
        print("[XLA] Patched torch.utils.checkpoint to use XLA checkpoint")

    except ImportError as e:
        print(f"[XLA] Warning: Could not patch checkpoint: {e}")


def enable_xla_gradient_checkpointing(model: "torch.nn.Module") -> None:
    """
    Enable gradient checkpointing for a model on TPU.

    This uses torch_xla's checkpoint implementation which properly handles
    XLA's lazy execution model, unlike PyTorch's standard checkpoint.

    Args:
        model: A HuggingFace model that supports gradient checkpointing.
    """
    import os

    if not is_tpu_available():
        return

    # Check if gradient checkpointing is enabled via env var
    if not os.environ.get("DLLM_XLA_GRADIENT_CHECKPOINTING", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return

    try:
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint

        # The key insight: HuggingFace's _set_gradient_checkpointing sets
        # _gradient_checkpointing_func on ALL submodules that have the
        # `gradient_checkpointing` attribute. We need to pass our XLA checkpoint
        # function directly to _set_gradient_checkpointing.
        if hasattr(model, "_set_gradient_checkpointing"):
            # Directly call the internal method with our XLA checkpoint function
            model._set_gradient_checkpointing(
                enable=True, gradient_checkpointing_func=xla_checkpoint
            )
            # Also enable input require grads (normally done by gradient_checkpointing_enable)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("[XLA] Gradient checkpointing enabled with XLA checkpoint")
        elif hasattr(model, "gradient_checkpointing_enable"):
            # Fallback: enable normally, then try to override
            model.gradient_checkpointing_enable()
            # Override on all submodules that have the attribute
            for module in model.modules():
                if hasattr(module, "_gradient_checkpointing_func"):
                    module._gradient_checkpointing_func = xla_checkpoint
            print("[XLA] Gradient checkpointing enabled with XLA checkpoint (fallback)")
        else:
            print("[XLA] Warning: Model does not support gradient_checkpointing_enable()")

    except ImportError as e:
        print(f"[XLA] Warning: Could not enable gradient checkpointing: {e}")


def get_xla_fsdp_layer_cls(model_name_or_path: str) -> type | None:
    """
    Get the appropriate transformer layer class for XLA FSDP auto-wrapping.

    Args:
        model_name_or_path: The model name or path to determine layer class.

    Returns:
        The transformer layer class, or None if unknown.
    """
    model_name_lower = model_name_or_path.lower()

    if "modernbert" in model_name_lower:
        try:
            from transformers.models.modernbert.modeling_modernbert import (
                ModernBertEncoderLayer,
            )
            return ModernBertEncoderLayer
        except ImportError:
            pass
    elif "bert" in model_name_lower:
        try:
            from transformers.models.bert.modeling_bert import BertLayer
            return BertLayer
        except ImportError:
            pass
    elif "llama" in model_name_lower or "llada" in model_name_lower:
        try:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            return LlamaDecoderLayer
        except ImportError:
            pass
    elif "qwen" in model_name_lower:
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            return Qwen2DecoderLayer
        except ImportError:
            pass

    return None
