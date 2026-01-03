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


def enable_xla_scan_layers(model: "torch.nn.Module") -> None:
    """
    Enable scan_layers optimization for transformer models on TPU.

    This replaces the for-loop over encoder/decoder layers with torch_xla's scan
    operator, which compiles only the first layer and reuses it for subsequent
    layers. This significantly reduces compilation time for models with many
    identical layers.

    Enable via DLLM_XLA_SCAN=1 environment variable.

    Args:
        model: A HuggingFace model with encoder/decoder layers.
    """
    import os

    if not is_tpu_available():
        return

    if not os.environ.get("DLLM_XLA_SCAN", "").lower() in ("1", "true", "yes"):
        return

    try:
        from torch_xla.experimental.scan import scan

        # Check if this is a ModernBERT model
        model_cls_name = model.__class__.__name__
        if "ModernBert" in model_cls_name:
            _patch_modernbert_with_scan(model, scan)
        else:
            print(f"[XLA Scan] Model {model_cls_name} not supported for scan optimization")

    except ImportError as e:
        print(f"[XLA Scan] Warning: Could not import scan: {e}")
    except Exception as e:
        print(f"[XLA Scan] Warning: Failed to enable scan: {e}")


def _patch_modernbert_with_scan(model: "torch.nn.Module", scan) -> None:
    """
    Patch ModernBERT to use scan_layers for the encoder layers.

    ModernBERT has 22 layers where layer 0 has Identity() for attn_norm while
    layers 1-21 have LayerNorm. We handle layer 0 separately, then use scan_layers
    for layers 1-21 which are homogeneous.

    scan_layers works by:
    1. Tracing the first layer's forward pass
    2. Reusing that compiled HLO for all subsequent layers
    This reduces compilation time significantly for models with many identical layers.
    """
    import types
    from functools import partial

    try:
        from torch_xla.experimental.scan_layers import scan_layers
    except ImportError:
        print("[XLA Scan] scan_layers not available in this torch_xla version")
        return

    # Find the ModernBertModel inside (could be wrapped in ForMaskedLM etc)
    bert_model = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        bert_model = model.model
    elif hasattr(model, "layers"):
        bert_model = model

    if bert_model is None:
        print("[XLA Scan] Could not find ModernBertModel layers")
        return

    layers = bert_model.layers
    if len(layers) < 2:
        print("[XLA Scan] Not enough layers to benefit from scan")
        return

    # Create a ModuleList of just the homogeneous layers (1-N)
    homogeneous_layers = torch.nn.ModuleList(list(layers)[1:])

    # Store reference to layer 0 and homogeneous layers on the model
    bert_model._layer0 = layers[0]
    bert_model._homogeneous_layers = homogeneous_layers

    # Store original forward
    original_forward = bert_model.forward

    def scan_forward(
        self,
        input_ids=None,
        attention_mask=None,
        sliding_window_mask=None,
        position_ids=None,
        inputs_embeds=None,
        indices=None,
        cu_seqlens=None,
        max_seqlen=None,
        batch_size=None,
        seq_len=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Modified forward that uses scan_layers for layers 1-N."""
        from transformers.modeling_outputs import BaseModelOutput

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # scan_layers doesn't support output_attentions or output_hidden_states
        if output_attentions or output_hidden_states:
            # Fall back to original forward
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                seq_len=seq_len,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if inputs_embeds is None:
                    from transformers.models.modernbert.modeling_modernbert import _unpad_modernbert_input
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask
                        )
                else:
                    from transformers.models.modernbert.modeling_modernbert import _unpad_modernbert_input
                    inputs_embeds, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask
                    )
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            attention_mask, sliding_window_mask = self._update_attention_mask(
                attention_mask, output_attentions=False
            )

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        # Run layer 0 separately (has different structure - Identity vs LayerNorm)
        layer_outputs = self._layer0(
            hidden_states,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=False,
        )
        hidden_states = layer_outputs[0]

        # Use scan_layers for layers 1-N (they're homogeneous)
        # scan_layers expects layers that take a single input and return a single output
        # We need to wrap our layers to handle the extra arguments

        # Create a wrapper that captures the extra arguments
        class LayerWrapper(torch.nn.Module):
            def __init__(self, layer, attn_mask, sw_mask, pos_ids, cu_seq, max_seq):
                super().__init__()
                self.layer = layer
                self.attn_mask = attn_mask
                self.sw_mask = sw_mask
                self.pos_ids = pos_ids
                self.cu_seq = cu_seq
                self.max_seq = max_seq

            def forward(self, hidden_states):
                out = self.layer(
                    hidden_states,
                    attention_mask=self.attn_mask,
                    sliding_window_mask=self.sw_mask,
                    position_ids=self.pos_ids,
                    cu_seqlens=self.cu_seq,
                    max_seqlen=self.max_seq,
                    output_attentions=False,
                )
                return out[0]

        # Wrap all homogeneous layers with the same extra arguments
        wrapped_layers = torch.nn.ModuleList([
            LayerWrapper(layer, attention_mask, sliding_window_mask, position_ids, cu_seqlens, max_seqlen)
            for layer in self._homogeneous_layers
        ])

        # Apply scan_layers to the wrapped layers
        hidden_states = scan_layers(wrapped_layers, hidden_states)

        hidden_states = self.final_norm(hidden_states)

        if repad:
            from transformers.models.modernbert.modeling_modernbert import _pad_modernbert_output
            hidden_states = _pad_modernbert_output(
                inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len
            )

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )

    # Bind the new forward method
    bert_model.forward = types.MethodType(scan_forward, bert_model)
    print(f"[XLA Scan] Patched ModernBERT to use scan_layers")
    print(f"[XLA Scan] Layer 0 runs separately, layers 1-{len(layers)-1} use scan_layers")
    print(f"[XLA Scan] This should reduce compilation time significantly")


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
