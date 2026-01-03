import os
from dataclasses import dataclass, field

import transformers

from dllm.utils.utils import get_default_logger, resolve_with_base_env

logger = get_default_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = None  # overwrite this
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    attn_implementation: str = None
    # --- fold PEFT args here ---
    lora: bool = False
    target_modules: str = "all-linear"
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    modules_to_save: str = None

    def __post_init__(self):
        self.model_name_or_path = resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class DataArguments:
    dataset_args: str = None  # overwrite this
    num_proc: int = 8
    disable_caching: bool = False
    max_length: int = 1024
    truncation: str = field(
        default="right",
        metadata={
            "help": (
                'The truncation strategy to use ("filter" or "right"). '
                '"filter" only keeps sequences that are shorter than max_length; '
                '"right" only keeps the rightmost max_length tokens for each sequence.'
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Pad all sequences to max_length. Critical for TPU/XLA training "
                "to avoid recompilation due to variable tensor shapes. "
                "Automatically enabled when TPU is detected (unless use_length_bucketing is True)."
            )
        },
    )
    use_length_bucketing: bool = field(
        default=False,
        metadata={
            "help": (
                "Use length bucketing to reduce padding waste. Groups sequences by length "
                "and pads to bucket boundaries instead of max_length. Provides significant "
                "compute savings (~35-40%) at the cost of multiple XLA compilations."
            )
        },
    )
    length_buckets: str = field(
        default="256,512,768,1024",
        metadata={
            "help": (
                "Comma-separated list of bucket sizes for length bucketing. "
                "Sequences are padded to the smallest bucket that fits them. "
                "Each bucket triggers one XLA compilation. Default: '256,512,768,1024'"
            )
        },
    )

    def __post_init__(self):
        # Auto-enable pad_to_max_length on TPU if not explicitly set
        # (unless length bucketing is enabled, which handles fixed shapes differently)
        from dllm.utils.device import is_tpu_available

        if is_tpu_available() and not self.pad_to_max_length and not self.use_length_bucketing:
            logger.info(
                "TPU detected: automatically enabling pad_to_max_length=True "
                "to avoid XLA recompilation. Use --use_length_bucketing for better efficiency."
            )
            self.pad_to_max_length = True

        # Parse length_buckets string to list
        if isinstance(self.length_buckets, str):
            self._parsed_buckets = [int(x.strip()) for x in self.length_buckets.split(",")]
        else:
            self._parsed_buckets = self.length_buckets

        # Warn if both pad_to_max_length and use_length_bucketing are set
        if self.pad_to_max_length and self.use_length_bucketing:
            logger.warning(
                "Both pad_to_max_length and use_length_bucketing are enabled. "
                "Length bucketing will take precedence."
            )
            self.pad_to_max_length = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = None  # overwrite this
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    num_train_epochs: float = 10
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    bf16: bool = True
    logging_steps: float = 10
    eval_on_start: bool = False
    eval_strategy: str = "steps"
    eval_steps: float = 0.1
    save_steps: float = 0.1
    save_only_model: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Disable pin_memory on TPU (it's a CUDA-only optimization)
        from dllm.utils.device import is_tpu_available

        if is_tpu_available():
            self.dataloader_pin_memory = False
            # NOTE: dataloader_num_workers > 0 causes pickle errors with streaming
            # datasets that use local filter functions. Keep at 0 for compatibility.
            # Drop last incomplete batch to ensure consistent tensor shapes
            # (reduces XLA recompilation)
            if not self.dataloader_drop_last:
                self.dataloader_drop_last = True
                logger.info(
                    "TPU detected: enabling dataloader_drop_last=True to avoid "
                    "recompilation from variable batch sizes."
                )
            # Disable group_by_length on TPU - provides no benefit with pad_to_max_length
            # and causes slow preprocessing on large datasets
            if self.group_by_length:
                self.group_by_length = False
                logger.info(
                    "TPU detected: disabling group_by_length (no benefit with fixed-length padding)."
                )
