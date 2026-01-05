"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/bert/sft.py

- 8 GPUs (ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/bert/sft.py

- TPU with streaming (memory-efficient, for large datasets):
    accelerate launch \
        --config_file scripts/accelerate_configs/tpu.yaml \
        examples/bert/sft.py \
        --streaming True --num_train_epochs 10
    # max_steps is auto-computed from num_train_epochs for known datasets

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (ZeRO-2):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/bert/sft.py"

- 2 Nodes, 16 GPUs (ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/bert/sft.py"
"""

import os
from dataclasses import dataclass, field
from functools import partial

# Import dllm FIRST to disable torch.compile on TPU before transformers loads
import dllm

import accelerate
import transformers

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-large"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    max_length: int = 512
    streaming: bool = False
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/ModernBERT-large/alpaca"
    group_by_length: bool = True
    num_train_epochs: int = 20
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # Disable warn_if_padding_and_no_attention_mask on TPU - it triggers device-to-host
    # sync via __contains__ check on input_ids tensor every forward pass
    if dllm.utils.device.is_tpu_available():
        model.warn_if_padding_and_no_attention_mask = lambda *args, **kwargs: None
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    # For preprocessed data, skip the barrier - all processes can load simultaneously
    # since it's read-only. The barrier can cause hangs on TPU with XLA distributed.
    if data_args.load_preprocessed_data:
        logger.info("Loading preprocessed dataset (no barrier)...")
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        # Remove any extra columns (like 'id') that can't be converted to tensors
        # Only keep columns needed for training
        train_columns = {"input_ids", "labels", "attention_mask"}
        for split in dataset:
            extra_cols = [c for c in dataset[split].column_names if c not in train_columns]
            if extra_cols:
                logger.info(f"Removing extra columns from {split}: {extra_cols}")
                dataset[split] = dataset[split].remove_columns(extra_cols)
        # Skip post_process_dataset for preprocessed data - it was already
        # filtered/truncated during preprocessing
    else:
        with accelerate.PartialState().local_main_process_first():
            dataset = dllm.data.load_sft_dataset(
                data_args.dataset_args,
                streaming=data_args.streaming,
                load_preprocessed_data=data_args.load_preprocessed_data,
            )
            map_fn = partial(
                dllm.utils.default_mdlm_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            # For streaming, remove 'messages' column; for non-streaming, remove all original columns
            if data_args.streaming:
                remove_cols = ["messages"]
            else:
                remove_cols = dataset["train"].column_names
            dataset = dataset.map(
                map_fn,
                remove_columns=remove_cols,
                **({} if data_args.streaming else {"num_proc": data_args.num_proc}),
                **({} if data_args.streaming else {"desc": "Mapping dataset to SFT format"}),
            )
            # truncate / filter long sequences if needed
            if not data_args.streaming:
                dataset = dllm.utils.post_process_dataset(dataset, data_args)
            else:
                # For streaming, truncate and shuffle the dataset
                dataset = dllm.utils.post_process_dataset_streaming(dataset, data_args)
                # Use larger shuffle buffer for better randomization and prefetching
                # buffer_size=10000 provides good balance of memory vs randomization
                dataset = dataset.shuffle(seed=training_args.seed, buffer_size=10000)

    # ----- Auto-compute max_steps for streaming -----------------------------------
    if data_args.streaming and training_args.max_steps <= 0:
        max_steps = dllm.data.compute_max_steps(
            dataset_args=data_args.dataset_args,
            num_epochs=training_args.num_train_epochs,
            per_device_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        )
        if max_steps is not None:
            training_args.max_steps = max_steps
        else:
            raise ValueError(
                "Streaming mode requires --max_steps to be set, "
                "or dataset size must be known for auto-computation."
            )

    # ----- Disable group_by_length for streaming (incompatible with IterableDataset)
    if data_args.streaming and training_args.group_by_length:
        logger.warning(
            "group_by_length is incompatible with streaming datasets, disabling it."
        )
        training_args.group_by_length = False

    # ----- Disable eval_strategy if no eval dataset available --------------------
    eval_dataset = dataset.get("test", None)
    if eval_dataset is None and training_args.eval_strategy != "no":
        logger.warning(
            "No eval dataset available, setting eval_strategy to 'no'."
        )
        training_args.eval_strategy = "no"

    # ----- Training --------------------------------------------------------------
    # NOTE: wait_for_everyone() removed - causes hangs on TPU v6 with PJRT
    # because PartialState sees 4 devices but we're running 1 OS process
    print("[DEBUG] About to start training setup...", flush=True)
    logger.info("Start training...")

    print("[DEBUG] Building data collator...", flush=True)
    # Build data collator - use fixed-length padding on TPU to avoid XLA recompilation
    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=-100,  # ignore padded tokens in loss
    )

    if data_args.pad_to_max_length:
        print("[DEBUG] Using pad_to_max_length...", flush=True)
        logger.info(
            f"pad_to_max_length=True: using fixed-length padding to max_length={data_args.max_length}"
        )
        # With fixed-length padding, all sequences are the same length, so we can
        # remove attention_mask (model will attend to all positions including padding)
        data_collator = dllm.utils.NoAttentionMaskWrapper(base_collator)
        data_collator = dllm.utils.collators.FixedLengthPaddingWrapper(
            data_collator,
            max_length=data_args.max_length,
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=-100,
        )
    else:
        # Default: remove attention_mask (original behavior for non-TPU or simple cases)
        data_collator = dllm.utils.NoAttentionMaskWrapper(base_collator)

    print("[DEBUG] Creating MDLMTrainer...", flush=True)
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    print("[DEBUG] Trainer created!", flush=True)

    # Disable num_items_in_batch computation on TPU - it triggers _tpu_gather every step
    # which causes device-to-host sync. With fixed batch sizes, this is unnecessary.
    if dllm.utils.device.is_tpu_available():
        trainer.model_accepts_loss_kwargs = False

    print("[DEBUG] Starting trainer.train()...", flush=True)
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


main = train  # alias for TPU launcher compatibility

if __name__ == "__main__":
    train()
