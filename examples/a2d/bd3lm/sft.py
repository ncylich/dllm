"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/a2d/bd3lm/sft.py

- 8 GPUs (ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/a2d/bd3lm/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (ZeRO-2):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/sft.py"

- 2 Nodes, 16 GPUs (ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/sft.py"
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
    model_name_or_path: str = "models/a2d/Qwen3-0.6B"


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
    output_dir: str = "models/a2d/Qwen3-0.6B/mdlm/alpaca"
    group_by_length: bool = True
    num_train_epochs: int = 20
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    # a2d-specific
    block_size: int = 32
    right_shift_logits: bool = False


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
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
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
        # truncate / filter long sequences if needed (only for non-streaming)
        if not data_args.streaming:
            dataset = dllm.utils.post_process_dataset(dataset, data_args)
        else:
            # For streaming, shuffle the dataset
            dataset = dataset.shuffle(seed=training_args.seed)

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

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.BD3LMTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        block_size=training_args.block_size,
        right_shift_logits=training_args.right_shift_logits,
        data_collator=(
            dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                ),
                block_size=training_args.block_size,
            )
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


main = train  # alias for TPU launcher compatibility

if __name__ == "__main__":
    train()
