"""
Local users
------------
- 1 GPU (4bit quant & LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/llada/sft.py \
        --load_in_4bit True --lora True

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/llada/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (FSDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"

- 2 Nodes, 16 GPUs (FSDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"
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
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    streaming: bool = False
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/LLaDA-8B-Base/tulu-3-sft-mixture[train:10000,test:1000]"
    group_by_length: bool = True
    num_train_epochs: float = 5
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4


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
        # truncate / filter long sequences if needed
        if not data_args.streaming:
            dataset = dllm.utils.post_process_dataset(dataset, data_args)
        else:
            # For streaming, truncate and shuffle the dataset
            dataset = dllm.utils.post_process_dataset_streaming(dataset, data_args)
            # Use larger shuffle buffer for better randomization and prefetching
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
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=(
            dllm.utils.NoAttentionMaskWrapper(  # padded <eos_token> should be visible
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=tokenizer.pad_token_id,  # finetune on padded <eos_token>
                ),
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
