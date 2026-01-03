from dataclasses import dataclass
from typing import Any

import torch
import transformers


@dataclass
class CollatorWrapper:
    """
    Gym-style DataCollator wrapper.
    Enables stacking multiple wrappers: Wrapper3(Wrapper2(Wrapper1(BaseCollator()))).
    """

    collator: Any

    def before(self, features):
        return features

    def after(self, outputs):
        return outputs

    def __call__(self, features, return_tensors=None):
        # Pre-hook
        features = self.before(features)

        # Call the wrapped collator
        outputs = self.collator(features, return_tensors=return_tensors)

        # Post-hook
        outputs = self.after(outputs)
        return outputs

    def __getattr__(self, name: str):
        """
        If an attribute is not found on this wrapper, automatically delegate
        the lookup to `self.collator`.

        This supports arbitrarily nested wrappers, because the inner collator
        may itself implement `__getattr__`, allowing recursive delegation.
        """
        # Python only calls __getattr__ when normal attribute lookup fails,
        # so it's safe to attempt fetching from the wrapped collator here.
        collator = self.__dict__.get("collator", None)
        if collator is not None:
            try:
                return getattr(collator, name)
            except AttributeError:
                pass  # Fall through and raise below if still not found

        # By protocol, __getattr__ must raise AttributeError if the attribute
        # truly does not exist anywhere.
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@dataclass
class NoAttentionMaskWrapper(CollatorWrapper):
    """
    Collator wrapper that removes attention_mask from outputs.

    Useful when the model doesn't need explicit attention masks or when
    all sequences are of equal length.
    """

    def after(self, outputs):
        outputs.pop("attention_mask", None)
        return outputs


@dataclass
class PrependBOSWrapper(CollatorWrapper):
    """
    Collator wrapper that prepends BOS token to sequences.

    Prepends the beginning-of-sequence token to input_ids, and correspondingly
    prepends an ignored label (-100) to labels and a 1 to attention_mask.

    Attributes:
        bos_token_id: The BOS token ID to prepend.
        label_pad_token_id: Token ID to use for ignored labels (default: -100).
    """

    bos_token_id: int | None = None
    label_pad_token_id: int = -100

    def after(self, outputs):
        assert self.bos_token_id
        input_ids = outputs.get("input_ids")

        bsz, _ = input_ids.shape

        # prepend BOS to input_ids
        bos = torch.full(
            (bsz, 1),
            self.bos_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([bos, input_ids], dim=1)
        outputs["input_ids"] = input_ids

        # prepend ignored label if labels exist
        labels = outputs.get("labels", None)
        if labels is not None:
            ignore_labels = torch.full(
                (bsz, 1),
                self.label_pad_token_id,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([ignore_labels, labels], dim=1)
            outputs["labels"] = labels

        # prepend attention mask if it exists
        attention_mask = outputs.get("attention_mask", None)
        if attention_mask is not None:
            bos_attention = torch.ones(
                (bsz, 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([bos_attention, attention_mask], dim=1)
            outputs["attention_mask"] = attention_mask

        return outputs


@dataclass
class FixedLengthPaddingWrapper(CollatorWrapper):
    """
    Collator wrapper that pads all sequences to a fixed length.

    This is critical for TPU/XLA training to avoid recompilation due to
    variable tensor shapes. Pads input_ids, labels, and attention_mask
    to max_length.

    Attributes:
        max_length: The fixed length to pad all sequences to.
        pad_token_id: Token ID to use for padding input_ids.
        label_pad_token_id: Token ID to use for padding labels (default: -100).
    """

    max_length: int = 1024
    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def after(self, outputs):
        batch_size = outputs["input_ids"].shape[0]
        current_length = outputs["input_ids"].shape[1]

        if current_length >= self.max_length:
            # Truncate if longer than max_length
            for key in ["input_ids", "labels", "attention_mask"]:
                if key in outputs:
                    outputs[key] = outputs[key][:, : self.max_length]
        else:
            # Pad to max_length
            pad_length = self.max_length - current_length

            # Pad input_ids
            input_pad = torch.full(
                (batch_size, pad_length),
                self.pad_token_id,
                dtype=outputs["input_ids"].dtype,
                device=outputs["input_ids"].device,
            )
            outputs["input_ids"] = torch.cat([outputs["input_ids"], input_pad], dim=1)

            # Pad labels if present
            if "labels" in outputs:
                label_pad = torch.full(
                    (batch_size, pad_length),
                    self.label_pad_token_id,
                    dtype=outputs["labels"].dtype,
                    device=outputs["labels"].device,
                )
                outputs["labels"] = torch.cat([outputs["labels"], label_pad], dim=1)

            # Pad attention_mask if present
            if "attention_mask" in outputs:
                attn_pad = torch.zeros(
                    (batch_size, pad_length),
                    dtype=outputs["attention_mask"].dtype,
                    device=outputs["attention_mask"].device,
                )
                outputs["attention_mask"] = torch.cat(
                    [outputs["attention_mask"], attn_pad], dim=1
                )

        return outputs


@dataclass
class RandomTruncateWrapper(CollatorWrapper):
    """
    Collator wrapper that randomly truncates sequences during training.

    With probability random_length_ratio, truncates all sequences in the batch
    to a random length. Also removes attention_mask if it's all ones (no padding).

    Attributes:
        random_length_ratio: Probability of applying random truncation (default: 0.01).
    """

    random_length_ratio: float = 0.01

    def after(self, outputs):
        if torch.rand(1) < self.random_length_ratio:
            random_length = torch.randint(1, outputs["input_ids"].shape[1] + 1, (1,))
            for key in ["input_ids", "labels", "attention_mask"]:
                if key in outputs:
                    outputs[key] = outputs[key][:, :random_length]
        # Check if attention_mask is all ones and set it to None
        if "attention_mask" in outputs and torch.all(outputs["attention_mask"] == 1):
            outputs.pop("attention_mask")
        return outputs


@dataclass
class BucketedPaddingWrapper(CollatorWrapper):
    """
    Collator wrapper that pads sequences to bucket boundaries instead of max_length.

    This reduces wasted compute on TPU/XLA by padding shorter sequences to smaller
    fixed lengths. Each unique bucket size triggers one XLA compilation, so the
    number of buckets trades off between compute savings and compilation overhead.

    Attributes:
        buckets: List of bucket sizes in ascending order (e.g., [256, 512, 768, 1024]).
            Sequences are padded to the smallest bucket that fits them.
        pad_token_id: Token ID to use for padding input_ids.
        label_pad_token_id: Token ID to use for padding labels (default: -100).
    """

    buckets: list[int] = None
    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __post_init__(self):
        if self.buckets is None:
            self.buckets = [256, 512, 768, 1024]
        # Ensure buckets are sorted
        self.buckets = sorted(self.buckets)

    def _get_bucket_size(self, length: int) -> int:
        """Find the smallest bucket that can fit the given length."""
        for bucket in self.buckets:
            if length <= bucket:
                return bucket
        # If longer than all buckets, use the largest bucket (will truncate)
        return self.buckets[-1]

    def after(self, outputs):
        batch_size = outputs["input_ids"].shape[0]
        current_length = outputs["input_ids"].shape[1]

        # Find the appropriate bucket for this batch
        target_length = self._get_bucket_size(current_length)

        if current_length >= target_length:
            # Truncate if longer than target bucket
            for key in ["input_ids", "labels", "attention_mask"]:
                if key in outputs:
                    outputs[key] = outputs[key][:, :target_length]
        else:
            # Pad to target bucket length
            pad_length = target_length - current_length

            # Pad input_ids
            input_pad = torch.full(
                (batch_size, pad_length),
                self.pad_token_id,
                dtype=outputs["input_ids"].dtype,
                device=outputs["input_ids"].device,
            )
            outputs["input_ids"] = torch.cat([outputs["input_ids"], input_pad], dim=1)

            # Pad labels if present
            if "labels" in outputs:
                label_pad = torch.full(
                    (batch_size, pad_length),
                    self.label_pad_token_id,
                    dtype=outputs["labels"].dtype,
                    device=outputs["labels"].device,
                )
                outputs["labels"] = torch.cat([outputs["labels"], label_pad], dim=1)

            # Pad attention_mask if present
            if "attention_mask" in outputs:
                attn_pad = torch.zeros(
                    (batch_size, pad_length),
                    dtype=outputs["attention_mask"].dtype,
                    device=outputs["attention_mask"].device,
                )
                outputs["attention_mask"] = torch.cat(
                    [outputs["attention_mask"], attn_pad], dim=1
                )

        return outputs


class BucketedBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that groups samples by length into buckets for efficient padding.

    This sampler ensures that samples within each batch have similar lengths,
    minimizing padding waste. Combined with BucketedPaddingWrapper, this provides
    significant compute savings on TPU/XLA.

    Args:
        lengths: List/array of sequence lengths for each sample in the dataset.
        batch_size: Number of samples per batch.
        buckets: List of bucket boundaries (e.g., [256, 512, 768, 1024]).
        drop_last: If True, drop the last incomplete batch in each bucket.
        shuffle: If True, shuffle samples within each bucket and shuffle batch order.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        buckets: list[int] | None = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.buckets = sorted(buckets) if buckets else [256, 512, 768, 1024]
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Assign each sample to a bucket
        self.bucket_indices = self._assign_buckets()

    def _assign_buckets(self) -> dict[int, list[int]]:
        """Assign each sample index to its appropriate bucket."""
        bucket_indices = {b: [] for b in self.buckets}

        for idx, length in enumerate(self.lengths):
            # Find smallest bucket that fits
            for bucket in self.buckets:
                if length <= bucket:
                    bucket_indices[bucket].append(idx)
                    break
            else:
                # Longer than all buckets - assign to largest
                bucket_indices[self.buckets[-1]].append(idx)

        return bucket_indices

    def __iter__(self):
        # Create a generator for shuffling
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []

        for bucket, indices in self.bucket_indices.items():
            if len(indices) == 0:
                continue

            # Shuffle indices within this bucket
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle the order of batches across buckets
        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]

        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self.bucket_indices.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total += n_batches
        return total

    def set_epoch(self, epoch: int):
        """Set the epoch for shuffling reproducibility."""
        self.epoch = epoch


def get_bucket_sampler_and_collator(
    dataset,
    tokenizer,
    batch_size: int,
    buckets: list[int] | None = None,
    drop_last: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    remove_attention_mask: bool = True,
):
    """
    Create a bucketed batch sampler and collator for efficient TPU training.

    This is a convenience function that sets up both components needed for
    length-bucketed training:
    1. BucketedBatchSampler - groups similar-length samples into batches
    2. BucketedPaddingWrapper - pads each batch to its bucket boundary

    Args:
        dataset: HuggingFace dataset with 'input_ids' column.
        tokenizer: Tokenizer for padding configuration.
        batch_size: Samples per batch.
        buckets: Bucket boundaries (default: [256, 512, 768, 1024]).
        drop_last: Drop incomplete batches.
        shuffle: Shuffle within buckets.
        seed: Random seed.
        remove_attention_mask: Remove attention_mask from outputs.

    Returns:
        Tuple of (batch_sampler, data_collator)

    Example:
        >>> sampler, collator = get_bucket_sampler_and_collator(
        ...     dataset["train"], tokenizer, batch_size=8,
        ...     buckets=[256, 512, 768, 1024]
        ... )
        >>> trainer = Trainer(..., data_collator=collator)
        >>> # Pass sampler via custom dataloader
    """
    if buckets is None:
        buckets = [256, 512, 768, 1024]

    # Extract lengths from dataset
    if "input_ids" in dataset.column_names:
        lengths = [len(x) for x in dataset["input_ids"]]
    else:
        raise ValueError("Dataset must have 'input_ids' column for bucketing")

    # Create batch sampler
    batch_sampler = BucketedBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        buckets=buckets,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
    )

    # Create collator chain
    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    if remove_attention_mask:
        collator = NoAttentionMaskWrapper(base_collator)
    else:
        collator = base_collator

    collator = BucketedPaddingWrapper(
        collator,
        buckets=buckets,
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
    )

    return batch_sampler, collator


if __name__ == "__main__":
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")

    # Base HF collator
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        return_tensors="pt",
        padding=True,
    )

    # Wrap it
    collator = NoAttentionMaskWrapper(collator)

    # Dummy samples
    samples = [
        {"input_ids": tokenizer("hello world")["input_ids"]},
        {"input_ids": tokenizer("goodbye")["input_ids"]},
    ]

    # Apply collator
    batch = collator(samples, return_tensors="pt")

    # Print output
    print("Batch keys:", batch.keys())
    print("input_ids:\n", batch["input_ids"])
    print("labels:\n", batch["labels"])

    # Check attention_mask is removed
    assert "attention_mask" not in batch
    print("\nTest passed: attention_mask was removed.")
