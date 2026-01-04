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
