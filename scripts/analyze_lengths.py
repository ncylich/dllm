#!/usr/bin/env python3
"""
Analyze sequence length distribution to determine optimal bucketing strategy.

Usage:
    python scripts/analyze_lengths.py --dataset tatsu-lab/alpaca
    python scripts/analyze_lengths.py --dataset HuggingFaceTB/smoltalk --split train
    python scripts/analyze_lengths.py --dataset path/to/preprocessed --load_preprocessed
"""

import argparse
from collections import Counter

import numpy as np


def analyze_lengths(lengths: list[int], buckets: list[int] | None = None):
    """Analyze length distribution and compute bucket statistics."""
    lengths = np.array(lengths)

    print(f"\n{'='*60}")
    print("SEQUENCE LENGTH DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Total sequences: {len(lengths):,}")
    print(f"Min length: {lengths.min()}")
    print(f"Max length: {lengths.max()}")
    print(f"Mean length: {lengths.mean():.1f}")
    print(f"Median length: {np.median(lengths):.1f}")
    print(f"Std dev: {lengths.std():.1f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(lengths, p):.0f}")

    # Default buckets if not specified
    if buckets is None:
        buckets = [64, 128, 256, 512, 1024, 2048, 4096]

    # Filter buckets to those <= max length
    max_len = lengths.max()
    buckets = [b for b in buckets if b <= max_len * 1.5]
    if not buckets or buckets[-1] < max_len:
        # Add a bucket that covers max length
        next_power = 2 ** int(np.ceil(np.log2(max_len)))
        buckets.append(next_power)

    print(f"\n{'='*60}")
    print("BUCKET ANALYSIS")
    print(f"{'='*60}")

    # Analyze each potential max_length setting
    for max_len_setting in sorted(set(buckets)):
        eligible = lengths[lengths <= max_len_setting]
        if len(eligible) == 0:
            continue
        pct_kept = 100 * len(eligible) / len(lengths)
        avg_padding = max_len_setting - eligible.mean()
        efficiency = 100 * eligible.mean() / max_len_setting

        print(f"\nIf max_length = {max_len_setting}:")
        print(f"  Sequences kept: {len(eligible):,} ({pct_kept:.1f}%)")
        print(f"  Avg tokens used: {eligible.mean():.1f}")
        print(f"  Avg padding: {avg_padding:.1f} tokens")
        print(f"  Compute efficiency: {efficiency:.1f}%")

    print(f"\n{'='*60}")
    print("MULTI-BUCKET STRATEGY")
    print(f"{'='*60}")

    # Simulate different bucket configurations
    bucket_configs = [
        [128, 256, 512, 1024],
        [256, 512, 1024],
        [128, 384, 768, 1024],
        [64, 128, 256, 512, 1024],
        [256, 512, 768, 1024],
    ]

    # Filter to configs where max bucket >= max sequence length (or close)
    valid_configs = []
    for config in bucket_configs:
        if config[-1] >= np.percentile(lengths, 99):
            valid_configs.append(config)

    if not valid_configs:
        # Create a reasonable config based on data
        p99 = np.percentile(lengths, 99)
        max_bucket = 2 ** int(np.ceil(np.log2(p99)))
        valid_configs = [[max_bucket // 4, max_bucket // 2, max_bucket]]

    for config in valid_configs:
        print(f"\nBuckets: {config}")
        total_compute = 0
        total_seqs = 0
        bucket_counts = []

        for i, bucket in enumerate(config):
            lower = config[i-1] if i > 0 else 0
            in_bucket = lengths[(lengths > lower) & (lengths <= bucket)]
            count = len(in_bucket)
            bucket_counts.append(count)
            total_compute += count * bucket
            total_seqs += count

        # Sequences exceeding max bucket (would be truncated)
        exceeding = lengths[lengths > config[-1]]
        if len(exceeding) > 0:
            total_compute += len(exceeding) * config[-1]
            total_seqs += len(exceeding)
            bucket_counts.append(len(exceeding))

        avg_padded_len = total_compute / total_seqs if total_seqs > 0 else 0
        fixed_compute = len(lengths) * config[-1]
        savings = 100 * (1 - total_compute / fixed_compute) if fixed_compute > 0 else 0

        for i, bucket in enumerate(config):
            lower = config[i-1] if i > 0 else 0
            print(f"  {lower+1:4d}-{bucket:4d}: {bucket_counts[i]:6,} sequences")
        if len(exceeding) > 0:
            print(f"  >{config[-1]:4d} (truncated): {len(exceeding):6,} sequences")

        print(f"  ---")
        print(f"  Avg padded length: {avg_padded_len:.1f} (vs {config[-1]} fixed)")
        print(f"  Compute savings: {savings:.1f}%")
        print(f"  Recompilations: {len(config)} (one per bucket)")


def main():
    parser = argparse.ArgumentParser(description="Analyze sequence length distribution")
    parser.add_argument("--dataset", type=str, default="data/sft/bert/tulu-3-sft-mixture+smoltalk-1024",
                        help="Dataset name or path")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to analyze")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-base",
                        help="Model/tokenizer to use for tokenization")
    parser.add_argument("--load_preprocessed", action="store_true", default=True,
                        help="Load preprocessed dataset (already has input_ids)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to analyze (for large datasets)")
    parser.add_argument("--buckets", type=str, default=None,
                        help="Comma-separated bucket sizes to analyze")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")

    if args.load_preprocessed:
        from datasets import load_from_disk
        dataset = load_from_disk(args.dataset)
        if isinstance(dataset, dict):
            dataset = dataset[args.split]

        # Get lengths from input_ids
        print("Extracting lengths from preprocessed input_ids...")
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        lengths = [len(x) for x in dataset["input_ids"]]
    else:
        from datasets import load_dataset
        import transformers

        dataset = load_dataset(args.dataset, split=args.split)
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        print(f"Loading tokenizer: {args.model}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

        # Check if dataset has messages or text
        if "messages" in dataset.column_names:
            print("Tokenizing messages...")
            def get_length(example):
                text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False
                )
                return {"length": len(tokenizer.encode(text))}
        elif "text" in dataset.column_names:
            print("Tokenizing text...")
            def get_length(example):
                return {"length": len(tokenizer.encode(example["text"]))}
        elif "input_ids" in dataset.column_names:
            print("Using existing input_ids...")
            def get_length(example):
                return {"length": len(example["input_ids"])}
        else:
            # Try to find a text-like column
            text_cols = [c for c in dataset.column_names if "text" in c.lower() or "content" in c.lower()]
            if text_cols:
                col = text_cols[0]
                print(f"Tokenizing column '{col}'...")
                def get_length(example):
                    return {"length": len(tokenizer.encode(str(example[col])))}
            else:
                raise ValueError(f"Could not find text column. Available: {dataset.column_names}")

        dataset = dataset.map(get_length, num_proc=8, desc="Computing lengths")
        lengths = dataset["length"]

    # Parse custom buckets if provided
    buckets = None
    if args.buckets:
        buckets = [int(x.strip()) for x in args.buckets.split(",")]

    analyze_lengths(lengths, buckets)


if __name__ == "__main__":
    main()
