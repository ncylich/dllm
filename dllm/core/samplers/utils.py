import torch

from dllm.core.schedulers import BaseAlphaScheduler


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute the number of tokens to unmask at each diffusion step.

    For each sample, determines how many masked tokens should be revealed
    per step based on the reverse diffusion schedule.

    This implementation is vectorized to avoid device synchronization issues
    on TPU/XLA while remaining fully compatible with CUDA/CPU.

    Args:
        mask_index: Boolean tensor [B, L] indicating masked positions.
        steps: Number of diffusion steps.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected number of tokens.

    Returns:
        Integer tensor [B, steps] with number of tokens to unmask per step.
    """
    B = mask_index.size(0)
    device = mask_index.device

    # Total masks per sample: [B]
    mask_num = mask_index.sum(dim=1).to(torch.float64)

    # Precompute reverse_transfer_prob for all steps (vectorized): [steps]
    # t goes from steps down to 1, s goes from steps-1 down to 0
    t_vals = torch.arange(steps, 0, -1, device=device, dtype=torch.float64) / steps
    s_vals = torch.arange(steps - 1, -1, -1, device=device, dtype=torch.float64) / steps
    # scheduler.reverse_mask_prob supports tensor inputs
    reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s_vals, t=t_vals)  # [steps]

    if not stochastic:
        # Deterministic path: fully vectorized, no loops needed
        # Compute cumulative unmask fractions using the product formula
        mask_prob = 1 - reverse_transfer_prob  # probability of staying masked at each step
        cumulative_mask_prob = torch.cumprod(mask_prob, dim=0)  # [steps]
        cumulative_unmask_frac = 1 - cumulative_mask_prob  # [steps]

        # Expected cumulative unmasked tokens at each step: [B, steps]
        cumulative_unmasked = mask_num.unsqueeze(1) * cumulative_unmask_frac.unsqueeze(0)

        # Round cumulative counts
        cumulative_unmasked_rounded = torch.round(cumulative_unmasked).to(torch.int64)

        # Clamp to not exceed total masks per sample
        mask_num_int = mask_num.to(torch.int64).unsqueeze(1)
        cumulative_unmasked_rounded = torch.minimum(cumulative_unmasked_rounded, mask_num_int)

        # Per-step tokens = diff of cumulative (prepend zeros)
        zero_col = torch.zeros(B, 1, device=device, dtype=torch.int64)
        cumulative_with_zero = torch.cat([zero_col, cumulative_unmasked_rounded], dim=1)
        num_transfer_tokens = cumulative_with_zero[:, 1:] - cumulative_with_zero[:, :-1]

        # Ensure non-negative (can happen from rounding)
        num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)
    else:
        # Stochastic path: must iterate since each step depends on remaining masks
        # But we vectorize across the batch dimension to minimize syncs
        num_transfer_tokens = torch.zeros(B, steps, device=device, dtype=torch.int64)
        remaining = mask_num.clone()  # [B]

        for j in range(steps):
            prob = reverse_transfer_prob[j].expand(B)
            # Sample from binomial distribution (vectorized across batch)
            samples = torch.distributions.Binomial(remaining, prob).sample()
            samples = torch.minimum(samples, remaining).to(torch.int64)
            num_transfer_tokens[:, j] = samples
            remaining = remaining - samples.to(torch.float64)

    # Compact trailing zero-only columns for efficiency (fewer loop iterations in caller)
    # On TPU/XLA, .any() in boolean context forces a device sync, so we skip compaction there.
    # On CUDA/CPU, compaction is safe and beneficial.
    is_xla = device.type == "xla"

    if not is_xla:
        # CUDA/CPU path: use .any() for efficient compaction
        has_tokens = (num_transfer_tokens > 0).any(dim=0)  # [steps]
        if has_tokens.any():
            # Find last column with tokens
            reversed_has = has_tokens.flip(0)
            reversed_cumsum = reversed_has.to(torch.int64).cumsum(dim=0)
            keep_mask = reversed_cumsum.flip(0) > 0
            num_transfer_tokens = num_transfer_tokens[:, keep_mask]
        else:
            num_transfer_tokens = torch.zeros(B, 1, device=device, dtype=torch.int64)

    # On XLA (TPU), skip compaction - the calling code handles zero-transfer steps as no-ops
    return num_transfer_tokens


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
