import math
import os

import torch
import transformers

from dllm.utils.device import is_tpu_available


class XLAProfilerCallback(transformers.TrainerCallback):
    """
    Callback that profiles TPU execution using torch_xla's built-in profiler.

    Enable via DLLM_XLA_PROFILE=1 environment variable.
    Profiles steps between profile_start_step and profile_end_step.
    Saves traces to profile_logdir (default: /tmp/xla_profile).

    Usage:
        DLLM_XLA_PROFILE=1 accelerate launch ...

    View traces with:
        tensorboard --logdir=/tmp/xla_profile
    """

    def __init__(
        self,
        profile_start_step: int = 10,
        profile_end_step: int = 20,
        profile_logdir: str = "/tmp/xla_profile",
    ):
        self.profile_start_step = profile_start_step
        self.profile_end_step = profile_end_step
        self.profile_logdir = profile_logdir
        self._profiler = None
        self._profiling_active = False
        self._initialized = False

        # Only check env var at init time - TPU check is done lazily
        self._env_enabled = os.environ.get("DLLM_XLA_PROFILE", "").lower() in ("1", "true", "yes")
        self._enabled = False  # Will be set True on first step if TPU available

    def _lazy_init(self):
        """Initialize profiler settings on first step when TPU runtime is available."""
        if self._initialized:
            return
        self._initialized = True

        if self._env_enabled and is_tpu_available():
            self._enabled = True
            print(f"[XLA Profiler] Enabled. Will profile steps {self.profile_start_step}-{self.profile_end_step}")
            print(f"[XLA Profiler] Traces will be saved to: {self.profile_logdir}")
            os.makedirs(self.profile_logdir, exist_ok=True)

    def on_step_begin(self, args, state, control, **kwargs):
        """Start profiling at the designated step."""
        # Lazy init on first step when XLA runtime is available
        self._lazy_init()

        if not self._enabled:
            return control

        if state.global_step == self.profile_start_step and not self._profiling_active:
            try:
                import torch_xla.debug.profiler as xp

                # Start the profiler server
                self._server = xp.start_server(9012)
                print("[XLA Profiler] Started profiler server on port 9012")
                print(f"[XLA Profiler] Starting trace at step {state.global_step}")

                # Start tracing
                xp.trace_detached(
                    "localhost:9012",
                    self.profile_logdir,
                    duration_ms=60000,  # 60 seconds max
                )
                self._profiling_active = True
                print(f"[XLA Profiler] Trace started, saving to {self.profile_logdir}")

            except Exception as e:
                print(f"[XLA Profiler] Warning: Could not start profiler: {e}")

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Stop profiling at the designated step."""
        if not self._enabled:
            return control

        if state.global_step == self.profile_end_step and self._profiling_active:
            print(f"[XLA Profiler] Profiling complete at step {state.global_step}")
            print(f"[XLA Profiler] View traces with: tensorboard --logdir={self.profile_logdir}")
            self._profiling_active = False

        return control


class XLAMarkStepCallback(transformers.TrainerCallback):
    """
    Callback that calls xm.mark_step() periodically during TPU training.

    XLA builds up a computation graph lazily. Without periodic mark_step() calls,
    the graph can grow unbounded, causing memory issues and compilation overhead.
    This callback forces XLA to execute the accumulated graph at regular intervals.

    Args:
        mark_step_interval: Call mark_step() every N training steps. Default is 1
            (every step), which ensures consistent graph sizes. Higher values may
            improve throughput but risk larger graphs.
    """

    def __init__(self, mark_step_interval: int = 1):
        self.mark_step_interval = mark_step_interval
        self._is_tpu = is_tpu_available()
        self._xm = None
        if self._is_tpu:
            try:
                import torch_xla.core.xla_model as xm
                self._xm = xm
            except ImportError:
                pass

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if self._xm is not None and state.global_step % self.mark_step_interval == 0:
            self._xm.mark_step()
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        """Called at the end of each gradient accumulation substep."""
        # Mark step after each substep to keep graph size bounded during
        # gradient accumulation. Without this, the graph grows across all
        # accumulation steps and can exceed TPU memory.
        if self._xm is not None:
            self._xm.mark_step()
        return control


class EpochPPLMeter(transformers.TrainerCallback):
    """
    Keeps running sums for dataset-level NLL/token and logs PPL once per epoch.

    Usage:
      - Trainer calls: self.ppl_meter.update(split, nll_sum, token_cnt)
      - Callback hooks:
          * on_epoch_begin: reset train accumulators
          * on_epoch_end: finalize+log train PPL
          * on_evaluate:   finalize+log eval  PPL (one per evaluate call)

    TPU Optimization:
      When running on TPU, accumulates tensors on-device to avoid per-step
      host sync which kills XLA performance. Only syncs to CPU at epoch/eval
      boundaries.
    """

    def __init__(
        self,
        trainer: "transformers.Trainer",
        train_prefix: str = "train",
        eval_prefix: str = "eval",
    ):
        self.trainer = trainer
        self.train_prefix = train_prefix
        self.eval_prefix = eval_prefix
        self._is_tpu = is_tpu_available()

        # For TPU: accumulate on-device tensors (initialized lazily)
        self._train_nll_tensor = None
        self._train_tok_tensor = None
        self._eval_nll_tensor = None
        self._eval_tok_tensor = None

        # For non-TPU: use float accumulators (original behavior)
        self._train_nll_sum = 0.0
        self._train_token_cnt = 0.0
        self._eval_nll_sum = 0.0
        self._eval_token_cnt = 0.0

    def reset(self, split: str) -> None:
        if split == "train":
            self._train_nll_sum = 0.0
            self._train_token_cnt = 0.0
            self._train_nll_tensor = None
            self._train_tok_tensor = None
        elif split == "eval":
            self._eval_nll_sum = 0.0
            self._eval_token_cnt = 0.0
            self._eval_nll_tensor = None
            self._eval_tok_tensor = None
        else:
            raise ValueError(f"Unknown split={split}")

    def update(self, split: str, nll_sum: torch.Tensor, token_cnt: torch.Tensor) -> None:
        if self._is_tpu:
            # TPU: accumulate on-device to avoid per-step sync
            nll_detached = nll_sum.detach().double()
            tok_detached = token_cnt.detach().double()

            if split == "train":
                if self._train_nll_tensor is None:
                    self._train_nll_tensor = nll_detached.clone()
                    self._train_tok_tensor = tok_detached.clone()
                else:
                    self._train_nll_tensor = self._train_nll_tensor + nll_detached
                    self._train_tok_tensor = self._train_tok_tensor + tok_detached
            elif split == "eval":
                if self._eval_nll_tensor is None:
                    self._eval_nll_tensor = nll_detached.clone()
                    self._eval_tok_tensor = tok_detached.clone()
                else:
                    self._eval_nll_tensor = self._eval_nll_tensor + nll_detached
                    self._eval_tok_tensor = self._eval_tok_tensor + tok_detached
            else:
                raise ValueError(f"Unknown split={split}")
        else:
            # Non-TPU: original behavior with immediate CPU transfer
            nll_sum_f = float(nll_sum.detach().double().cpu().item())
            tok_cnt_f = float(token_cnt.detach().double().cpu().item())

            if split == "train":
                self._train_nll_sum += nll_sum_f
                self._train_token_cnt += tok_cnt_f
            elif split == "eval":
                self._eval_nll_sum += nll_sum_f
                self._eval_token_cnt += tok_cnt_f
            else:
                raise ValueError(f"Unknown split={split}")

    def _finalize(self, split: str):
        """
        All-reduce (sum) across processes, then compute:
            mean_nll = total_nll / total_tokens
            ppl      = exp(mean_nll)

        Returns (mean_nll, ppl) as python floats, or (None, None) if no tokens.
        Also resets that split after finalizing.
        """
        if self._is_tpu:
            # TPU path: get accumulated tensors and sync to CPU only now
            if split == "train":
                nll_tensor = self._train_nll_tensor
                tok_tensor = self._train_tok_tensor
            elif split == "eval":
                nll_tensor = self._eval_nll_tensor
                tok_tensor = self._eval_tok_tensor
            else:
                raise ValueError(f"Unknown split={split}")

            self.reset(split)

            if nll_tensor is None or tok_tensor is None:
                return None, None

            # Stack into single tensor for all-reduce
            stats = torch.stack([nll_tensor, tok_tensor])

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

            # Single sync to CPU at finalize time
            total_nll = float(stats[0].cpu().item())
            total_tok = float(stats[1].cpu().item())
        else:
            # Non-TPU path: original behavior
            if split == "train":
                local_nll, local_tok = self._train_nll_sum, self._train_token_cnt
            elif split == "eval":
                local_nll, local_tok = self._eval_nll_sum, self._eval_token_cnt
            else:
                raise ValueError(f"Unknown split={split}")

            self.reset(split)

            if local_tok <= 0.0:
                return None, None

            device = getattr(self.trainer.args, "device", torch.device("cpu"))
            stats = torch.tensor([local_nll, local_tok], device=device, dtype=torch.float64)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

            total_nll = float(stats[0].item())
            total_tok = float(stats[1].item())

        if total_tok <= 0.0:
            return None, None

        mean_nll = total_nll / total_tok
        ppl = math.exp(mean_nll)
        return mean_nll, ppl

    # ---- callback hooks ----

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.reset("train")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        mean_nll, ppl = self._finalize("train")
        if mean_nll is not None and self.trainer.is_world_process_zero():
            logs = {f"{self.train_prefix}_nll": mean_nll, f"{self.train_prefix}_ppl": ppl}
            self.trainer.log(logs)
            print(f"[epoch {state.epoch}] {self.train_prefix}_nll={mean_nll:.6f} {self.train_prefix}_ppl={ppl:.6f}")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        mean_nll, ppl = self._finalize("eval")
        if mean_nll is not None and self.trainer.is_world_process_zero():
            logs = {f"{self.eval_prefix}_nll": mean_nll, f"{self.eval_prefix}_ppl": ppl}
            self.trainer.log(logs)
            print(f"[epoch {state.epoch}] {self.eval_prefix}_nll={mean_nll:.6f} {self.eval_prefix}_ppl={ppl:.6f}")
        return control
