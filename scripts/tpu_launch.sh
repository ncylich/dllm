#!/bin/bash
# Wrapper script for TPU training with accelerate.
# This sets PJRT_SELECT_DEFAULT_DEVICE=0 to prevent XLA runtime initialization
# at import time, which is required for xmp.spawn() to work correctly.
#
# Usage:
#   ./scripts/tpu_launch.sh examples/bert/sft.py --model_name_or_path "..." [other args]
#
# This is equivalent to:
#   PJRT_SELECT_DEFAULT_DEVICE=0 accelerate launch --config_file scripts/accelerate_configs/tpu.yaml [script] [args]

export PJRT_SELECT_DEFAULT_DEVICE=0

exec accelerate launch --config_file scripts/accelerate_configs/tpu.yaml "$@"
