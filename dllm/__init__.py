# Prevent XLA runtime initialization at import time.
# This must be set BEFORE importing torch_xla (via accelerate/transformers).
# Required for TPU training with xmp.spawn() in accelerate's TPU launcher.
import os

os.environ.setdefault("PJRT_SELECT_DEFAULT_DEVICE", "0")

from . import core, data, pipelines, utils
