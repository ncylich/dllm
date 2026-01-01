# Disable torch.compile/dynamo on TPU BEFORE importing transformers
# This must happen before ModernBERT's @torch.compile decorators are applied
from dllm.utils.device import is_tpu_available

if is_tpu_available():
    import torch._dynamo

    torch._dynamo.config.disable = True

from . import core, data, pipelines, utils
