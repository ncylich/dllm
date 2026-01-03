#!/usr/bin/env python3
"""Test TPU detection before any other imports."""

import os
print(f"PJRT_DEVICE env var: {os.environ.get('PJRT_DEVICE', 'NOT SET')}")

# Test libtpu.so detection
try:
    import ctypes
    ctypes.CDLL("libtpu.so")
    print("libtpu.so: FOUND")
except OSError as e:
    print(f"libtpu.so: NOT FOUND ({e})")

# Now test the actual function
from dllm.utils.device import is_tpu_available
print(f"is_tpu_available(): {is_tpu_available()}")

# Check if dynamo was disabled
import torch._dynamo
print(f"torch._dynamo.config.disable: {torch._dynamo.config.disable}")
