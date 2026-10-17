import os

# Test tensor semantics on CPU without compiling a CUDA backend.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
