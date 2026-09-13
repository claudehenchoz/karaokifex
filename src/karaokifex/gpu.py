"""GPU housekeeping shared by the model-running steps."""

from __future__ import annotations

import gc


def free_gpu_memory() -> None:
    """Release cached CUDA memory once a model is no longer needed, so the next one fits."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
