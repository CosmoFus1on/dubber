"""Device detection and VRAM hygiene. One model family in VRAM at a time on a 10 GB card."""
from __future__ import annotations

import gc
import os


def device(force_cpu: bool = False) -> str:
    if force_cpu or os.environ.get("DUBBER_CPU") == "1":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def unload(*objs) -> None:
    """Drop references, collect, and free CUDA cache. Call after every stage that loads a model."""
    objs = None  # noqa: F841 - the caller's own references must also be dropped (del / reassign)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def vram_used_mb() -> float:
    try:
        import torch
        return torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0
