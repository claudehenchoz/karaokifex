"""GPU housekeeping shared by the model-running steps."""

from __future__ import annotations

import gc
import sys
from pathlib import Path


def free_gpu_memory() -> None:
    """Release cached CUDA memory once a model is no longer needed, so the next one fits."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SharedGpu:
    """A lock file that several karaokifex runs on one machine share, so their models take turns on the GPU.

    Two songs made at once each keep their own models in video memory; running two at the same time can
    fill it, and then both crawl (the driver pages memory out) instead of either finishing. Held with
    `with`, it waits for the other runs' turns, however long they take.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._file = None

    def __enter__(self) -> SharedGpu:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a+b")
        if sys.platform == "win32":
            import msvcrt
            self._file.seek(0)
            while True:
                try:
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_LOCK, 1)  # gives up after 10 s: try again
                    break
                except OSError:
                    continue
        else:
            import fcntl
            fcntl.flock(self._file, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: object) -> None:
        if sys.platform == "win32":
            import msvcrt
            self._file.seek(0)
            msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._file, fcntl.LOCK_UN)
        self._file.close()
        self._file = None
