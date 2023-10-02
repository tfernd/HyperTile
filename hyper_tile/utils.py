from __future__ import annotations

import gc
import torch


def flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def auto_tile(value: int, chunk: int, min_chunk: int) -> int:
    """
    Automatically calculate the tiling size for a given `value` based on a `chunk` size.
    """

    min_chunk = min(min_chunk, chunk, value)

    idx = torch.arange(min_chunk, value + 1)
    divisors = idx[value == value // idx * idx]
    pos = divisors.sub(chunk).abs().argmin()

    return value // int(divisors[pos].item())
