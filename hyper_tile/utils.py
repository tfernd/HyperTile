from __future__ import annotations

import gc
import torch


def flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def possible_tile_sizes(
    dimension: int,
    tile_size: int,
    min_tile_size: int,
    tile_options: int,
) -> list[int]:
    assert tile_options >= 1

    min_tile_size = min(min_tile_size, tile_size, dimension)

    idx = torch.arange(min_tile_size, dimension + 1)
    divisors = idx[dimension == dimension // idx * idx]
    pos = divisors.sub(tile_size).abs().argsort()
    pos = pos[:tile_options]

    n = dimension // divisors[pos]

    return n.tolist()
