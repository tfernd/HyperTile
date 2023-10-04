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

    # all divisors that are themselves divisible by 8 and give tile-size above min
    n = torch.arange(1, dimension + 1)
    n = n[dimension // n // 8 * 8 * n == dimension]
    n = n[dimension // n >= min_tile_size]

    pos = (dimension // n).sub(tile_size).abs().argsort()
    pos = pos[:tile_options]

    return n[pos].tolist()


def parse_list(x: list[int], /) -> str:
    if len(x) == 0:
        return str(x[0])
    return str(x)
