from __future__ import annotations

import gc
import torch
import random


def flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def random_divisor(value: int, min_value: int, /, max_options: int = 1) -> int:
    min_value = min(min_value, value)

    # All big divisors of value (inclusive)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0]

    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element

    idx = random.randint(0, len(ns) - 1)

    return ns[idx]
