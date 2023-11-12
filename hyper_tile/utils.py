from __future__ import annotations

import gc
import torch
import random
import math
from functools import cache

RNG_INSTANCE = random.Random()

def set_seed(seed: int) -> None:
    RNG_INSTANCE.seed(seed)

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

    idx = RNG_INSTANCE.randint(0, len(ns) - 1)

    return ns[idx]

def iterative_closest_divisors(hw:int, aspect_ratio:float) -> tuple[int, int]:
    """
    Finds h and w such that h*w = hw and h/w = aspect_ratio
    We check all possible divisors of hw and return the closest to the aspect ratio
    """
    divisors = [i for i in range(2, hw + 1) if hw % i == 0] # all divisors of hw
    pairs = [(i, hw // i) for i in divisors] # all pairs of divisors of hw
    ratios = [w/h for h, w in pairs] # all ratios of pairs of divisors of hw
    closest_ratio = min(ratios, key=lambda x: abs(x - aspect_ratio)) # closest ratio to aspect_ratio
    closest_pair = pairs[ratios.index(closest_ratio)] # closest pair of divisors to aspect_ratio
    return closest_pair

@cache
def find_hw_candidates(hw:int, aspect_ratio:float) -> tuple[int, int]:
    """
    Finds h and w such that h*w = hw and h/w = aspect_ratio
    """
    h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))
    # find h and w such that h*w = hw and h/w = aspect_ratio
    if h * w != hw:
        w_candidate = hw / h
        # check if w is an integer
        if not w_candidate.is_integer():
            h_candidate = hw / w
            # check if h is an integer
            if not h_candidate.is_integer():
                return iterative_closest_divisors(hw, aspect_ratio)
            else:
                h = int(h_candidate)
        else:
            w = int(w_candidate)
    return h, w