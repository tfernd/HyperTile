from __future__ import annotations
from typing import Callable

import logging
from functools import wraps
from contextlib import contextmanager

import random
import math
import torch.nn as nn
from einops import rearrange

from .utils import possible_tile_sizes, parse_list


@contextmanager
def split_attention(
    layer: nn.Module,
    /,
    height: int,
    width: int,
    *,
    tile_size: int = 256,
    min_tile_size: int = 256,  # 128 for VAE
    swap_size: int = 1,  # 2 for U-Net
    disable: bool = False,
    depth: int = 0,  # ! keep at 0
):
    # Hijacks AttnBlock from ldm and Attention from diffusers

    if disable:
        logging.info(f"Attention for {layer.__class__.__qualname__} not splitted")
        yield
        return

    # Aspect ratio
    ar = height / width

    # Possible sub-grids that fit into the image
    nhs = possible_tile_sizes(height, tile_size, min_tile_size, swap_size)
    nws = possible_tile_sizes(width, tile_size, min_tile_size, swap_size)

    # Random sub-grid indices # TODO remove randomness. seed?
    make_ns = lambda: (nhs[random.randint(0, len(nhs) - 1)], nws[random.randint(0, len(nws) - 1)])

    H, W = [height // s for s in nhs], [width // s for s in nws]
    print(
        f"Attention for {layer.__class__.__qualname__} split image of size {height}x{width} into {parse_list(nhs)}x{parse_list(nws)} tiles of sizes {parse_list(H)}x{parse_list(W)}"
    )

    def self_attn_forward(forward: Callable) -> Callable:
        @wraps(forward)
        def wrapper(*args, **kwargs):
            nh, nw = make_ns()

            x = args[0]
            if x.ndim == 4:  # VAE or U-Net
                if nh * nw > 1:
                    x = rearrange(x, "b c (nh h) (nw w) -> (b nh nw) c h w", nh=nh, nw=nw)

                out = forward(x, *args[1:], **kwargs)

                if nh * nw > 1:
                    out = rearrange(out, "(b nh nw) c h w -> b c (nh h) (nw w)", nh=nh, nw=nw)
            else:
                hw = x.size(1)
                h, w = round(math.sqrt(ar * hw)), round(math.sqrt(hw / ar))

                down_ratio = height // 8 // h
                curr_depth = round(math.log(down_ratio, 2))

                # Scale-up the tile-size the deeper we go
                nh = max(1, nh // down_ratio)
                nw = max(1, nw // down_ratio)

                do_split = curr_depth <= depth and h % nh == 0 and w % nw == 0 and nh * nw > 1

                if do_split:
                    # print((h, w), (nh, nw), curr_depth, down_ratio)
                    x = rearrange(x, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)

                out = forward(x, *args[1:], **kwargs)

                if do_split:
                    out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                    out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)

            return out

        return wrapper

    # Handle hikajing the forward methdo and recovering after
    try:
        for name, module in layer.named_modules():
            if module.__class__.__qualname__ in ("Attention", "CrossAttention", "AttnBlock"):
                # skip cross-attention layers
                if name.endswith("attn2") or name.endswith("attn_2"):
                    continue

                # save original forward for recovery later
                setattr(module, "_original_forward", module.forward)
                setattr(module, "forward", self_attn_forward(module.forward))
        yield
    finally:
        for name, module in layer.named_modules():
            # remove hijack
            if hasattr(module, "_original_forward"):
                setattr(module, "forward", module._original_forward)
                del module._original_forward
