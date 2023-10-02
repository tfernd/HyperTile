from __future__ import annotations
from typing import Callable, Literal

import logging
from functools import wraps
from contextlib import contextmanager

import math
from einops import rearrange

from diffusers import UNet2DConditionModel, AutoencoderKL

from .utils import auto_tile


@contextmanager
def split_attention(
    # TODO add support for CompVis
    layer: UNet2DConditionModel | AutoencoderKL,
    /,
    height: int,
    width: int,
    *,
    # TODO make this into a list, so we select depth by chunks
    chunk: int = 384,
    min_chunk: int = 384, 
    # TODO remove depth in favor of chunk list
    depth: Literal[0, 1, 2, 3] = 0,  # ! >= 1 produces garbage! LoRA can fix it? bigger scaling chunks?
    disable: bool = False,
):
    """
    A context manager for splitting attention within a Stable-Diffusion layer.

    Args:
        layer (UNet2DConditionModel | AutoencoderKL): The Stable-Diffusion layer to apply attention splitting to.
        height (int): The height of the input data.
        width (int): The width of the input data.
        chunk (int, optional): The size of the chunks for tiling attention. Default is 384.
        min_chunk (int, optional): The minimum chunk size. Chunks smaller than this are not recommended. Default is 384.
        depth (Literal[0, 1, 2, 3], optional): The depth of the attention splitting. Default is 0.
        disable (bool, optional): Disable attention splitting. Default is False.

    Yields:
        None: When attention splitting is disabled or not applicable for the given layer.

    Raises:
        ValueError: If an unsupported layer type is provided.

    Example:
        with split_attention(my_layer, height=512, width=512, chunk=256):
            ...
            # Perform operations with attention splitting.
        # Attention splitting is automatically restored after the context.
    """

    # Aspect ratio
    ar = height / width

    nh = auto_tile(height, chunk, min_chunk)
    nw = auto_tile(width, chunk, min_chunk)

    if nh * nw == 1 or disable:
        logging.info(f"Attention for {layer.__class__.__qualname__} not splitted")
        yield
        return

    logging.info(f"Attention for {layer.__class__.__qualname__} split into {nh}x{nw} parts of size {height//nh}x{width//nw}")

    def vae_attn_hijack(fun: Callable):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            # yes, it's just that!
            x = rearrange(args[0], "b c (nh h) (nw w) -> (b nh nw) c h w", nh=nh, nw=nw)
            out = fun(x, *args[1:], **kwargs)
            out = rearrange(out, "(b nh nw) c h w -> b c (nh h) (nw w)", nh=nh, nw=nw)

            return out

        return wrapper

    def unet_attn1_hijack(fun: Callable):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            x = args[0]

            # Find size of the downsampled latents
            hw = x.size(1)
            h = round(math.sqrt(ar * hw))
            w = round(h / ar)
            assert h * w == hw

            # Find the downsampling depth
            curr_depth = round(math.log(height // h // 8, 2))  # 8 is specific for Stable-Diffusion latents downsampling
            assert curr_depth in (0, 1, 2, 3)

            # Can this layer be splitted?
            can_split = curr_depth <= depth and h % nh == 0 and w % nw == 0  # ? why the latter would be the case?

            if can_split:
                # separate hw, split h and w so we merge the factors to the batch dimension
                x = rearrange(x, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)

            out = fun(x, *args[1:], **kwargs)

            if can_split:
                # Reverse
                out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)

            return out

        return wrapper

    try:
        for name, module in layer.named_modules():
            # unet self-attention or VAE
            is_good_layer = name.endswith("attn1") or name.endswith("mid_block.attentions.0")

            if module.__class__.__qualname__ == "Attention" and is_good_layer:
                setattr(module, "_original_forward", module.forward)

                # Hijack forward method of VAE and U-Net
                if isinstance(layer, UNet2DConditionModel):
                    setattr(module, "forward", unet_attn1_hijack(module.forward))
                elif isinstance(layer, AutoencoderKL):
                    setattr(module, "forward", vae_attn_hijack(module.forward))
                else:
                    raise ValueError("!")
        yield
    except Exception as e:
        logging.error(f"Failed to split attention: {e}")
        raise e
    finally:
        for name, module in layer.named_modules():
            # Remove hijack
            if hasattr(module, "_original_forward"):
                setattr(module, "forward", module._original_forward)
                del module._original_forward
