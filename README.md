# HyperTile: 3-4x Stable-Diffusion acceleration at 4K resolution!

**TL-DR**: HyperTile accelerates Stable-Diffusion by tiling the self-attention in the U-Net and VAE at the first depth, avoiding tile-overlap. It maintains full global picture attention even at 4K resolution, resulting in a 3-4x speed increase.

**Note**: Please note that the code for this project will be made available by October 8, 2023!

## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Approach](#approach)
  - [Implications](#implications)
- [Performance](#performance)
- [To-Do List](#To-Do)

# Introduction

HyperTile's primary objective is to enhance the image generation capabilities of [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion) by enabling the use of 4K resolution. For those acquainted with Stable-Diffusion, it is recognized as a potent tool for producing high-fidelity images. Nonetheless, as is typical with groundbreaking technologies, it encounters obstacles, particularly when confronted with the demands of handling substantial image resolutions, such as 4K or anything exceeding 1K.

In this context, we will show how HyperTile effectively tackles these challenges, yielding a remarkable 3-4x sped-up in Stable-Diffusion's image generation performance at 4K resolutions, all the while upholding image quality. Prior to delving into the technical intricacies, it is imperative to establish the project's rationale and draw insights from existing methodologies that have informed our approach.

## Motivation

- [Flash-Attention](https://github.com/Dao-AILab/flash-attention) revolutionized the field by introducing faster attention mechanisms, making it feasible to train models on consumer-grade graphics cards.

- In a parallel effort, [TOME](https://github.com/dbolya/tomesd) aimed to expedite SD image generation by reducing the number of "redundant" tokens, resulting in the removal of up to 50% of tokens. While this approach achieved a slight speed increase, it also led to a degradation in image generation quality.

- On a different front, [MultiDiffusion](https://multidiffusion.github.io/), also known as tiled-diffusion, adopted a novel strategy. Recognizing the computational challenges of computing the full attention matrix, this approach divides the image into distinct "tiles" of fixed size and diffuses these tiles individually. Subsequently, a clever method is employed to reassemble the tiles, reconstructing the complete image by leveraging the overlap between them. [Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers) pursues a similar approach with subtle variations.

This raises a couple of pivotal questions:

- Is it possible to find a middle ground between token pruning and preserving image fidelity?
- Can we attain the synthesis of high-resolution images without sacrificing computational efficiency?

These queries serve as the impetus for our investigation into "HyperTile," a methodology that combines insights from token reduction strategies inspired by TOME (albeit with distinct approaches) and the tiling concept from MultiDiffusion.

## Approach

In Stable-Diffusion, specifically version 1.5, the training initially employed $256 \times 256$ images and later extended to $512 \times 512$. Consequently, the model encounters difficulties when pushed beyond these dimensions. The constraint on larger image training stemmed from the absence of Flash-Attention and the associated high computational cost (a challenge addressed by SD-XL).

The U-Net architecture within Stable-Diffusion reduces image (latents) resolution (tokens) by a factor of 2 (4) at each depth. As a result, attention is computed at resolutions corresponding to $1/(1, 2, 4, 8)^2$ of the original latent dimension (square because attention relies on image width times height, tokens). Consequently, only the initial U-Net depth requires significant attention computation, while subsequent depths handle lower resolutions with minimal computational overhead.

This prompts several critical questions: How can we address this challenge? How can we enhance the process? Is it possible to generate 4K images using Stable-Diffusion while relying on consumer-grade graphics cards?

To tackle this issue, we propose a fusion of concepts drawn from TOME and tiled-diffusion. Our approach involves tiling the query, key, and values exclusively at the initial depth before attention computation and subsequently re-assmpling the data. Essentially, this process transforms the shape from $(b, c, [h, w])$ to $([b, nh, nw], c, [h/nh, w/nw])$. Here, the brackets operator denotes dimension concatenation, with b representing batch, c representing channels, h for height, w for width, and nh/nw indicating the number of chunks in the height/width dimensions. This approach confines attention computation to a localized tile.

You might wonder about potential overlap issues. Since pixels attend only to content within their respective tiles, one might expect overlap concerns. However, this is not the case. At other depths in the model, the entirety of the image is considered, ensuring that attention possesses a holistic view of the image. Consequently, there is no overlap concern to contend with.

### Implications

What does this mean, and why does it matter? The U-Net's four depth levels play a crucial role in enabling the neural network to understand images both locally and globally. At the initial depth, pixels communicate with their nearby counterparts, fostering local awareness. However, as we move to higher levels, the network gains a broader perspective of the entire image. This leads to a critical question: is it necessary for pixels to engage in long-range interactions with distant ones? As we approach the final stages of image generation, pixels mainly focus on a small neighborhood, considering factors like color and shape. Therefore, we can safely eliminate long-range pixel interactions without degrading image quality.

What are the practical implications of these insights? Firstly, it becomes clear that we can generate 4K images using Stable-Diffusion with SD 1.5, and the larger the image, the more speedup we achieve. Additionally, we can now fine-tune Stable-Diffusion with 2K, 2.5K, 3K, or 4K images using consumer-grade graphics cards. This eliminates the issues of duplicated and cloned heads in image generation, which arose because SD was initially trained on $512 \times 512$ resolution images and struggled with larger ones, primarily due to an "overlap" problem, in a sense.

## Performance

In this performance evaluation, I conducted three image generation experiments, each consisting of 30 steps. I used the diffusers backend in PyTorch 2.0.1, with the assistance of [SDPA](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html). The images generated are square, and their dimensions vary along the x-axis. The black dots represent speed measurements without tiling, while all other colored dots consist of tiles, with each dot corresponding to a specific ratio of the dimension (size/chunk), maintaining a minimum tile size of 128.

![Average Speed](assets/Average-Speed.jpg)

The subsequent graph illustrates the speed-up achieved for each tile-ratio. As the target image dimension increases, the potential speed-up becomes more substantial.

![Speed-Up](assets/SpeedUp.jpg)

It's important to note that, currently, I have exclusively tested with the diffusers backend due to its superior performance. Additionally, there is currently no LoRA model available for HD resolution that is compatible with diffusers. Consequently, text-to-image generation, whether tiled or non-tiled, may exhibit aberrations. Addressing this issue necessitates the development of a fine-tuned LoRA model specifically tailored for high-resolution images with a Hyper-Tiled enabled.

## To-Do

You're convinced, and now you're wondering how to put it into action? Rest assured, I'll be sharing the complete code shortly. In essence, it offers a context manager that you can use to encapsulate your U-Net and VAE, effortlessly managing the tiling process without overlaps or sequential evolution.

**TODO List:**

- [ ] Add examples
- [ ] Share the Code
- [ ] Develop an Automatic1111 WebUI Extension for Hyper-Tile
- [ ] Initiate a Fork and Submit a Pull Request to Kohya-ss for Training Large Image LoRAS

Stay tuned for the forthcoming code release, by 8/10/2023!
