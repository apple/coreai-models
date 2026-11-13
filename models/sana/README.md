# Sana Sprint

NVIDIA's Sana Sprint few-step text-to-image model for on-device image generation via Core AI.

## Supported Models

| Model            | Parameters                    | Resolution | Steps | macOS | iOS |
| ---------------- | ----------------------------- | ---------- | ----- | ----- | --- |
| Sana Sprint 0.6B | 0.6B DiT + 2.6B Gemma-2 + VAE | 1024x1024  | 1-4   | Yes   | No  |
| Sana Sprint 1.6B | 1.6B DiT + 2.6B Gemma-2 + VAE | 1024x1024  | 1-4   | Yes   | No  |

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```bash
uv run coreai.diffusion.export sana-sprint-0.6b
uv run coreai.diffusion.export sana-sprint-1.6b
```

These load `Efficient-Large-Model/Sana_Sprint_{0.6B,1.6B}_1024px_diffusers` with diffusers'
`SanaSprintPipeline` and export in `bfloat16` with the `t8bit-te4bit-asym` compression preset:

- **Transformer:** INT8 per-channel weights. INT4 here visibly degrades fine detail and text rendering.
- **Text encoder:** INT4 asymmetric per-block weights, exported in `float16`, since the Core AI GPU
  INT4 kernels need `float16` scales. INT4 here shifts compositions without degrading them.
- **VAE decoder:** not quantized.

For the full-precision `bfloat16` export, pass `--compression none`:

```bash
uv run coreai.diffusion.export sana-sprint-1.6b --compression none
```

### Neural Engine (0.6B)

```bash
uv run coreai.diffusion.export sana-sprint-0.6b-neural-engine
```

This writes `exports/Sana_Sprint_0.6B_1024px_diffusers_neural_engine/` with all three components
re-authored for the Neural Engine in `float16` (`python/src/coreai_models/models/ios/sana_*.py`).
The preset's `te6bit-tok8-tx8bit-pal` compression palettizes the text encoder to 6 bits (k-means,
one lookup table per 16 output channels) with its token table in `int8` (one scale per row), and the
transformer to 8 bits (k-means, one lookup table per tensor; the timestep, guidance, caption and
input/output projections stay `float16`). The VAE decoder stays uncompressed. Pass
`--compression te6bit-pal` for a `float16` token table and dense transformer, or `--compression none`
for no compression. The bundle's `metadata.json` maps each component to `["neural_engine"]` under
`diffusion.compute_units`, and the runtime loads those on the Neural Engine; the token lookup runs
on the GPU. Inputs, outputs and the Swift API are the same as for the GPU bundle.

`float16` keeps 3 more mantissa bits than `bfloat16`, and each component holds its activations in range
with exact rewrites, so it is closer to the `float32` reference than the GPU export (PSNR on 8 prompts,
M4 Pro):

| Component / output | Neural Engine | Dense `bfloat16` | Default GPU bundle |
| ------------------ | ------------- | ---------------- | ------------------ |
| Transformer        | 45.9 dB       | 42.3 dB          | 40.3 dB (INT8)     |
| Text encoder       | 53.9 dB       | 71.3 dB          | 46.7 dB (INT4)     |
| VAE decoder        | 63.9 dB       | 49.1 dB          | 49.1 dB            |
| Image (end to end) | 26.1 dB       | —                | 21.3 dB            |

The 8-bit transformer and the `int8` token table cost their components 8.8 and 0.2 dB (dense
transformer 54.7 dB, `float16` table 54.1 dB) but leave the image within 0.1 dB of `te6bit-pal`. With
`--compression none` the text encoder is 58.2 dB and the image 28.6 dB.

| Asset              | Neural Engine | Default GPU bundle |
| ------------------ | ------------- | ------------------ |
| Transformer        | 605 MB        | 570 MB             |
| Text encoder       | 2.0 GB        | 1.4 GB             |
| VAE decoder        | 466 MB        | 304 MB             |

Compute per image is on par with the GPU (~2.1 s on an M4 Pro). Loading weights onto the Neural
Engine runs at ~0.5 GB/s, so with lazy model loading an image takes ~4 s against ~3 s on the GPU
(~12 s more with the dense text encoder). A transformer step takes 195 ms (209 ms dense). The first
load specializes for the Neural Engine, ~10 minutes for the three components; later loads use the
cache.

## Components

| Component      | Asset                 | Size (default)                | Size (`--compression none`)   | Description                                                   |
| -------------- | --------------------- | ----------------------------- | ----------------------------- | ------------------------------------------------------------- |
| `transformer`  | `Transformer.aimodel` | 570 MB (0.6B) / 1.5 GB (1.6B) | 1.1 GB (0.6B) / 3.0 GB (1.6B) | Linear-attention DiT (28 / 20 blocks) on a 32x32x32 latent    |
| `text_encoder` | `TextEncoder.aimodel` | 1.4 GB                        | 4.9 GB                        | Gemma-2-2B; returns the 300 tokens the transformer attends to |
| `vae_decoder`  | `VAEDecoder.aimodel`  | 304 MB                        | 304 MB                        | DC-AE, 32x32 latent to 1024x1024 image                        |

Both variants share the text encoder, VAE, tokenizer, and sampler settings; only the transformer differs.

The bundle also has `tokenizer/` and `metadata.json`. The metadata carries the instruction prefix that
diffusers prepends to every prompt, and the timestep schedule.

## Running

### In your macOS applications

```swift
import CoreAIDiffusionPipeline

let pipeline = try await FlowTransformerPipeline(from: modelURL)

let config = PipelineConfiguration(
    prompt: "a cyberpunk cat with a neon sign that says \"Sana\"",
    seed: 42,
    stepCount: 2,
    guidanceScale: 4.5,
    schedulerType: .discreteFlow
)

let image = try await pipeline.generateImages(configuration: config).images.first!
```

`FlowTransformerPipeline` also runs FLUX.2 bundles and picks the model family from
`metadata.json`. Sana Sprint supports text-to-image at full resolution only.

### On your Mac using built-in Command Line Tool

```bash
swift run -c release diffusion-runner --model exports/Sana_Sprint_0.6B_1024px_diffusers \
    --prompt "a cyberpunk cat with a neon sign that says \"Sana\"" --seed 42
```

For 1.6B, pass `--model exports/Sana_Sprint_1.6B_1024px_diffusers`.

The defaults (2 steps, guidance 4.5) come from `metadata.json`. `--guidance-scale` feeds the
transformer's guidance embedding; there is no classifier-free guidance pass. `--seed` matches
diffusers with `generator=torch.Generator("cpu").manual_seed(seed)`.
