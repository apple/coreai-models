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

The uniform `4bit` and `4bit-asym` presets are rejected for these models: INT4 weights quantized from
`bfloat16` export fine but fail at runtime on the GPU.

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
