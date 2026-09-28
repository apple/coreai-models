# Sana Sprint

NVIDIA's Sana Sprint few-step text-to-image model for on-device image generation via Core AI.

## Supported Models

| Model            | Parameters                    | Resolution | Steps | macOS | iOS |
| ---------------- | ----------------------------- | ---------- | ----- | ----- | --- |
| Sana Sprint 0.6B | 0.6B DiT + 2.6B Gemma-2 + VAE | 1024x1024  | 1-4   | Yes   | No  |

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```bash
uv run coreai.diffusion.export sana-sprint-0.6b
```

This loads `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` with diffusers'
`SanaSprintPipeline` and exports in `bfloat16`. Other dtypes may be added in the future.

## Components

| Component      | Asset                 | Size   | Description                                                   |
| -------------- | --------------------- | ------ | ------------------------------------------------------------- |
| `transformer`  | `Transformer.aimodel` | 1.1 GB | Linear-attention DiT (28 blocks) on a 32x32x32 latent          |
| `text_encoder` | `TextEncoder.aimodel` | 4.9 GB | Gemma-2-2B; returns the 300 tokens the transformer attends to |
| `vae_decoder`  | `VAEDecoder.aimodel`  | 304 MB | DC-AE, 32x32 latent to 1024x1024 image                        |

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

The defaults (2 steps, guidance 4.5) come from `metadata.json`. `--guidance-scale` feeds the
transformer's guidance embedding; there is no classifier-free guidance pass. `--seed` matches
diffusers with `generator=torch.Generator("cpu").manual_seed(seed)`.
