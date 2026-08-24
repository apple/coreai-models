# Wan 2.1

Wan AI's text-to-video diffusion models for on-device video generation via Core AI.

## Supported Models

| Model                  | Parameters | Resolution | Frames |
|------------------------|------------|------------|--------|
| Wan 2.1 T2V 1.3B      | 1.3B       | 480x832    | 17-81  |

## Setup

```bash
brew install uv
```

## Export

```bash
# Full precision (fp16)
uv run coreai.diffusion.export Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# 4-bit quantized (~5 GB, recommended for constrained devices)
uv run coreai.diffusion.export Wan-AI/Wan2.1-T2V-1.3B-Diffusers --compression 4bit-asym

# 8-bit quantized (near-lossless)
uv run coreai.diffusion.export Wan-AI/Wan2.1-T2V-1.3B-Diffusers --compression 8bit

# Preview config without exporting
uv run coreai.diffusion.export Wan-AI/Wan2.1-T2V-1.3B-Diffusers --dry-run
```

## Run

```bash
# Full quality (50 steps, 81 frames / 5 seconds)
videodiffusion-runner --model exports/Wan2.1-T2V-1.3B-Diffusers \
    --quality best --prompt "A cat walking on grass"

# Balanced (30 steps, cfg-cutoff for speed)
videodiffusion-runner --model exports/Wan2.1-T2V-1.3B-Diffusers \
    --quality balanced --prompt "Ocean waves crashing on rocks"

# Fast preview (12 steps, 33 frames / ~2 seconds)
videodiffusion-runner --model exports/Wan2.1-T2V-1.3B-Diffusers \
    --quality fast --prompt "A butterfly landing on a flower"
```

## Quality Presets

| Preset       | Steps | cfg-cutoff | Frames |
|--------------|-------|------------|--------|
| `best`       | 50    | none       | 81     |
| `balanced`   | 30    | 0.5        | 81     |
| `fast`       | 12    | 0.5        | 33     |

## Options

| Flag              | Description                                    |
|-------------------|------------------------------------------------|
| `--quality`       | Preset: `fast`, `balanced`, `best`             |
| `--steps`         | Override denoising steps (default: 50)          |
| `--num-frames`    | Output frames: 17, 33, 49, 65, or 81           |
| `--duration`      | Alternative to --num-frames (seconds, max 5)   |
| `--seed`          | Random seed (default: 42)                      |
| `--guidance-scale`| CFG guidance scale (default: 5.0)              |
| `--cfg-cutoff`    | Skip unconditional pass for final N% of steps  |
| `--output`        | Output file path (default: output.mp4)         |

## Compression

| Preset       | Transformer Size | Notes                      |
|--------------|------------------|----------------------------|
| none (fp16)  | 2.7 GB           | Baseline quality           |
| 8bit         | 1.4 GB           | Near-lossless              |
| 4bit         | 803 MB           | Symmetric INT4             |
| 4bit-asym    | 818 MB           | Asymmetric INT4            |

## Using in an App

Add the `CoreAIVideoDiffusionPipeline` library to your Swift package dependencies:

```swift
.product(name: "CoreAIVideoDiffusion", package: "coreai-models")
```

Then import and use:

```swift
import CoreAIVideoDiffusionPipeline

let pipeline = try await WanPipeline(from: modelURL)
let result = try await pipeline.generateVideo(
    configuration: .from(preset: .balanced, prompt: "A cat walking on grass")
) { progress in true }
```
