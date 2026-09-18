# DiffusionGemma

Google's DiffusionGemma-26B-A4B block-diffusion language model, exported for on-device inference via Core AI.

Unlike autoregressive models, DiffusionGemma generates a fixed-length token *canvas* by iterative denoising: a causal encoder prefills a KV cache, and a bidirectional decoder refines the canvas over several steps until it converges. It exports as two components — `encoder.aimodel` and `decoder.aimodel`.

## Supported Models

| Model                        | Parameters        | macOS | iOS |
| ---------------------------- | ----------------- | ----- | --- |
| DiffusionGemma 26B-A4B Instruct | 25.2B (3.8B active) | Yes   | No  |

## Gated Access

This model is gated on [Hugging Face](https://huggingface.co/google/diffusiongemma-26b-a4b-it). Accept the license, generate an HF token, and log in before exporting:
```bash
brew install hf
hf auth login --token <YOUR_TOKEN_HERE>
```

## Setup

If you haven't installed `uv`:
```bash
brew install uv
```

## Export

Export the encoder + decoder bundle (4-bit weight quantization is the practical target at this size):
```bash
uv run models/diffusion_gemma/export.py \
    --model google/diffusiongemma-26b-a4b-it \
    --canvas-length 256 --compression 4bit \
    --output-dir ./exports/
```
