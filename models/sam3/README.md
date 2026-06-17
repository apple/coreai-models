# SAM 3

SAM 3 (Segment Anything Model 3) is a unified vision model from Meta for promptable image and video segmentation given text or visual prompts.[^1]

This export targets the **Apple Neural Engine** by re-authoring the model in a BC1S layout (channels-first, `Conv2d(1x1)` projections, fp16-safe primitives, rank-4 window attention) and splitting it into three independently optimizable functions:

| Function       | Compression                                | Inputs                               | Outputs                                                      |
|----------------|--------------------------------------------|--------------------------------------|--------------------------------------------------------------|
| `image_encode` | 4-bit k-means palettization (gs=32) + fp16 | `pixel_values`                       | `backbone_features`                                          |
| `text_encode`  | 6-bit k-means palettization (gs=8) + fp16  | `input_ids`, `attention_mask`        | `text_features`                                              |
| `detect`       | fp16 (no weight compression)               | `backbone_features`, `text_features` | `pred_masks`, `pred_boxes`, `pred_logits`, `presence_logits` |

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

### Gated Access
SAM3 is a gated model on [Hugging Face](https://huggingface.co/facebook/sam3) (HF). You will need to accept the terms of the [license](https://huggingface.co/facebook/sam3), generate a HF token, and add your HF token to your machine before exporting this model.
```bash
brew install hf
hf auth login --token <YOUR_TOKEN_HERE>
```

## Export

```sh
uv run export.py
```

Saves to `<repo-root>/exports/<model>_reauthored_<image-size>_w<n-bits>_static/` as a bundle directory containing `<...>.aimodel`, a `tokenizer/` folder, and a `metadata.json` (segmenter bundle, schema 0.2). Pass `--output-dir <path>` to override the destination. If exporting the baseline (vanilla HF model), saves to `<repo-root>/exports/<model>_<dtype>/`.

```sh
uv run export.py --help
```

**Options:**

| Flag                   | Description                                    | Default                |
|------------------------|------------------------------------------------|------------------------|
| `--baseline`           | Export plain HF `Sam3Model` (no re-authoring) | —                      |
| `--output-dir`         | Output directory for the bundle                | `<repo-root>/exports/` |
| `--output-name`        | Custom bundle directory name                   | derived                |
| `--image-size`         | Input resolution (336 optimized / 1008 baseline) | `336` / `1008`       |
| `--max-text-seq-len`   | (optimized) Static text sequence length        | `32`                   |
| `--n-bits`             | (optimized) Uniform palettization bit-width override applied to BOTH encoders | asymmetric: image w4, text w6 |
| `--group-size`         | (optimized) Uniform palettization group-size override applied to BOTH encoders | asymmetric: image gs32, text gs8 |
| `--dtype`              | (`--baseline`) Torch dtype: `float16` or `float32` | `float32`           |
| `--overwrite`          | Overwrite existing bundle                      | —                      |
| `--dry-run`            | Print resolved config and exit                 | —                      |

`image-size=336` keeps the global-attention sequence small enough to fit Neural Engine SRAM; it is the resolution we recommend for ANE deployment.

### ANE compatibility

The optimized export deliberately leaves `enable_per_channel_scale=False` (the coreai-opt default). The WWDC demo notebook used `enable_per_channel_scale=True` because per-channel scale produces visually nicer outputs in PyTorch — but in MPS lowering it becomes `mps.dequantize_lut` ops with rank-6 LUT tensors (`[num_groups, 1, 1, 1, num_palettes, 1]`). ANE hardware caps tensors at rank 5; with hundreds of such ops in image_encode + text_encode the ANE compiler errors out (`Too many fvmlibs (more than 255)`) and the runtime silently falls back to GPU. The fall-back is *slower than the baseline export* because palettized weights pay the dequantization cost without the LUT-cache speedup ANE provides. Keeping per-channel scale off keeps the asset compilable on ANE for the price of a small PyTorch-side quality regression.

### Baseline export (parity reference)

Pass `--baseline` to skip re-authoring and palettization and emit the unmodified `transformers.Sam3Model` as a single-entrypoint asset (one `main` function returning the five raw outputs). Useful for parity / regression checks against the optimized export.

```sh
uv run models/sam3/export.py --baseline                    # float32, 1008x1008
uv run models/sam3/export.py --baseline --dtype float16
```

Baseline bundles land at `<repo-root>/exports/<model>_<dtype>/` (e.g. `exports/sam3_float32/`), so they sit next to the optimized `sam3_reauthored_336_w4_static/` without colliding.

## Running

## In your iOS and macOS applications

```swift
import ImageSegmenter

// Load from a segmenter bundle directory (contains metadata.json, *.aimodel, and tokenizer/)
let segmenter = try await ImageSegmenter(resourcesAt: "coreai-models/exports/sam3_reauthored_336_w4_static")

// Text prompt (SAM3):
let segments = try await segmenter.segment(image: cgImage, prompt: "cat")
```

### On your Mac using built-in Command Line Tool
```bash
swift run -c release image-segmenter --model path/to/exported_model_folder --prompt "cat" --image path/to/image.jpg
```

## Supported models

| Model         | Parameters |
|---------------|------------|
| facebook/sam3 | 848M       |

[^1]: [Paper](https://arxiv.org/abs/2511.16719) · [HuggingFace](https://huggingface.co/facebook/sam3)
