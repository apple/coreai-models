# SAM 3

SAM 3 (Segment Anything Model 3) is a unified vision model from Meta for promptable image and video segmentation given text or visual prompts.[^1]

This export targets **iOS** by restructuring the model in a BC1S layout (channels-first, `Conv2d(1x1)` projections, fp16-safe primitives, rank-4 window attention) and splitting it into three independently optimizable functions:

| Function       | Compression                                | Inputs                               | Outputs                                                      |
|----------------|--------------------------------------------|--------------------------------------|--------------------------------------------------------------|
| `image_encode` | 4-bit k-means palettization (gs=32) + fp16 | `pixel_values`                       | `backbone_features`                                          |
| `text_encode`  | 6-bit k-means palettization (gs=8) + fp16  | `input_ids`                          | `text_features`                                              |
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

Saves to `<repo-root>/exports/<model>_lite_<image-size>_w<n-bits>_static/` as a bundle directory containing `<...>.aimodel`, a `tokenizer/` folder, and a `metadata.json` (segmenter bundle, schema 0.2). Pass `--output-dir <path>` to override the destination. If exporting the full model (vanilla HF model), saves to `<repo-root>/exports/<model>_<dtype>/`.

```sh
uv run export.py --help
```

**Options:**

| Flag                   | Description                                    | Default                |
|------------------------|------------------------------------------------|------------------------|
| `--full`               | Export plain HF `Sam3Model` (no iOS targeting) | —                      |
| `--output-dir`         | Output directory for the bundle                | `<repo-root>/exports/` |
| `--output-name`        | Custom bundle directory name                   | derived                |
| `--image-size`         | Input resolution (336 lite / 1008 full)        | `336` / `1008`         |
| `--max-text-seq-len`   | (lite) Static text sequence length             | `32`                   |
| `--n-bits`             | (lite) Uniform palettization bit-width override applied to BOTH encoders | asymmetric: image w4, text w6 |
| `--group-size`         | (lite) Uniform palettization group-size override applied to BOTH encoders | asymmetric: image gs32, text gs8 |
| `--dtype`              | (`--full`) Torch dtype: `float16` or `float32` | `float32`              |
| `--overwrite`          | Overwrite existing bundle                      | —                      |
| `--dry-run`            | Print resolved config and exit                 | —                      |

`image-size=336` is the resolution we recommend for iOS deployment.

### Lite export (targeting iOS)

The lite export, as shown in the WWDC26 [Dive into Core AI model authoring and optimization](https://developer.apple.com/videos/play/wwdc2026/325/) video, is the default (no extra flags needed). The model is restructured for iOS in BC1S layout with palettized encoders, and split into 3 independently compressed functions (image encoder, text encoder, detector) at a 336×336 input resolution.

### Full export (direct from Hugging Face)

Pass `--full` to skip iOS targeting and palettization and emit the unmodified `transformers.Sam3Model` as a single-entrypoint asset (one `main` function returning the five raw outputs). Higher quality than the lite variant at the cost of larger model size and slower on-device inference.

```sh
uv run models/sam3/export.py --full                    # float32, 1008x1008
uv run models/sam3/export.py --full --dtype float16
```

Full bundles land at `<repo-root>/exports/<model>_<dtype>/` (e.g. `exports/sam3_float32/`), so they sit next to the lite `sam3_lite_336_w4_static/` without colliding.

## Running

### In your iOS and macOS applications

```swift
import ImageSegmenter

// Load from a segmenter bundle directory (contains metadata.json, *.aimodel, and tokenizer/)
let segmenter = try await ImageSegmenter(resourcesAt: "coreai-models/exports/sam3_lite_336_w4_static")

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
