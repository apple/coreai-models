# SAM 3

SAM 3 (Segment Anything Model 3) is a unified vision model from Meta for promptable image and video segmentation given text or visual prompts.[^1]

This export targets the **Apple Neural Engine** by re-authoring the model in a BC1S layout (channels-first, `Conv2d(1x1)` projections, fp16-safe primitives, rank-4 window attention) and splitting it into three independently optimizable functions:

| Function       | Compression                              | Inputs                          | Outputs                                                    |
|----------------|------------------------------------------|---------------------------------|------------------------------------------------------------|
| `image_encode` | 4-bit k-means palettization + fp16       | `pixel_values`                  | `backbone_features`                                        |
| `text_encode`  | 4-bit k-means palettization + fp16       | `input_ids`, `attention_mask`   | `text_features`                                            |
| `detect`       | fp16 (no weight compression)             | `backbone_features`, `text_features` | `pred_masks`, `pred_boxes`, `pred_logits`, `presence_logits` |

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

SAM3 needs `transformers>=5.5.4,<5.10.1`, which doesn't overlap with the workspace's `transformers>=4.57,<5.0` pin. To avoid a workspace-wide split, this script ships as a [PEP 723 inline-script](https://peps.python.org/pep-0723/) — its dependencies (including the SAM3 transformers pin) are declared in a header at the top of `models/sam3/export.py` and uv resolves them in an isolated per-script env. No `uv sync` step is needed.

```sh
uv run models/sam3/export.py
```

Saves to `<repo-root>/exports/<model>_reauthored_<image-size>_w<n-bits>_static/` as a bundle directory containing `<...>.aimodel`, a `tokenizer/` folder, and a `metadata.json` (segmenter bundle, schema 0.2). Pass `--output-dir <path>` to override the destination.

```sh
uv run models/sam3/export.py --help
```

**Options:**

| Flag                   | Description                                    | Default                |
|------------------------|------------------------------------------------|------------------------|
| `--baseline`           | Export plain HF `Sam3Model` (no re-authoring) | —                      |
| `--output-dir`         | Output directory for the bundle                | `<repo-root>/exports/` |
| `--output-name`        | Custom bundle directory name                   | derived                |
| `--image-size`         | Input resolution (336 optimized / 1008 baseline) | `336` / `1008`       |
| `--max-text-seq-len`   | (optimized) Static text sequence length        | `32`                   |
| `--n-bits`             | (optimized) K-means palettization bit-width    | `4`                    |
| `--group-size`         | (optimized) Per-grouped-channel group size     | `16`                   |
| `--dtype`              | (`--baseline`) Torch dtype: `float16` or `float32` | `float32`           |
| `--overwrite`          | Overwrite existing bundle                      | —                      |
| `--dry-run`            | Print resolved config and exit                 | —                      |

`image-size=336` keeps the global-attention sequence small enough to fit Neural Engine SRAM; it is the resolution we recommend for ANE deployment.

### Baseline export (parity reference)

Pass `--baseline` to skip re-authoring and palettization and emit the unmodified `transformers.Sam3Model` as a single-entrypoint asset (one `main` function returning the five raw outputs). Useful for parity / regression checks against the optimized export.

```sh
uv run models/sam3/export.py --baseline                    # float32, 1008x1008
uv run models/sam3/export.py --baseline --dtype float16
```

Baseline bundles land at `<repo-root>/exports/<model>_<dtype>/` (e.g. `exports/sam3_float32/`), so they sit next to the optimized `sam3_reauthored_336_w4_static/` without colliding.

## Running (pure Python)

After exporting, you can run the model end-to-end on the host (loaded via `coreai.runtime`). `models/sam3/run.py` is also a PEP 723 inline-script, so it resolves its own SAM3-compatible env:

```sh
uv run models/sam3/run.py --image path/to/image.jpg --prompt "flower"
```

The runner picks the most recently exported bundle in `exports/` whose name starts with `sam3_reauthored_`. Pass `--bundle path/to/bundle/` to point at a specific bundle, and `--output result.png` to override where the annotated PNG is written. Defaults to Neural Engine specialization; pass `--compute-unit gpu|cpu` to override.

## Running on iOS / macOS

```swift
import ImageSegmenter

// Load from a segmenter bundle directory (contains metadata.json, *.aimodel, and tokenizer/)
let segmenter = try await ImageSegmenter(resourcesAt: "coreai-models/exports/sam3_reauthored_336_w4_static")

// Text prompt (SAM3):
let segments = try await segmenter.segment(image: cgImage, prompt: "cat")
```

```bash
swift run -c release image-segmenter --model path/to/exported_model_folder --prompt "cat" --image path/to/image.jpg
```

## Supported models

| Model         | Parameters |
|---------------|------------|
| facebook/sam3 | 848M       |

[^1]: [Paper](https://arxiv.org/abs/2511.16719) · [HuggingFace](https://huggingface.co/facebook/sam3)
