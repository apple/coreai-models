# Vision-Language Models (VLMs)

Export recipes for vision-language models, exposed through the
`coreai.vlm.export` CLI. The model definitions live in
[`python/src/coreai_models/vlm/export.py`](../../python/src/coreai_models/vlm/export.py);
supported models are registered in its `SUPPORTED_MODELS` table.

## Supported models

| Short-name | HuggingFace ID            | Notes                                    |
|------------|---------------------------|------------------------------------------|
| `qwen3-vl` | `Qwen/Qwen3-VL-2B-Instruct` | 448×448 vision encoder, f16 text decoder |

## Exporting

```bash
uv run coreai.vlm.export --list-models          # list supported VLMs
uv run coreai.vlm.export qwen3-vl               # full bundle (text + vision)
uv run coreai.vlm.export qwen3-vl --skip-vision # text decoder + embedding only
```

Options:

- `--max-context-length N` — KV cache context length (default: 4096)
- `--num-layers N` — truncate the text decoder to N layers (debugging)
- `--output-dir DIR` — bundle output directory (default: `<repo-root>/exports/`)
- `--overwrite` — overwrite existing output

## Bundle layout

The export produces a `<name>.llmasset/` directory (`metadata.json` `kind=vlm`)
with asset roles consumed by the Swift runner's `ModelBundle`:

| Asset       | File             | Role                                            |
|-------------|------------------|-------------------------------------------------|
| `main`      | `<name>.aimodel` | Text decoder (`inputs_embeds`, stateful KV)     |
| `embedding` | `embed.aimodel`  | Token-embedding lookup (`input_ids → embeds`)   |
| `vision`    | `vision.aimodel` | Vision encoder (`pixel_values → image_features`)|
| —           | `tokenizer/`     | Embedded HuggingFace tokenizer                  |

## Adding a model

Add a `VLMSpec(...)` entry to `SUPPORTED_MODELS` in
[`vlm/export.py`](../../python/src/coreai_models/vlm/export.py) with the
HuggingFace ID, output name, image token id, and vision geometry (resolution,
patch/merge sizes, normalization stats). Models whose text decoder needs a
new architecture also require a class registered in
[`models/registry.py`](../../python/src/coreai_models/models/registry.py).

## Image preprocessing

The vision encoder expects a fixed-size square input. How an arbitrary image
reaches that square is controlled by `image_strategy` in `metadata.json`:

| Strategy      | Behavior                                 | Use when                             |
|---------------|------------------------------------------|--------------------------------------|
| `stretch`     | Resize directly to target size           | Default. Works for most models.      |
| `center_crop` | Shortest-edge resize, then center crop   | CLIP-based vision towers (FastVLM)   |
| `pad`         | Longest-edge resize, zero-pad remainder  | Models expecting preserved geometry  |

The strategy is inferred from the model's `preprocessor_config.json` at export
time and written into the bundle's `metadata.json`. Override at runtime:

```bash
llm-runner --model vlm_bundle --image photo.jpg --image-strategy center_crop
```

### Original resolution in prompt

Some models (Qwen-VL family) benefit from knowing the original image
dimensions. When `include_image_info` is set in the bundle metadata (or
overridden via `--image-info on`), the original `W×H` is prepended to the
text prompt before tokenization.
