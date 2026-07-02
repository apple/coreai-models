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
patch/merge sizes, CLIP normalization stats). Models whose text decoder needs a
new architecture also require a class registered in
[`models/registry.py`](../../python/src/coreai_models/models/registry.py).
