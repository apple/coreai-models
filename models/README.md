# Models

This directory contains export recipes for converting supported open-source models to Core AI `.aimodel` format.

Only models listed in the catalog below or registered in the [model registry](../python/src/coreai_models/model_registry.py) are supported.

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Exporting Supported Models

### Listing Available Models

```bash
uv run coreai.model.registry --list-models --type llm               # all LLM presets
uv run coreai.model.registry --list-models --type llm --platform macOS # macOS only
uv run coreai.model.registry --list-models --type diffusion         # diffusion models
```

### Language Models

```bash
uv run coreai.llm.export Qwen/Qwen3-0.6B                 # macOS (default)
uv run coreai.llm.export Qwen/Qwen3-0.6B --platform iOS  # iOS
```

The export tool resolves compression, precision, and context length automatically for known models.

To try exporting a model that has Python source but no registry preset, use `--experimental`:

```bash
uv run coreai.llm.export org/NewModel \
    --experimental \
    --compute-precision float16 \
    --compression 4bit \
    --max-context-length 4096
```

#### Quantization Options

| Platform | Preset                                     | Description                                    |
|----------|--------------------------------------------|------------------------------------------------|
| macOS    | `4bit` (default)                           | INT4 weight-only, block size 32 (all layers)   |
| macOS    | `none`                                     | Full precision                                 |
| iOS      | `4bit_weight_palettized_group32` (default) | 4-bit palettization with channel group size 32 |
| iOS      | `4bit_weight_palettized_group8`            | 4-bit palettization with channel group size 8  |
| iOS      | `none`                                     | Full precision                                 |

**Note:** All `iOS` palettization presets quantize the Embedding to 8-bit per tensor by default.

Override the default with `--compression`:

```bash
uv run coreai.llm.export Qwen/Qwen3-0.6B --compression none                        # full precision
uv run coreai.llm.export Qwen/Qwen3-0.6B --platform iOS --compression 4bit_weight_palettized_group8
```

##### Specifying Compression Configs via YAML files

Specialized compression recipes that aren't covered by pre-defined presets can be specified as YAML files using the `--compression-config` option with the path to a [coreai-opt](https://github.com/apple/coreai-optimization) config.
This option should be used instead of `--compression` which is specifically for presets.

`--compression-config` takes a path to a YAML file:

```bash
uv run coreai.llm.export Qwen/Qwen3-0.6B --platform iOS \
    --compression-config my_custom_recipe.yaml
```

For more details on compression configurations, please refer to the [coreai-opt documentation](https://apple.github.io/coreai-optimization/introduction/how_to_use_coreaiopt.html).

Custom mixed precision compression recipes for some models are available alongside the respective model card under `models/<family>/` (for example, [`models/qwen3/qwen3_0_6b_mixed_4bit_8bit.yaml`](qwen3/qwen3_0_6b_mixed_4bit_8bit.yaml)). Some registry presets (e.g. `qwen3-0.6b` iOS) use one of these YAMLs by default. For instance `uv run coreai.llm.export qwen3-0.6b --platform iOS` already uses the right compression recipe without needing to pass in `--compression-config`.

#### Context Length

macOS models use dynamic KV cache and default to the model's maximum supported context. iOS models require a fixed context length at export time.

```bash
# macOS: omit for full model context, or cap it to reduce memory
uv run coreai.llm.export Qwen/Qwen3-0.6B --max-context-length 4096

# iOS: required (static shapes)
uv run coreai.llm.export Qwen/Qwen3-0.6B --platform iOS --max-context-length 4096
```

### Diffusion Models

```bash
uv run coreai.diffusion.export stabilityai/stable-diffusion-3.5-medium
uv run coreai.diffusion.export black-forest-labs/FLUX.2-klein-4B
```

### Vision-Language Models (VLMs)

```bash
uv run coreai.vlm.export --list-models   # list supported VLMs
uv run coreai.vlm.export qwen3-vl        # text decoder + token embedding + vision encoder
```

This produces a single `<name>.llmasset/` bundle (`kind=vlm`) holding the text
decoder (`main`), token-embedding lookup (`embedding`), vision encoder
(`vision`), tokenizer, and `metadata.json`. Pass `--skip-vision` to export the
text portion only.

### Standalone Export Scripts

Models with a standalone `export.py` are run directly:

```bash
uv run models/<name>/export.py
```

## Model Catalog

### Language Models (LLMs)

- [Gemma 3](gemma3)
- [GPT-OSS](gpt_oss)
- [Mistral](mistral)
- [Mixtral](mixtral)
- [Qwen2.5](qwen2)
- [Qwen3](qwen3)
- [Qwen3 MoE](qwen3_moe)

### Diffusion Models

- [Stable Diffusion 1.5, 2.1, 3.5 Medium](stable-diffusion/)
- [FLUX.2](flux2)

### Vision-Language Models (VLMs)

- [Qwen3-VL](vlm)

### Vision Models

- [CLIP](clip)
- [Depth Anything v3](depth-anything)
- [EDSR](edsr)
- [EfficientSAM](efficient-sam)
- [PVT v2](pvt)
- [SAM 3](sam3)
- [YOLOS](yolo)

### Audio Models

- [CLAP](clap)
- [Wav2Vec 2.0](wav2vec2)
- [Whisper](whisper)

### Text Models

- [RoBERTa](roberta)
- [T5](t5)

## Adding a Model

To make a new model exportable via short-name, add a `ModelPreset(...)` entry to `LLM_PRESETS` or `DIFFUSION_PRESETS` in [`python/src/coreai_models/model_registry.py`](../python/src/coreai_models/model_registry.py). Set the short name, HuggingFace ID, family, variant, and the export defaults (compression, compute precision, max context length).

For models with bespoke export logic that doesn't fit the standard `coreai.llm.export` / `coreai.diffusion.export` flow, write a standalone recipe under `models/<name>/export.py` — see existing recipes for the [PEP 723](https://peps.python.org/pep-0723/) pattern and `models/README.md` for the contribution checklist.

- `export.py` — Standalone conversion script with [PEP 723](https://peps.python.org/pep-0723/) inline dependencies.
- `README.md` — Model introduction, export recipe and example Swift code to make app integration easier.

For models that fit the standard `coreai.llm.export` or `coreai.diffusion.export` pipeline, add a `ModelPreset` entry to [`model_registry.py`](../python/src/coreai_models/model_registry.py) instead.

## Compiling models

Models can optionally be ahead-of-time compiled. Run `xcrun coreai-build compile --help` for usage. If you compile a model, replace the corresponding asset in the bundle directory and update `metadata.json` to reference the new filename.
