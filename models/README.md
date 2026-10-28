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

#### Downloading from ModelScope

By default models are downloaded from the HuggingFace Hub. You can switch the
download backend to [ModelScope](https://modelscope.cn) (useful where HF access
is slow or blocked, and for Qwen models which are mirrored on ModelScope):

```bash
# 1. Install the optional ModelScope backend (into the uv workspace env)
uv pip install modelscope

# 2a. Per-command: pass --backend modelscope (Qwen/Qwen3-0.6B exists
#     under the same name on both hubs, so the HF id works as-is)
uv run coreai.llm.export Qwen/Qwen3-0.6B --backend modelscope

# 2b. Or via environment variable
export COREAI_DOWNLOAD_BACKEND=modelscope
uv run coreai.llm.export Qwen/Qwen3-0.6B
```

Backend resolution precedence: `--backend` flag > `COREAI_DOWNLOAD_BACKEND`
(`huggingface` | `modelscope`) > HuggingFace (default). The `modelscope`
package is only imported when the ModelScope backend is active, so the
default path needs no extra dependency. Version 1.37 or newer is required —
older releases lack the `allow_patterns` support the download path relies on.

**Model ids belong to the selected hub's namespace.** Use a HuggingFace id
(`org/name`) with the default backend, or a ModelScope id together with the
ModelScope backend. The example above works both ways: `Qwen/Qwen3-0.6B`
exists under the same name on both hubs, so either id downloads fine with
the ModelScope backend selected. If the model lives under a *different*
namespace on ModelScope (see the table below), pass the ModelScope id
explicitly — passing the HF id to the ModelScope backend will 404 for those
repos.

> **Python API users:** The `--backend` flag only exists on the CLI. If you
> call `export_model(...)` directly from Python, set the backend via the
> environment variable instead — `export COREAI_DOWNLOAD_BACKEND=modelscope`
> (or `os.environ["COREAI_DOWNLOAD_BACKEND"]="modelscope"` before the call).

> **Standalone-script users:** `uv run models/<name>/export.py` runs in its own
> PEP 723 environment, which does not include `modelscope` — add it with
> `--with`:
>
> ```bash
> COREAI_DOWNLOAD_BACKEND=modelscope \
>   uv run --with modelscope models/whisper/export.py --model AI-ModelScope/whisper-large-v3
> ```

> **Scope note:** ModelScope download is supported across all export paths:
> LLM (`coreai.llm.export`), VLM (`coreai.vlm.export`), Diffusion
> (`coreai.diffusion.export`), Segmentation (`coreai.segmentation.export`),
> Muse-Glimmer, and standalone
> `models/<name>/export.py` scripts that load weights from a model hub
> (CLIP, CLAP, Whisper, T5, RoBERTa, YOLOS, Parakeet, EfficientSAM,
> Depth-Anything). Three scripts — Wav2Vec2, EDSR, PVT — let their
> respective libraries fetch weights themselves, outside this repo's
> download abstraction (`torchaudio` from download.pytorch.org, `torchSR`
> from GitHub Releases, `timm` from the HuggingFace Hub), so the backend
> switch does not apply to them. Note in particular that PVT downloads via
> HF regardless of the backend setting.
>
> Some models live under a different namespace on ModelScope than on
> HuggingFace (e.g. `openai/whisper-large-v3` exists only as
> `AI-ModelScope/whisper-large-v3` there). When using the ModelScope
> backend, pass the ModelScope id from the table below (standalone-script
> `--model` choices include them). Most models share the same `org/name` on
> both hubs, so the HF id works as-is with `--backend modelscope`.

> **ModelScope availability (verified):** The table below lists the
> ModelScope id to use per model (LLM registry presets + standalone scripts).
>
> | Model | HF ID | ModelScope id / status |
> |-------|-------|------------------------|
> | Whisper large-v3 / v3-turbo | `openai/whisper-*` | ✅ `AI-ModelScope/whisper-*` |
> | T5 small / base | `google-t5/t5-small` / `t5-base` | ✅ `AI-ModelScope/t5-small` / `t5-base` |
> | RoBERTa | `roberta-base` | ✅ `AI-ModelScope/roberta-base` |
> | CLIP | `openai/clip-vit-base-patch32` | ✅ `openai-mirror/clip-vit-base-patch32` |
> | GPT-OSS | `openai/gpt-oss-20b` | ⚠️ `openai-mirror/gpt-oss-20b` (see preset caveat below) |
> | Parakeet | `nvidia/parakeet-tdt-0.6b-v3` | ✅ `nv-community/parakeet-tdt-0.6b-v3` |
> | Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct` | ✅ same name |
> | YOLOS | `hustvl/yolos-*` | ✅ same name |
> | CLAP | `laion/clap-htsat-unfused` | ✅ same name |
> | Depth-Anything | `depth-anything/da3-small` | ✅ same name |
> | EfficientSAM | `merve/EfficientSAM` | ✅ same name (URL direct link) |
> | SAM3 (Segmentation / Video) | `facebook/sam3` | ✅ same name |
> | Qwen3 / Gemma3 / Phi / Mistral / FLUX.2 / SD3.5 | various | ✅ same name |
> | Qwen3.5 (0.8B / 2B / 4B / 9B / 27B) | `Qwen/Qwen3.5-*` | ✅ same name |
> | Gemma 3n (E2B / E4B) | `google/gemma-3n-*` | ✅ same name |
> | SmolLM2 (1.7B / 360M / 135M) | `HuggingFaceTB/SmolLM2-*-Instruct` | ✅ same name |
> | Wan 2.1 T2V | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | ✅ same name |
> | Sana Sprint 0.6B | `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` | ✅ same name |
> | DiffusionGemma 26B A4B | `google/diffusiongemma-26b-a4b-it` | ✅ same name |
> | OLMo 2 (1B) | `allenai/OLMo-2-0425-1B-Instruct` | ✅ same name |
> | Muse Glimmer 30B / drafter / VLM | `meta-models/Muse-Glimmer-30B` / `-assistant` / `-vision` | ✅ same name |
> | T5-large | `google-t5/t5-large` | ❌ not on ModelScope |
>
> Only T5-large is not available on ModelScope. It will fail with a 404 when
> `--backend modelscope` is used; fall back to the default HuggingFace backend
> for that model. (Note: `google/flan-t5-large` — the instruction-tuned variant
> — does exist on ModelScope, but it is a different model with different
> weights; `google-t5/t5-large` cannot be substituted with it.)

> **GPT-OSS preset caveat:** `gpt-oss-20b` is the only preset whose ModelScope
> repo lives under a different namespace. Its short name resolves to the HF id
> (`openai/gpt-oss-20b`), which 404s on ModelScope — export it from ModelScope
> by passing the ModelScope id with `--experimental` and this preset's defaults:
>
> ```bash
> uv run coreai.llm.export openai-mirror/gpt-oss-20b --backend modelscope \
>     --experimental --compute-precision bfloat16 \
>     --compression none --max-context-length 32768
> ```
>
> `--compression none` matters: without it the experimental path defaults to
> 4bit, but gpt-oss ships natively MXFP4 weights. All other presets are
> same-name on ModelScope and work as-is.

To try exporting a model that has Python source but no registry preset, use `--experimental`:

```bash
uv run coreai.llm.export org/NewModel \
    --experimental \
    --compute-precision float16 \
    --compression 4bit \
    --max-context-length 4096
```

#### Quantization Options

| Platform      | Preset                                     | Description                                          |
|---------------|--------------------------------------------|------------------------------------------------------|
| macOS         | `4bit` (default)                           | INT4 weight-only, block size 32 (all layers)         |
| macOS         | `4bit_weights_8bit_kv_cache`               | INT4 weight-only with INT8 per-tensor KV cache       |
| macOS (27.2+) | `fp4_weights_fp8_activations_fp8_kv_cache` | FP4 (e2m1) weights, FP8 (e4m3) activations, KV cache |
| macOS (27.2+) | `fp8_weights_fp8_activations_fp8_kv_cache` | FP8 (e4m3) weights, activations, and KV cache        |
| macOS         | `none`                                     | Full precision                                       |
| iOS           | `4bit_weight_palettized_group32` (default) | 4-bit palettization with channel group size 32       |
| iOS           | `4bit_weight_palettized_group8`            | 4-bit palettization with channel group size 8        |
| iOS           | `none`                                     | Full precision                                       |

**Note:** All `iOS` palettization presets quantize the Embedding to 8-bit per tensor by default.

**Note:** The `fp4_weights_fp8_activations_fp8_kv_cache` and `fp8_weights_fp8_activations_fp8_kv_cache` presets quantize weights to per-block FP4 (e2m1) and per-tensor FP8 (e4m3) respectively, with activations and KV Cache per-tensor quantized to FP8 (e4m3). Both presets require macOS 27.2+.

Override the default with `--compression`:

```bash
uv run coreai.llm.export Qwen/Qwen3-0.6B --compression none                        # full precision
uv run coreai.llm.export Qwen/Qwen3-0.6B --platform iOS --compression 4bit_weight_palettized_group8
```

**Note:** By default, all quantization presets (except the ones for KV Cache, please see below) use `coreai-opt`'s `eager` execution mode. Use the `--quantization-mode graph` argument to override and use graph-mode quantization.


##### KV Cache Quantization

KV Cache quantization can be specified for `macOS` models using `coreai-opt`'s `kv_cache_quant_configs` option in the config as follows:

```py
"kv_cache_quant_configs": {
    "mutable_cache_update_and_fetch": {
        "op_quantizer_config": {
            "op_input_spec": {
                1: {
                    "dtype": "int8",
                    "qscheme": "symmetric",
                    "granularity": {"type": "per_tensor"},
                }
            },
            "op_output_spec": None,
            "op_state_spec": None,
        },
    }
}
```

For more details, please see the `4bit_weights_8bit_kv_cache` preset in [presets.py](../python/src/coreai_models/export/presets.py). Note that KV Cache quantization requires `coreai-opt`'s graph execution mode.

Please see the [Qwen2.5](qwen2/README.md) and [Qwen3](qwen3/README.md) model cards for examples of models exported with KV Cache quantization.


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

#### Debug Information

Exports default to the converter's `RELEASE` mode, which embeds minimum debug information in the exported `.aimodel`. This keeps assets as small as possible and is what you want for anything you ship.

Pass `--include-debug-info` to switch the converter to `DEBUG` mode, which embeds full debug information in the exported `.aimodel`. That's worth doing when you're diagnosing a conversion — wrong numerics, an op that fails to lower, or a graph you need to map back to Python source:

```bash
uv run coreai.llm.export Qwen/Qwen3-0.6B --include-debug-info
```

The flag is available on `coreai.llm.export`, `coreai.vlm.export`, `coreai.diffusion.export`, `coreai.segmentation.export`, and every standalone `models/<name>/export.py` recipe, so all export paths produce assets carrying the same debug information by default.

**Note:** `--include-debug-info` is independent of `--verbose`/`-v`. `--verbose` only raises the console log level; it does not change what goes into the asset.

You don't need to re-export to shed debug information from an asset you already converted. Load it, strip the debug information in place, and save it back out:

```python
from pathlib import Path

from coreai.authoring import AIModelAsset
from coreai_torch.debugging.debug_info import strip_debug_info

source = AIModelAsset.load("inputModel.aimodel")

# `save_asset` writes only what the program carries, so capture the curated
# metadata first — it lives on the asset, not on the program.
metadata = source.metadata
author, license_, description = metadata.author, metadata.license, metadata.model_description

program = source.program
strip_debug_info(program)  # modifies the program in place
program.save_asset(Path("outputModel.aimodel"))  # save_asset requires a Path

# Re-attach it, otherwise `author`, `license` and `description` are lost.
AIModelAsset.load("outputModel.aimodel").update_metadata(
    lambda m: (
        setattr(m, "author", author),
        setattr(m, "license", license_),
        setattr(m, "model_description", description),
    )
)
```

`outputModel.aimodel` then carries the same minimum debug information a default `RELEASE` export would produce.

**Note:** the re-attach step is not optional bookkeeping. `program.save_asset()` persists only `creationDate`, `assetVersion` and `producer`; without it the round-trip silently drops the model's `author`, `license` and `description`, so a shipped asset would lose its attribution and license. The metadata attribute is `model_description`, even though the key serialized into `metadata.json` is `description`. `creationDate` is always reset to the time of the save.

### Diffusion Models

```bash
uv run coreai.diffusion.export black-forest-labs/FLUX.2-klein-4B
uv run coreai.diffusion.export sana-sprint-0.6b
```

### Vision-Language Models (VLMs)

```bash
uv run coreai.vlm.export --list-models   # list supported VLMs
uv run coreai.vlm.export qwen3-vl        # text decoder + token embedding + vision encoder
```

This produces a single `<name>/` bundle (`kind=vlm`) holding the text
decoder (`main`), token-embedding lookup (`embedding`), vision encoder
(`vision`), tokenizer, and `metadata.json`. Pass `--skip-vision` to export the
text portion only.

### Standalone Export Scripts

Models with a standalone `export.py` are run directly:

```bash
uv run models/<name>/export.py
uv run models/<name>/export.py --include-debug-info   # embed debug information in exported .aimodel
```

## Model Catalog

### Language Models (LLMs)

- [Gemma 3](gemma3)
- [Gemma 4](gemma4)
- [GPT-OSS](gpt_oss)
- [Mistral](mistral)
- [Mixtral](mixtral)
- [Muse Glimmer](muse_glimmer)
- [Phi](phi)
- [Qwen2.5](qwen2)
- [Qwen3](qwen3)
- [Qwen3 MoE](qwen3_moe)
- [Qwen3.5](qwen3_5)

### Diffusion Models

- [FLUX.2](flux2)
- [Sana Sprint](sana)

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
- [Parakeet TDT](parakeet)
- [Wav2Vec 2.0](wav2vec2)
- [Whisper](whisper)

### Text Models

- [RoBERTa](roberta)
- [T5](t5)

## Adding a Model

To make a new model exportable via short-name, add a `ModelPreset(...)` entry to `LLM_PRESETS` or `DIFFUSION_PRESETS` in [`python/src/coreai_models/model_registry.py`](../python/src/coreai_models/model_registry.py). Set the short name, HuggingFace ID, family, variant, and the export defaults (compression, compute precision, max context length).

For models with bespoke export logic that doesn't fit the standard `coreai.llm.export` / `coreai.diffusion.export` flow, write a standalone recipe under `models/<name>/export.py` — see existing recipes for the [PEP 723](https://peps.python.org/pep-0723/) pattern and `models/README.md` for the contribution checklist.

- `export.py` — Standalone conversion script with [PEP 723](https://peps.python.org/pep-0723/) inline dependencies. It must accept a `--include-debug-info` flag and construct its `TorchConverter` with `TorchConverter.Mode.RELEASE` by default, so that every export path in the repo produces assets with the same debug information. See [Debug Information](#debug-information).
- `README.md` — Model introduction, export recipe and example Swift code to make app integration easier.

For models that fit the standard `coreai.llm.export` or `coreai.diffusion.export` pipeline, add a `ModelPreset` entry to [`model_registry.py`](../python/src/coreai_models/model_registry.py) instead.

### Declaring State Kinds

The Swift runtime infers each language-model state's kind from its shape and name: a state with a dynamic dimension is a KV cache (`kv_cache`); a fixed-shape state whose name contains `sliding` is a sliding-window ring cache (`sliding_cache`); any other fixed-shape state whose name contains `cache` or `kv` is a KV cache with a fixed capacity, such as one exported at its full context; and any other state is `fixed`, such as a convolution or recurrent state. A `fixed` state cannot be partially rewound with `reset(to:)`. When the inference is wrong, for example for a ring cache whose name doesn't contain `sliding`, set the kinds in the `language` section of the bundle's `metadata.json`:

```json
"language": {
  "states": {
    "keyCache": "kv_cache",
    "valueCache": "kv_cache",
    "convStates": "fixed",
    "recurrentStates": "fixed"
  }
}
```

States the map leaves out keep their inferred kind. Only `kv_cache` states may have a dynamic dimension; a bundle that gives a dynamic-shaped state another kind fails to load.

## Compiling models

Models can optionally be ahead-of-time compiled. Run `xcrun coreai-build compile --help` for usage. If you compile a model, replace the corresponding asset in the bundle directory and update `metadata.json` to reference the new filename.
