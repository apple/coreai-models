# Gemma 4

Google's Gemma 4 models for on-device inference via Core AI (iOS).

## Supported Models

| Model                | Parameters | iOS |
| -------------------- | ---------- | --- |
| Gemma 4 E2B Instruct | 5.0B       | Yes (27.2+) |
| Gemma 4 E4B Instruct | 8.0B       | Yes (27.2+) |

These are the Gemma 4 checkpoints sized for a phone. They export through the
standalone [`export.py`](export.py) rather than the generic `coreai.llm.export`
pipeline, for two reasons: the blocked attention's graph depends on the context
length, so each context bucket is traced as its own static graph, and the
Per-Layer Embeddings (PLE) table is dumped as a sidecar next to the asset.

## Gated Access

These Gemma 4 models are gated on [Hugging Face](https://huggingface.co/google/gemma-4-E2B-it) (HF). Before you export, accept the [license](https://huggingface.co/google/gemma-4-E2B-it), make an HF token, and add the token to your machine.

```bash
brew install hf
hf auth login --token <YOUR_TOKEN_HERE>
```

## Setup to export models

If you don't have `uv` yet, install it:

```bash
brew install uv
```

Then, from the repo root:

```bash
uv sync
```

## Export models

[`export.py`](export.py) runs directly rather than through `coreai.llm.export`.
It runs in the workspace environment, so `uv sync` is all the setup needed. Run it
from this directory:

```bash
cd models/gemma4
uv run export.py --model google/gemma-4-E2B-it
```

> **Note:** The export defaults to a mixed-precision 4-bit palettization recipe
> shipped alongside the script,
> [`4bit_palettized.yaml`](4bit_palettized.yaml). iOS requires
> `float16` and defaults to a 131072 context, which is also its maximum. Bigger
> contexts cost export time and program size, so pass `--max-context-length` (a
> power of two) if you need less.

**Options:**

```bash
# Other variant
uv run export.py --model google/gemma-4-E4B-it

# No palettization. The embedding and PLE tables are still 8-bit quantized.
uv run export.py --model google/gemma-4-E2B-it --compression none

# Your own compression recipe
uv run export.py --model google/gemma-4-E2B-it --compression-config my_palett.yaml

# Cap the context to reduce export time and program size
uv run export.py --model google/gemma-4-E2B-it --max-context-length 8192

# Custom output directory
uv run export.py --model google/gemma-4-E2B-it --output-dir ./my-models/
```

The export follows the standard iOS flow (`coreai_models.export.ios`), except
that each transformer (context bucket, query length) pair is traced as its own
fully static program, `extend_{ctx}_{q}` / `prompt_opt_{ctx}_{q}`: the blocked
attention unrolls its block loop, so the graph depends on the context length. A
flat global KV cache is paired with a fixed-depth sliding-window ring, RoPE
arrives precomputed as `rope_cos`/`rope_sin` inputs, and the INT8 Per-Layer
Embeddings table is written as a sidecar next to the asset.

## Note on AOT Compilation

Due to a known issue with coreai-build in Xcode 27.2 beta, Ahead-of-time compilation is not supported for this model. Please use on-device specialization from .aimodel directly.

### Runner side

There is no Gemma-specific engine. The generic static-shape engine drives these
assets, wiring up handlers from what the graph declares plus the bundle
metadata: precomputed dual-RoPE rows (`language.overrides.rope`), the
sliding-window mask and ring write offset (`language.overrides.sliding_window`),
the final-logit soft cap applied before sampling
(`language.overrides.final_logit_softcapping`), and the per-layer embeddings
gather (`auxiliary_assets.per_layer_embeddings`). The global KV cache is classified as
per-bucket automatically — its backing buffer varies across the ladder — so it is
allocated at the running context bucket and re-laid-out as decode grows into
larger ones, while the fixed-depth sliding ring is allocated once.

## Run a Core AI Language Model

Exports land in `<repo-root>/exports/` unless you pass `--output-dir`.

### In your iOS applications via Foundation Models

```swift
import FoundationModels
import CoreAILanguageModels

let model = try await CoreAILanguageModel(resourcesAt: modelURL)

let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "What is quantum computing?")

print(response)
```

### On your Mac using built-in Command Line Tool

```bash
swift run -c release llm-runner --model path/to/exported_model_folder --prompt "Hello"
```

## Benchmark a Core AI Language Model

```bash
swift run -c release llm-benchmark --model path/to/exported_model_folder
```

Defaults: 512 prompt tokens, 1024 generation tokens, 5 trials. Override with `-p`, `-g`, and `-n`.

## Evaluation

| HF ID | MMLU (BF16) | MMLU (4-bit palettized, 8-bit PLE) |
| --------------------- | ----------- | ---------------------------------- |
| google/gemma-4-E2B-it | 0.5752 | 0.5240 |
| google/gemma-4-E4B-it | 0.6995 | 0.6546 |
