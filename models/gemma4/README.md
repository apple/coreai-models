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

### Image input (vision)

Pass `--include-vision` to also export a vision encoder into the bundle,
upgrading it to `kind:vlm`:

```bash
uv run export.py --model google/gemma-4-E2B-it --include-vision
```

The vision encoder is a ladder of canvas-size buckets, not one fixed
resolution: at runtime, an input image is letterboxed (aspect-preserving
resize + zero-pad, never upscaled) into the smallest bucket that fits it, and
the padding is masked out of attention. Options:

```bash
# Custom bucket ladder (px), smallest first. Each must be a multiple of
# patch_size * pooling_kernel_size (48 for E2B/E4B) so padding stays
# pool-block-aligned. Defaults to 384/768/1152.
uv run export.py --model google/gemma-4-E2B-it --include-vision \
  --vision-image-sizes 384,768,1152

# Compression for the vision tower (default matches the text decoder's
# 4-bit palettization). Use 'none' to ship the tower at fp16.
uv run export.py --model google/gemma-4-E2B-it --include-vision \
  --vision-compression none
```

### Draft head (speculative decoding)

A Gemma 4 assistant checkpoint can be exported as a draft head that proposes tokens for the
exported model. Export the model with its final hidden state as an extra decode output, then add
the draft head to the bundle:

```bash
uv run export.py --model google/gemma-4-E2B-it --emit-hidden-states
uv run export_drafter.py --model google/gemma-4-E2B-it --bundle <bundle>
```

The assistant is the target's own, `<model>-assistant` on the hub (`google/gemma-4-E2B-it-assistant`
for E2B, `google/gemma-4-E4B-it-assistant` for E4B). `export_drafter.py` downloads its safetensors
and `config.json` on first use. `--drafter` takes another hub repo id or a local directory instead.
An assistant is trained for one target, so the script refuses one whose hidden size does not match
`--model`. `--bundle` takes the bundle that `export.py` wrote with `--emit-hidden-states`; the
script refuses a bundle exported without it.

`--emit-hidden-states` makes the decode graphs return their post-norm final hidden state as a
second output, `hidden_states`, and records `language.overrides.emits_hidden_states` in the bundle
metadata. The default export is unchanged.

`export_drafter.py` exports the head's layer stack as one static-shape program, `draft_step.aimodel`,
with one function per context bucket of the target export's ladder, up to `--max-context-length`.
The token-embedding lookup and the candidate scan stay outside the program, over
`draft_scan.bin`. `--bundle` copies the files into the bundle and declares them as the `drafter`
component: the program under `assets.drafter`, and `drafter_config`, `drafter_scan` and
`drafter_scan_index` under `auxiliary_assets`. It also sets `speculative.drafter_kind` to `mtp`,
because other drafters use the same `drafter` asset key with different semantics; `--bundle`
refuses a bundle that already declares one. The head attends to the target's K/V in the most
recent 512 positions in its sliding layers and to the whole context in its full-attention layer.

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

For a bundle exported with `--include-vision`, pass `--image`:

```bash
swift run -c release llm-runner --model path/to/exported_model_folder \
  --image path/to/photo.jpg --prompt "Describe this image."
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
