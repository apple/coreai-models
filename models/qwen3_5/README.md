# Qwen3.5, 3.6, and 3.8

Alibaba's Qwen3.5 models for on-device inference via Core AI.

A hybrid architecture: `config.layer_types` marks each decoder layer as either
`full_attention` (gated attention over a growing KV cache) or `linear_attention`
(a gated delta net over fixed-size conv and recurrent state). The exported graph
therefore carries four states — `keyCache`, `valueCache`, `convStates`,
`recurrentStates` — rather than the usual two.

The Qwen3.6 and Qwen3.8 variants share this implementation and export the same way.

The export also emits the optional second `prefill` entrypoint beside `main`: same
inputs and states, but no LM head and no outputs, so it only fills the four states.
The runner uses it for the prompt when present and holds the last token back for
`main`, which is what produces the logits that seed sampling.

## Supported Models

| Model             | Parameters        | macOS        | iOS |
| ----------------- | ----------------- | ------------ | --- |
| Qwen3.5 0.8B      | 0.8B              | Yes (27.2+)  | No  |
| Qwen3.5 2B        | 2B                | Yes (27.2+)  | Yes |
| Qwen3.5 4B        | 4B                | Yes (27.2+)  | Yes |
| Qwen3.5 9B        | 9B                | Yes (27.2+)  | No  |
| Qwen3.5 27B       | 27B               | Yes (27.2+)  | No  |
| Qwen3.6 27B       | 27B               | Yes (27.2+)  | No  |
| Qwen3.8 27B       | 27B               | Yes (27.2+)  | No  |
| Qwen3.5 35B-A3B   | 35B (3B active)   | Yes (27.2+)  | No  |
| Qwen3.6 35B-A3B   | 35B (3B active)   | Yes (27.2+)  | No  |
| Qwen3.5 122B-A10B | 122B (10B active) | Yes (27.2+)  | No  |

The MoE checkpoints (`-A3B`, `-A10B`) use the same model class as the dense ones.

On iOS the 2B and 4B share one architecture and one export path, differing only in size.
See [Neural Engine (iOS)](#neural-engine-ios) for what that path does differently. The 0.8B
also exports but is not a verified target (no compression recipe).

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```

## Export models

```bash
# Defaults to macOS, 4-bit weights via qwen3_5_cfg.yaml, bfloat16 compute
uv run coreai.llm.export Qwen/Qwen3.5-0.8B
```

**Options:**

```bash
# Full precision instead of the default 4-bit recipe
uv run coreai.llm.export Qwen/Qwen3.5-0.8B --compression none

# Custom output directory
uv run coreai.llm.export Qwen/Qwen3.5-0.8B --output-dir ./my-models/

# Preview resolved config without exporting
uv run coreai.llm.export Qwen/Qwen3.5-0.8B --dry-run
```

## Run a Core AI Language Model

### In your macOS applications via Foundation Models

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

## Tests

```bash
uv run pytest python/tests/test_model_units/test_models/test_macos_layers/test_qwen3_5.py -v
uv run pytest python/tests/test_model_units/test_models/test_ios_layers/test_qwen3_5.py -v
```

## Neural Engine (iOS)

The 2B and 4B export to iOS through a standalone recipe, `models/qwen3_5/export.py`. The hybrid
decoder carries two mutable states beyond the KV cache and its graphs bake their shapes at trace
time, neither of which the generic `coreai.llm.export` pipeline models.

### Architecture

|                          | 2B                              | 4B                              |
| ------------------------ | ------------------------------- | ------------------------------- |
| Decoder layers           | 24 — 18 DeltaNet, 6 attention   | 32 — 24 DeltaNet, 8 attention   |
| Hidden size              | 2048                            | 2560                            |
| MLP                      | SwiGLU, `intermediate_size` 6144 | SwiGLU, `intermediate_size` 9216 |
| Attention heads (Q / KV) | 8 / 2                           | 16 / 4                          |
| Attention `head_dim`     | 256, partial RoPE (25%, θ = 1e7) | 256, partial RoPE (25%, θ = 1e7) |
| DeltaNet heads (K / V)   | 16 / 16, `head_dim` 128         | 16 / 32, `head_dim` 128         |
| DeltaNet conv            | 4 taps, depthwise, no bias      | 4 taps, depthwise, no bias      |
| Vocab                    | 248320, tied input/output embedding | 248320, tied input/output embedding |
| Norm                     | RMSNorm, `eps = 1e-6`, pre-norm around operator and MLP | same |

Three DeltaNet layers per attention layer. A DeltaNet layer keeps a fixed-size recurrent state
instead of a KV cache, so only the attention layers' state grows with context.

Two authoring choices, both documented in code:

- The depthwise conv is written as a shift-multiply-accumulate over raw per-tap parameters
  (`GatedDeltaNet._depthwise_conv`), not as a grouped conv, so it runs on the Neural Engine.
  Holding the taps outside a `Conv`/`Linear` module also keeps them out of the palettization
  recipe.
- The conv's left context lives in a ring buffer indexed by absolute position mod `W = 128`,
  written as a partial slice at a runtime offset and read through a one-hot selector. See
  `primitives/ios/ssm_cache.py`.

### Supported context

The ladder ships `{256, 1024, 8192, 32768}`. 32768 is the export's ceiling; the checkpoint's
native context is 262144. `position_ids` is a `uint16` graph input, so 32768 is the largest
bucket under its 65535 ceiling. Lifting it needs precomputed `rope_cos`/`rope_sin` inputs in
place of `position_ids`, as Gemma 4 uses.

The runner picks the smallest bucket past the current position, so over-provisioning costs only
the attention layers' KV, while every extra bucket costs a whole program to compile and load.

### Quantization and memory

4-bit k-means weight palettization, with a different group size per checkpoint:

| Checkpoint | Recipe                            | Group size |
| ---------- | --------------------------------- | ---------- |
| 2B         | `qwen3_5_2b_4bit_palettized.yaml` | 8          |
| 4B         | `qwen3_5_4b_4bit_palettized.yaml` | 32         |

The tied embedding table is int8.

| Component                    | 2B                | 4B                | Notes                                        |
| ---------------------------- | ----------------- | ----------------- | -------------------------------------------- |
| Embedding table (int8, tied) | ≈ 509 MB          | ≈ 636 MB          | 248320 × hidden                              |
| Attention KV cache (fp16)    | ≈ 402 MB @ 32768  | ≈ 1074 MB @ 32768 | scales with the context bucket                |
| Conv ring state (fp16)       | ≈ 28.3 MB         | ≈ 50.3 MB         | 128 ring columns; context-independent         |
| Recurrent state (fp16)       | ≈ 9.4 MB          | ≈ 25.2 MB         | `dk × dv` per V-head; context-independent     |
| Conv taps (fp16)             | ≈ 885 KB          | ≈ 1.5 MB          | 4 taps per channel; not palettized            |

### Export models

Run it from this directory:

```bash
cd models/qwen3_5
uv run export.py --model Qwen/Qwen3.5-2B --max-context-length 32768
uv run export.py --model Qwen/Qwen3.5-4B --max-context-length 32768

# No palettization. The tied embedding table is still 8-bit quantized.
uv run export.py --model Qwen/Qwen3.5-2B --compression none

# Your own compression recipe
uv run export.py --model Qwen/Qwen3.5-2B --compression-config my_recipe.yaml

# Cap the context to reduce export time and program size
uv run export.py --model Qwen/Qwen3.5-2B --max-context-length 8192

# Truncate to N layers (a dev-loop knob; the result is not the checkpoint)
uv run export.py --model Qwen/Qwen3.5-2B --num-layers 4
```

Each checkpoint gets its own palettization recipe by default; `--compression-config` overrides
it, and a checkpoint with no recipe of its own requires one.

`--dev-extend-qlens`, `--dev-prompt-qlens` and `--dev-ladder-only` narrow what the ladder
emits so one rung can be debugged in minutes. A shipping export passes none of them, and any
that is set is logged as a `DEV OVERRIDE` warning.

#### What the export produces

| Function                | Role                                        |
| ----------------------- | ------------------------------------------- |
| `load_embeddings`       | loads the int8 embedding table               |
| `gather_embeddings_{q}` | token ids → embeddings, at each query length |
| `extend_{ctx}_8`        | decode, one new token per step               |
| `prompt_opt_{ctx}_64`   | prefill, a full 64-token chunk               |

One decode and one prefill program per bucket. Each is traced fully static
(`dynamic_shapes=None`), because `BlockedSDPA` unrolls the key axis and the delta rule unrolls
the query block.

### Runner side

`InputHandler+DeltaNet.swift` fills the three extra graph inputs. `conv_step` and
`conv_seed_selector` address the ring; `ssm_update_flag` marks which slots hold a real,
not-yet-processed token.

That last one matters because a block's start is aligned down to a multiple of the query
length, so a decode call re-sends tokens the model has already seen. The KV cache is
position-addressed and idempotent under that; a recurrent state is not, and re-applying a
token's update corrupts it. `metadata.json` pins `language.states` so the engine classifies the
two DeltaNet states as `fixed`; they cannot be truncated, since a partial reset would leave a
recurrent state matching no prefix. The engine reports this as `rollbackCapability == .startOnly`:
any resume that would rewind restarts at 0 and replays the prompt, `reset(to:)` accepts only 0,
and a bundle whose graph has DeltaNet inputs but no `fixed` states is rejected at load.

The checkpoint's multi-token-prediction (MTP) head is not exported.

### Decoding parameters

The checkpoint ships no `generation_config.json`. The model card's non-thinking text recipe
is `temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0`. The
`0.7 / 0.8 / 20` figures on the card are its VL recipe. `llm-runner` takes all of these,
including `--presence-penalty`.

The 2B's chat template is non-thinking by default. The 4B's default template opens a `<think>`
block; for direct answers, render the prompt with `enable_thinking=False` and pass it with
`--apply-chat-template=false`:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

## Evaluation

Perplexity on [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) and
accuracy on tinyMMLU, computed with the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on the Core AI PyTorch
models (context 16384). HuggingFace rows are the stock checkpoint in `bfloat16`.

| Model        | Compression                                   | Platform    | Perplexity Score | tinyMMLU |
| ------------ | --------------------------------------------- | ----------- | ---------------- | -------- |
| Qwen3.5 2B   | HuggingFace (`bfloat16`)                      | —           | 14.23            | 0.558    |
| Qwen3.5 2B   | none (`float16`)                              | iOS         | 14.26            | 0.558    |
| Qwen3.5 2B   | [4-bit palettized][qwen3-5-2b-yaml]           | iOS         | 16.38            | 0.588    |
| Qwen3.5 4B   | HuggingFace (`bfloat16`)                      | —           | 10.96            | 0.638    |
| Qwen3.5 4B   | none (`float16`)                              | iOS         | 10.96            | 0.629    |
| Qwen3.5 4B   | [4-bit palettized][qwen3-5-4b-yaml]           | iOS         | 12.55            | 0.611    |

[qwen3-5-2b-yaml]: qwen3_5_2b_4bit_palettized.yaml
[qwen3-5-4b-yaml]: qwen3_5_4b_4bit_palettized.yaml

Measured on iPhone with the shipping bundles, greedy decoding, non-thinking prompts.

**Quality.** Needle: three facts retrieved from a haystack. Summarization: one numbered bullet
per section of a report.

| Check (prompt tokens)              | 2B                          | 4B          |
| ---------------------------------- | --------------------------- | ----------- |
| Needle (332 / 922 / 6124 / 22626)  | 12/12                       | 12/12       |
| Summarization, 2 sections (614)    | 2/2                         | 2/2         |
| Summarization, 6 sections (1515)   | 4/6, then empty bullets     | 5/6         |
| Summarization, 6 sections (26887)  | 2/6, then repeats a bullet  | 6/6         |

**Throughput**, 64 generated tokens with `--ignore-eos`, tokens/sec:

| Context bucket              | 256   | 1024   | 8192   | 32768 |
| --------------------------- | ----- | ------ | ------ | ----- |
| Prompt tokens               | 165   | 907    | 6002   | 22516 |
| 2B prefill                  | 805.8 | 1791.4 | 1431.6 | 770.0 |
| 2B decode                   | 48.4  | 41.0   | 34.9   | 18.5  |
| 4B prefill                  | 270.0 | 630.7  | 422.9  | 252.4 |
| 4B decode                   | 16.2  | 12.0   | 11.5   | 7.7   |

**Bundle size and first-load compile:** 2B 1.2 GB / 1326 s; 4B 2.3 GB / 2610 s. The compile
result is cached, so this is a one-time cost per device.
