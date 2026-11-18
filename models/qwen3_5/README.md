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

| Model             | Parameters        | macOS        | iOS          |
| ----------------- | ----------------- | ------------ | ------------ |
| Qwen3.5 0.8B      | 0.8B              | Yes (27.2+)  | No           |
| Qwen3.5 2B        | 2B                | Yes (27.2+)  | Yes (27.2+)  |
| Qwen3.5 4B        | 4B                | Yes (27.2+)  | Yes (27.2+)  |
| Qwen3.5 9B        | 9B                | Yes (27.2+)  | No           |
| Qwen3.5 27B       | 27B               | Yes (27.2+)  | No           |
| Qwen3.6 27B       | 27B               | Yes (27.2+)  | No           |
| Qwen3.8 27B       | 27B               | Yes (27.2+)  | No           |
| Qwen3.5 35B-A3B   | 35B (3B active)   | Yes (27.2+)  | No           |
| Qwen3.6 35B-A3B   | 35B (3B active)   | Yes (27.2+)  | No           |
| Qwen3.5 122B-A10B | 122B (10B active) | Yes (27.2+)  | No           |

The MoE checkpoints (`-A3B`, `-A10B`) use the same model class as the dense ones. On iOS, the
2B and 4B export through `models/qwen3_5/export.py` instead of `coreai.llm.export`.

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

# iOS variant (2B and 4B), 4-bit palettized
uv run models/qwen3_5/export.py --model Qwen/Qwen3.5-2B

# iOS variant with a smaller context cap, to reduce export time and program size
uv run models/qwen3_5/export.py --model Qwen/Qwen3.5-2B --max-context-length 8192
```

## Run a Core AI Language Model

### In your iOS and macOS applications via Foundation Models

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

Perplexity score on the [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) dataset computed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/README.md) with the Core AI PyTorch models.

| Model      | Compression                               | Platform | Perplexity Score |
| ---------- | ----------------------------------------- | -------- | ---------------- |
| Qwen3.5 2B | none (`float16`)                          | iOS      | 14.26            |
| Qwen3.5 2B | [4-bit palettized][qwen3-5-2b-yaml]       | iOS      | 16.38            |
| Qwen3.5 4B | none (`float16`)                          | iOS      | 10.96            |
| Qwen3.5 4B | [4-bit palettized][qwen3-5-4b-yaml]       | iOS      | 12.55            |

[qwen3-5-2b-yaml]: qwen3_5_2b_4bit_palettized.yaml
[qwen3-5-4b-yaml]: qwen3_5_4b_4bit_palettized.yaml
