# Qwen3

Alibaba's Qwen3 models for on-device inference via Core AI.

## Supported Models

| Model      | Parameters | macOS | iOS |
| ---------- | ---------- | ----- | --- |
| Qwen3 0.6B | 0.6B       | Yes   | Yes |
| Qwen3 4B   | 4.0B       | Yes   | Yes |
| Qwen3 8B   | 8.0B       | Yes   | No  |

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```
## Export models

```bash
# Defaults to macOS variant
uv run coreai.llm.export Qwen/Qwen3-0.6B
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export Qwen/Qwen3-0.6B --compression none

# iOS variant
uv run coreai.llm.export Qwen/Qwen3-0.6B --platform iOS

# Custom output directory
uv run coreai.llm.export Qwen/Qwen3-0.6B --output-dir ./my-models/

# Truncate to N layers (for debugging)
uv run coreai.llm.export Qwen/Qwen3-0.6B --num-layers 1 --compression none

# Preview resolved config without exporting
uv run coreai.llm.export Qwen/Qwen3-0.6B --dry-run
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

| Model      | Compression                                          | Bits Per Weight (BPW) | Platform | Perplexity Score |
| ---------- | ---------------------------------------------------- | --------------------- | -------- | ---------------- |
| Qwen3 0.6B | none (`float16`)                                     | 16.00                 | iOS      | 26.16            |
| Qwen3 0.6B | [Mixed 4-bit/8-bit palettized][mixed-4bit-8bit-yaml] | 5.71\*                | iOS      | 30.90            |
| Qwen3 4B   | none (`float16`)                                     | 16.00                 | macOS    | 16.41            |
| Qwen3 4B   | [4-bit quantized][presets-info]                      | 4.50                  | macOS    | 18.33            |
| Qwen3 4B   | none (`float16`)                                     | 16.00                 | iOS      | 16.41            |
| Qwen3 4B   | [Mixed 4-bit/8-bit palettized][qwen3-4b-mixed-yaml]  | 4.89\*                | iOS      | 18.80            |
| Qwen3 8B   | none (`float16`)                                     | 16.00                 | macOS    | 12.19            |
| Qwen3 8B   | [4-bit quantized][presets-info]                      | 4.50                  | macOS    | 12.90            |

\* BPW includes the Embedding which is quantized to INT8 per-tensor.

[presets-info]: ../README.md#quantization-options
[mixed-4bit-8bit-yaml]: qwen3_0_6b_mixed_4bit_8bit.yaml
[qwen3-4b-mixed-yaml]: qwen3_4b_mixed_4bit_8bit.yaml
