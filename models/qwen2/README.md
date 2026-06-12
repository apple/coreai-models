# Qwen2.5

Alibaba's Qwen2.5 models for on-device inference via Core AI.

## Supported Models

| Model                 | Parameters | macOS | iOS |
| --------------------- | ---------- | ----- | --- |
| Qwen2.5 1.5B Instruct | 1.5B       | Yes   | Yes |

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```
## Export models

```bash
# Defaults to macOS variant
uv run coreai.llm.export Qwen/Qwen2.5-1.5B-Instruct
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export Qwen/Qwen2.5-1.5B-Instruct --compression none

# iOS variant
uv run coreai.llm.export Qwen/Qwen2.5-1.5B-Instruct --platform iOS

# Custom output directory
uv run coreai.llm.export Qwen/Qwen2.5-1.5B-Instruct --output-dir ./my-models/

# Truncate to N layers (for debugging)
uv run coreai.llm.export Qwen/Qwen2.5-1.5B-Instruct --num-layers 1 --compression none

# Preview resolved config without exporting
uv run coreai.llm.export Qwen/Qwen2.5-1.5B-Instruct --dry-run
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

| Model                 | Compression                               | Bits Per Weight (BPW) | Platform | Perplexity Score |
| --------------------- | ----------------------------------------- | --------------------- | -------- | ---------------- |
| Qwen2.5 1.5B Instruct | none (`float16`)                          | 16.00                 | macOS    | 12.21            |
| Qwen2.5 1.5B Instruct | [4-bit quantized][p-4bit]                 | 4.50                  | macOS    | 14.79            |
| Qwen2.5 1.5B Instruct | none (`float16`)                          | 16.00                 | iOS      | 12.21            |
| Qwen2.5 1.5B Instruct | [4-bit palettized (group size 8)][p-4bit] | 4.63\*                | iOS      | 14.64            |

\* BPW includes the Embedding which is quantized to INT8 per-tensor.

[p-4bit]: ../README.md#quantization-options
