# Muse Glimmer

Meta's Muse Glimmer 30B for on-device agentic tasks via Core AI. Apache 2.0 license.

## Supported Models

| Model            | Parameters | Context | macOS | iOS |
|------------------|-----------|---------|-------|-----|
| Muse-Glimmer-30B | ~29.6B    | 131072  | Yes   | No  |

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```

## Export models

```bash
# Defaults to macOS variant
uv run coreai.llm.export muse-glimmer-30b
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export muse-glimmer-30b --compression none

# Custom output directory
uv run coreai.llm.export muse-glimmer-30b --output-dir ./my-models/

# Preview resolved config without exporting
uv run coreai.llm.export muse-glimmer-30b --dry-run
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
swift run -c release llm-runner --model path/to/exported_model --prompt "Hello"
```

## Benchmark a Core AI Language Model

```bash
swift run -c release llm-benchmark --model path/to/exported_model
```

Defaults: 512 prompt tokens, 1024 generation tokens, 5 trials. Override with `-p`, `-g`, and `-n`.

## Evaluation

Perplexity score on the [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) dataset computed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/README.md) with the Core AI PyTorch models.

| Model | Compression               | Bits Per Weight (BPW) | Platform | Perplexity Score |
|-------|---------------------------|-----------------------|----------|------------------|
| 30B   | none (`float16`)          | 16.00                 | macOS    | 7.71             |
| 30B   | [4-bit quantized][p-4bit] | 4.50                  | macOS    | 8.46             |

[p-4bit]: ../README.md#quantization-options
