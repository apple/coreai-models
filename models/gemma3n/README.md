# Gemma 3n

Google's Gemma 3n for on-device inference via Core AI.

## Supported Models

| Model           | Parameters          | macOS | iOS |
| --------------- | ------------------- | ----- | --- |
| gemma-3n-E2B-it | ~5B (E2B effective) | Yes   | No  |

## Export models

```bash
uv run coreai.llm.export google/gemma-3n-E2B-it
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export google/gemma-3n-E2B-it --compression none

# Custom output directory
uv run coreai.llm.export google/gemma-3n-E2B-it --output-dir ./my-models/
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

| Model           | Compression                       | BPW   | Platform | Perplexity Score |
| --------------- | --------------------------------- | ----- | -------- | ---------------- |
| gemma-3n-E2B-it | none (`bfloat16`)                 | 16.00 | macOS    | 30.63            |
| gemma-3n-E2B-it | [4-bit quantized][presets-info]   | 4.50  | macOS    | 35.57            |

Note: High wikitext perplexity is expected for instruction-tuned models. The probability
distribution is optimized for dialogue, not raw text prediction.

[presets-info]: ../README.md#quantization-options
