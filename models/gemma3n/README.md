# Gemma 3n

Google's Gemma 3n for on-device inference via Core AI.

## Supported Models

| Model           | Parameters           | macOS | iOS |
| --------------- | -------------------- | ----- | --- |
| gemma-3n-E2B-it | ~5B  (E2B effective) | Yes   | No  |
| gemma-3n-E4B-it | ~10B (E4B effective) | Yes   | No  |
| gemma-3n-E2B    | ~5B  (E2B effective) | Yes   | No  |
| gemma-3n-E4B    | ~10B (E4B effective) | Yes   | No  |

## Export models

```bash
# E2B (smaller, fits 8GB+ Macs)
uv run coreai.llm.export google/gemma-3n-E2B-it

# E4B (larger, fits 16GB+ Macs)
uv run coreai.llm.export google/gemma-3n-E4B-it
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
| gemma-3n-E2B    | none (`bfloat16`)                 | 16.00 | macOS    | 11.66            |
| gemma-3n-E2B    | [4-bit quantized][presets-info]   | 4.50  | macOS    | 15.77            |
| gemma-3n-E4B    | none (`bfloat16`)                 | 16.00 | macOS    | 9.83             |
| gemma-3n-E4B    | [4-bit quantized][presets-info]   | 4.50  | macOS    | 12.46            |

Note: Perplexity is measured on base (non-instruct) models. Instruction-tuned models have
artificially high wikitext perplexity (~30 for E2B-it) because their probability distribution
is optimized for dialogue, not raw text prediction.

[presets-info]: ../README.md#quantization-options
