# Gemma 3

Google's Gemma 3 models for on-device inference via Core AI.

## Supported Models

| Model               | Parameters | macOS | iOS           |
| ------------------- | ---------- | ----- | ------------- |
| Gemma 3 4B Instruct | 4.0B       | Yes   | No            |
| Gemma 3 12B Instruct| 12.0B      | Yes   | No            |

## Gated Access
These Gemma 3 models are gated on [Hugging Face](https://huggingface.co/google/gemma-3-4b-it) (HF). You will need to accept the terms of the [license](https://huggingface.co/google/gemma-3-4b-it), generate a HF token, and add your HF token to your machine before exporting this model.
```bash
brew install hf
hf auth login --token <YOUR_TOKEN_HERE>
```

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```
## Export models

```bash
# Defaults to macOS variant
uv run coreai.llm.export google/gemma-3-4b-it
```

> **Note:** Gemma 3 requires `--compute-precision bfloat16` for correct results. This is already configured in the registry preset.

**Options:**

```bash
# Full precision
uv run coreai.llm.export google/gemma-3-4b-it --compression none --compute-precision bfloat16

# Custom output directory
uv run coreai.llm.export google/gemma-3-4b-it --output-dir ./my-models/

# Preview resolved config without exporting
uv run coreai.llm.export google/gemma-3-4b-it --dry-run
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

| Model       | Compression               | Bits Per Weight (BPW) | Platform | Perplexity Score |
| ----------- | ------------------------- | --------------------- | -------- | ---------------- |
| Gemma 3 4B  | none (`float16`)          | 16.00                 | macOS    | 17.90            |
| Gemma 3 4B  | [4-bit quantized][p-4bit] | 4.50                  | macOS    | 19.28            |
| Gemma 3 12B | none (`float16`)          | 16.00                 | macOS    | 11.24            |
| Gemma 3 12B | [4-bit quantized][p-4bit] | 4.50                  | macOS    | 11.75            |

[p-4bit]: ../README.md#quantization-options
