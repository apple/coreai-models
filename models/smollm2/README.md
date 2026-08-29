# SmolLM2

HuggingFace's SmolLM2 for on-device inference via Core AI.

## Supported Models

| Model                 | Parameters | macOS | iOS |
| --------------------- | ---------- | ----- | --- |
| SmolLM2-1.7B-Instruct | 1.7B       | Yes   | Yes |
| SmolLM2-360M-Instruct | 360M       | Yes   | Yes |
| SmolLM2-135M-Instruct | 135M       | Yes   | Yes |

## Export models

```bash
# 1.7B (INT4 quantized with FP16 embedding)
uv run coreai.llm.export HuggingFaceTB/SmolLM2-1.7B-Instruct

# 360M (FP16)
uv run coreai.llm.export HuggingFaceTB/SmolLM2-360M-Instruct

# 135M (FP16)
uv run coreai.llm.export HuggingFaceTB/SmolLM2-135M-Instruct
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export HuggingFaceTB/SmolLM2-1.7B-Instruct --compression none

# iOS variant
uv run coreai.llm.export HuggingFaceTB/SmolLM2-1.7B-Instruct --platform iOS
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

| Model | Compression                                | BPW   | Platform | Perplexity |
| ----- | ------------------------------------------ | ----- | -------- | ---------- |
| 1.7B  | none (`float16`)                           | 16.00 | macOS    | 12.27      |
| 1.7B  | [INT4 with FP16 embedding][smollm2-yaml]   | 4.56  | macOS    | 14.17      |
| 1.7B  | [6-bit palettized][smollm2-6bit-yaml]      | 6.00  | iOS      | 13.63      |
| 360M  | none (`float16`)                           | 16.00 | macOS    | 18.23      |
| 135M  | none (`float16`)                           | 16.00 | macOS    | 24.99      |

[smollm2-yaml]: smollm2_4bit_embedding_excluded.yaml
[smollm2-6bit-yaml]: smollm2_1_7b_6bit.yaml
