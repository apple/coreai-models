# Phi Family

Microsoft's Phi-3, Phi-3.5, and Phi-4 mini models for on-device inference via Core AI.

## Supported Models

| Model                    | Parameters | Context | macOS | iOS |
| ------------------------ | ---------- | ------- | ----- | --- |
| Phi-4-mini-instruct      | 3.8B       | 131072  | Yes   | No  |
| Phi-3.5-mini-instruct    | 3.8B       | 131072  | Yes   | No  |
| Phi-3-mini-4k-instruct   | 3.8B       | 4096    | Yes   | No  |

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```

## Export models

```bash
# Phi-4-mini (recommended)
uv run coreai.llm.export microsoft/Phi-4-mini-instruct

# Phi-3.5-mini
uv run coreai.llm.export microsoft/Phi-3.5-mini-instruct

# Phi-3-mini (4K context)
uv run coreai.llm.export microsoft/Phi-3-mini-4k-instruct
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

Perplexity score on the [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) dataset computed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/README.md) with the Core AI PyTorch models. The full precision scores have been validated against HuggingFace transformers baseline (within 0.3%).

| Model            | Compression                              | Bits Per Weight (BPW) | Platform | Perplexity Score |
| ---------------- | ---------------------------------------- | --------------------- | -------- | ---------------- |
| Phi-3-mini       | none (`float16`)                         | 16.00                 | macOS    | 9.47             |
| Phi-3-mini       | [INT4 with FP16 embedding][phi-4bit-yaml]| 4.56\*                | macOS    | 11.24            |
| Phi-3.5-mini     | none (`float16`)                         | 16.00                 | macOS    | 9.98             |
| Phi-3.5-mini     | [INT4 with FP16 embedding][phi-4bit-yaml]| 4.56\*                | macOS    | 12.04            |
| Phi-4-mini       | none (`float16`)                         | 16.00                 | macOS    | 11.12            |
| Phi-4-mini       | [INT4 with FP16 embedding][phi-4bit-yaml]| 4.56\*                | macOS    | 12.80            |

\* BPW: INT4 body (4.50) + FP16 embedding. Embedding is excluded from quantization
because Phi-4 ties embedding and lm_head weights — INT4 on lm_head degrades generation quality.

[phi-4bit-yaml]: phi_4bit_embedding_excluded.yaml
