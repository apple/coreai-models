# OLMo 2

Allen AI's OLMo 2 models for on-device inference via Core AI. Apache 2.0 license.

## Supported Models

| Model                   | Parameters | macOS | iOS |
| ----------------------- | ---------- | ----- | --- |
| OLMo-2-0425-1B-Instruct | 1B        | Yes   | Yes |

## Export models

```bash
# macOS (INT4 quantized)
uv run coreai.llm.export allenai/OLMo-2-0425-1B-Instruct

# iOS (6-bit palettized)
uv run coreai.llm.export allenai/OLMo-2-0425-1B-Instruct --platform iOS
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export allenai/OLMo-2-0425-1B-Instruct --compression none

# iOS variant
uv run coreai.llm.export allenai/OLMo-2-0425-1B-Instruct --platform iOS
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

| Platform | Compression                          | BPW   | Perplexity | Delta vs FP16 |
| -------- | ------------------------------------ | ----- | ---------- | ------------- |
| macOS    | none (`float16`)                     | 16.00 | 17.06      | —             |
| macOS    | [4-bit quantized][presets-info]      | 4.50  | 18.72      | +10%          |
| iOS      | none (`float16`)                     | 16.00 | 17.09      | —             |
| iOS      | 8-bit palettized                     | 8.00  | 17.10      | +0.01%        |
| iOS      | [**6-bit palettized**][olmo2-6bit-yaml] | **6.00**  | **17.19**  | **+0.6%**     |
| iOS      | 4-bit palettized                     | 4.00  | 21.50      | +26%          |

[presets-info]: ../README.md#quantization-options
[olmo2-6bit-yaml]: olmo2_1b_6bit.yaml
