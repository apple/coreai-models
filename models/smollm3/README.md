# SmolLM3

HuggingFace's SmolLM3 for on-device inference via Core AI.

## Supported Models

| Model           | Parameters | macOS | iOS |
| --------------- | ---------- | ----- | --- |
| SmolLM3-3B      | 3B         | Yes   | Yes |
| SmolLM3-3B-Base | 3B         | Yes   | Yes |

## Export models

```bash
# 3B (INT4 quantized with FP16 embedding)
uv run coreai.llm.export HuggingFaceTB/SmolLM3-3B

# 3B Base
uv run coreai.llm.export HuggingFaceTB/SmolLM3-3B-Base
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export HuggingFaceTB/SmolLM3-3B --compression none

# iOS variant
uv run coreai.llm.export HuggingFaceTB/SmolLM3-3B --platform iOS
```

## Run a Core AI Language Model

### On your Mac using built-in Command Line Tool

```bash
swift run -c release llm-runner --model path/to/exported_model_folder --prompt "Hello"
```

## Benchmark a Core AI Language Model

```bash
swift run -c release llm-benchmark --model path/to/exported_model_folder
```

Defaults: 512 prompt tokens, 1024 generation tokens, 5 trials. Override with `-p`, `-g`, and `-n`.
