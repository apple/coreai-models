# Phi-4-mini

Microsoft's Phi-4-mini for on-device inference via Core AI.

## Supported Models

| Model      | Parameters | macOS | iOS |
| ---------- | ---------- | ----- | --- |
| Phi-4-mini-instruct | 3.8B | Yes | No |

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```
## Export models

```bash
# Defaults to macOS variant - INT4 quantized (recommended)
uv run coreai.llm.export microsoft/Phi-4-mini-instruct
```

**Options:**

```bash
# Full precision (float16, ~7.6GB)
uv run coreai.llm.export microsoft/Phi-4-mini-instruct --compression none

# INT4 quantized (~2GB, 2.8x faster generation)
uv run coreai.llm.export microsoft/Phi-4-mini-instruct --compression 4bit

# Custom output directory
uv run coreai.llm.export microsoft/Phi-4-mini-instruct --output-dir ./my-models/

# Preview resolved config without exporting
uv run coreai.llm.export microsoft/Phi-4-mini-instruct --dry-run
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

## Performance

On Apple Silicon (M-series):

| Variant | Prompt (t/s) | Generation (t/s) | Model Size |
| ------- | ------------ | ---------------- | ---------- |
| INT4    | ~253         | ~108             | ~2 GB      |
| FP16    | ~32          | ~38              | ~7.6 GB    |

## Architecture Notes

- MIT license
- 3.8B parameters, 32 layers
- GQA: 24 heads, 8 KV heads, head_dim=128
- Partial rotary embedding (75% of head_dim)
- SiLU-gated MLP
- Vocabulary: 200K tokens
