# SmolLM2

HuggingFace's SmolLM2 for on-device inference via Core AI.

## Supported Models

| Model      | Parameters | macOS | iOS |
| ---------- | ---------- | ----- | --- |
| SmolLM2-1.7B-Instruct | 1.7B | Yes | No |

## Export models

```bash
# Defaults to macOS variant - INT4 quantized (recommended)
uv run coreai.llm.export HuggingFaceTB/SmolLM2-1.7B-Instruct
```

**Options:**

```bash
# Full precision (float16)
uv run coreai.llm.export HuggingFaceTB/SmolLM2-1.7B-Instruct --compression none

# INT4 quantized (~1GB, faster generation)
uv run coreai.llm.export HuggingFaceTB/SmolLM2-1.7B-Instruct --compression 4bit
```

## Run

```bash
swift run -c release llm-runner --model path/to/exported_model_folder --prompt "Hello"
```

## Performance

On Apple Silicon (M-series):

| Variant | Prompt (t/s) | Generation (t/s) | Model Size |
| ------- | ------------ | ---------------- | ---------- |
| INT4    | ~502         | ~176             | ~1 GB      |

## Architecture Notes

- Apache 2.0 license
- 1.7B parameters, 24 layers
- MHA: 32 heads, 32 KV heads, head_dim=64
- Standard Llama architecture (SiLU MLP, RMSNorm, RoPE)
- Vocabulary: 49K tokens
