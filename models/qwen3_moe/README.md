# Qwen3 MoE

Alibaba's Qwen3 Mixture-of-Experts models for on-device inference via Core AI.

## Supported Models

| Model                        | Parameters (Total/Active) | macOS | iOS |
| ---------------------------- | ------------------------- | ----- | --- |
| Qwen3 Coder 30B-A3B Instruct | 30B / 3B                  | Yes   | No  |

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```bash
# Defaults to macOS variant
uv run coreai.llm.export Qwen/Qwen3-Coder-30B-A3B-Instruct --compression 4bit
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export Qwen/Qwen3-Coder-30B-A3B-Instruct --compression none

# Custom output directory
uv run coreai.llm.export Qwen/Qwen3-Coder-30B-A3B-Instruct --output-dir ./my-models/

# Truncate to N layers (for debugging)
uv run coreai.llm.export Qwen/Qwen3-Coder-30B-A3B-Instruct --num-layers 2 --compression none

# Preview resolved config without exporting
uv run coreai.llm.export Qwen/Qwen3-Coder-30B-A3B-Instruct --dry-run
```

## Benchmark a Core AI Language Model

```bash
swift run -c release llm-benchmark --model path/to/exported_model_folder
```

Defaults: 512 prompt tokens, 1024 generation tokens, 5 trials. Override with `-p`, `-g`, and `-n`.

## Evaluation

Perplexity score on the [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) dataset computed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/README.md) with the Core AI PyTorch models.

| Model                        | Compression               | Bits Per Weight (BPW) | Platform | Perplexity Score |
| ---------------------------- | ------------------------- | --------------------- | -------- | ---------------- |
| Qwen3 Coder 30B-A3B Instruct | none (`float16`)          | 16.00                 | macOS    | 11.06            |
| Qwen3 Coder 30B-A3B Instruct | [4-bit quantized][p-4bit] | 4.50                  | macOS    | 11.90            |

[p-4bit]: ../README.md#quantization-options
