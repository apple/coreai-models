# SmolLM3

HuggingFace's SmolLM3 for on-device inference via Core AI.

## Supported Models

| Model           | Parameters | macOS | iOS |
| --------------- | ---------- | ----- | --- |
| SmolLM3-3B      | 3B         | Yes   | Yes |
| SmolLM3-3B-Base | 3B         | Yes   | Yes |

## Export models

```bash
# 3B (Mixed 4-bit/8-bit quantized with FP16 embedding)
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

## Evaluation

Perplexity score on the [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) dataset computed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/README.md) with the Core AI PyTorch models.

| Model            | Compression                                      | BPW    | Platform | Perplexity |
| ---------------- | ------------------------------------------------ | ------ | -------- | ---------- |
| SmolLM3-3B-Base  | none (`float16`)                                 | 16.00  | macOS    | 10.07      |
| SmolLM3-3B-Base  | [Mixed INT4/INT8, FP16 embedding][smollm3-mixed] | 6.41\* | macOS    | 10.71      |

\* BPW includes the Embedding which is kept in FP16.

[smollm3-mixed]: smollm3_mixed_4bit_8bit.yaml

