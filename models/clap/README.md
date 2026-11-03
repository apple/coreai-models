# CLAP

CLAP (Contrastive Language-Audio Pretraining) learns joint representations of audio and text, enabling zero-shot audio classification with natural language labels.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run coreai.embedding.export clap-htsat
```

This writes the bundle `<repo-root>/exports/clap-htsat/`:

- `text_encoder.aimodel`: `input_ids`, `attention_mask` `[B, 77]` → `embedding` `[B, 512]`
- `audio_encoder.aimodel`: log-mel `input_features` `[B, 1, 1001, 64]` → `embedding` `[B, 512]`
- `tokenizer/`
- `metadata.json`: the tokenizer settings, audio front end, batch range and `logit_scale`

`B` is 1 to 64. Embeddings are L2-normalized; `logit_scale * dot(text, audio)` gives Hugging Face's `logits_per_audio`. Pass `--output-dir <path>` to write the bundle somewhere else.

A clip longer than 10 s keeps its first 10 s; Hugging Face's `ClapFeatureExtractor` (`truncation="rand_trunc"`) takes a random 10 s window instead.

`logit_scale` is 18.66, `exp(logit_scale_a)`. Hugging Face's `logits_per_text` uses `logit_scale_t` instead, which stayed at its initial value (14.29) in training.

```sh
uv run coreai.embedding.export clap-htsat --fused           # a single fused asset, including a `main`
uv run coreai.embedding.export clap-htsat --platform iOS    # bundle clap-htsat-ios
uv run coreai.embedding.export clap-htsat --compute-precision float16
uv run coreai.embedding.export --help
```

`--fused` writes a single `model.aimodel` instead. It holds both tower functions and `main`, which runs Hugging Face's `ClapModel` forward:

- inputs: `input_ids`, `attention_mask` (text length 77), `input_features`, `is_longer`
- outputs: `logits_per_audio`, `logits_per_text`, `text_embeds`, `audio_embeds`

**Options:**

| Flag                   | Description                               | Default                |
| ---------------------- | ----------------------------------------- | ---------------------- |
| `clap-htsat`           | Short name, or `laion/clap-htsat-unfused` | —                      |
| `--compute-precision`  | `float32`, `float16`                      | `float32`              |
| `--platform`           | `macOS` or `iOS` (same GPU programs)      | `macOS`                |
| `--fused`              | A single fused asset, including a `main`  | —                      |
| `--text-only`          | Text tower only                           | —                      |
| `--output-dir`         | Parent of the bundle directory            | `<repo-root>/exports/` |
| `--overwrite`          | Replace an existing bundle                | —                      |
| `--include-debug-info` | Embed debug information in the assets     | —                      |

**Supported models:**

| Model                    | Parameters |
| ------------------------ | ---------- |
| laion/clap-htsat-unfused | 153M       |

[^1]: [Paper](https://arxiv.org/abs/2211.06687) · [HuggingFace](https://huggingface.co/laion/clap-htsat-unfused)
