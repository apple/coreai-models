# EmbeddingGemma 2

Multimodal retrieval embedding model that maps text and images into one shared space: each input becomes a single 768-dimensional, L2-normalized vector, and its leading 512, 256 or 128 dimensions are valid smaller embeddings.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run coreai.embedding.export embeddinggemma-2
```

Saves to `<repo-root>/exports/embeddinggemma-2/` (text and image towers, GPU). Pass `--output-dir <path>` to override the destination.

```sh
uv run coreai.embedding.export --help
```

**Options:**

| Flag                  | Description                                          | Default                              |
| --------------------- | ---------------------------------------------------- | ------------------------------------ |
| `--platform`          | `macOS` (GPU bundle) or `iOS` (Neural Engine bundle) | `macOS`                              |
| `--skip-vision`       | Export the text tower only                           | —                                    |
| `--compute-precision` | `float32`, `bfloat16` on macOS; `float16` on iOS     | `float32` on macOS, `float16` on iOS |
| `--output-dir`        | Parent directory for the bundle                      | `<repo-root>/exports/`               |
| `--overwrite`         | Overwrite an existing bundle                         | —                                    |
| `--dry-run`           | Print the export plan without writing                | —                                    |

**Supported models:**

| Model            | HuggingFace ID          | Parameters                   |
| ---------------- | ----------------------- | ---------------------------- |
| embeddinggemma-2 | google/embeddinggemma-2 | 439M (text 271M, image 168M) |

## Bundle

`metadata.json` lists each `.aimodel` file once under `assets`. Each entry of `embedding.components` names the asset it lives in (`asset`), its functions and its `output_dtype`:

| Component       | macOS asset (GPU)            | iOS asset                       | Functions          | Output dtype                   |
| --------------- | ---------------------------- | ------------------------------- | ------------------ | ------------------------------ |
| `text_encoder`  | `text_encoder`               | —                               | `len{L}_b{B}`      | `float32`                      |
| `token_table`   | `text_encoder` (images only) | `token_table` (GPU)             | `main`             | `float32` macOS, `float16` iOS |
| `encoder`       | `text_encoder` (images only) | `encoder` (Neural Engine)       | `embeds_len{L}_b1` | `float32` macOS, `float16` iOS |
| `image_encoder` | `image_encoder`              | `image_encoder` (Neural Engine) | `main`             | `float32` macOS, `float16` iOS |

The encoder functions are named `embeds_len{L}_b1` on both platforms; read the names from `components.encoder.functions`.

[^1]: [Documentation](https://ai.google.dev/gemma/docs/embeddinggemma) · [HuggingFace](https://huggingface.co/google/embeddinggemma-2)
