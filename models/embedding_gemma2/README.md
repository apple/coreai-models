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

[^1]: [Documentation](https://ai.google.dev/gemma/docs/embeddinggemma) · [HuggingFace](https://huggingface.co/google/embeddinggemma-2)
