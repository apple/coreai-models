# Qwen3-Embedding

Multilingual text embedding model: each text becomes a single 1024-dimensional, L2-normalized vector, and its leading dimensions, down to 32 and re-normalized, are valid smaller embeddings. Queries carry a one-sentence task instruction; documents do not.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run coreai.embedding.export qwen3-embedding-0.6b
```

Saves to `<repo-root>/exports/qwen3-embedding-0.6b/` (text tower, GPU). The export is for macOS. Pass `--output-dir <path>` to change the parent directory.

```sh
uv run coreai.embedding.export --help
```

**Options:**

| Flag                   | Description                                         | Default                |
| ---------------------- | --------------------------------------------------- | ---------------------- |
| `--compute-precision`  | `float32`, `float16` or `bfloat16`                  | `float32`              |
| `--output-dir`         | Parent directory for the bundle                     | `<repo-root>/exports/` |
| `--overwrite`          | Overwrite an existing bundle                        | —                      |
| `--include-debug-info` | Embed conversion debug info in the `.aimodel` file  | —                      |
| `--dry-run`            | Print the export plan without writing               | —                      |
| `--list-models`        | List the supported models and exit                  | —                      |

**Supported models:**

| Model                | HuggingFace ID            | Parameters |
| -------------------- | ------------------------- | ---------- |
| qwen3-embedding-0.6b | Qwen/Qwen3-Embedding-0.6B | 596M       |

## Bundle

`metadata.json` lists `text_encoder.aimodel` under `assets`. `embedding.components.text_encoder` gives its `inputs` and `outputs` (`dtype[shape]`) and its `functions`:

| Component      | Asset (GPU)    | Functions     | Output dtype |
| -------------- | -------------- | ------------- | ------------ |
| `text_encoder` | `text_encoder` | `len{L}_b{B}` | `float32`    |

The graph takes the final hidden state of the last real token and L2-normalizes it. Read the function names from `embedding.components.text_encoder.functions`.

## Using the bundle

A host runs a bundle as follows:

- **Text:** tokenize `prompts[name] + text` with the bundled tokenizer. It adds no `<bos>` and ends every text with `<|endoftext|>` (`eos_token_id`), which is also `pad_token_id`. Right-pad to the smallest function that fits the batch and the token count, with `attention_mask` 1 on real tokens.
- **Queries:** `prompts.query` is the instruction for web search: `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`. For another task, write `Instruct: {task}\nQuery:{query}` with a one-sentence task. Documents take no prefix (`prompts.document` is empty).
- **Vectors:** to use a smaller Matryoshka dimension, keep the leading dimensions and L2-normalize again. Compare vectors with cosine similarity.

[^1]: [Blog](https://qwenlm.github.io/blog/qwen3-embedding/) · [GitHub](https://github.com/QwenLM/Qwen3-Embedding) · [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
