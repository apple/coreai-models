# CLIP

CLIP (Contrastive Language-Image Pretraining) learns joint representations of images and text, enabling zero-shot image classification with natural language labels.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run coreai.embedding.export clip-vit-b32
```

This writes the bundle `<repo-root>/exports/clip-vit-b32/`:

- `text_encoder.aimodel`: `input_ids`, `attention_mask` `[B, 77]` → `embedding` `[B, 512]`
- `image_encoder.aimodel`: `pixel_values` `[B, 3, 224, 224]` → `embedding` `[B, 512]`
- `tokenizer/`
- `metadata.json`: the tokenizer settings, image preprocessing, batch range and `logit_scale`

`B` is 1 to 64. Embeddings are L2-normalized; `logit_scale * dot(text, image)` gives Hugging Face's `CLIPModel` logits. Pass `--output-dir <path>` to write the bundle somewhere else.

```sh
uv run coreai.embedding.export clip-vit-b32 --fused           # a single fused asset, including a `main`
uv run coreai.embedding.export clip-vit-b32 --platform iOS    # bundle clip-vit-b32-ios
uv run coreai.embedding.export clip-vit-b32 --compute-precision float16
uv run coreai.embedding.export --help
```

`--fused` writes a single `model.aimodel` instead. It holds both tower functions and `main`, which runs Hugging Face's `CLIPModel` forward:

- inputs: `pixel_values`, `input_ids`, `attention_mask` (text length 77)
- outputs: `logits_per_image`, `logits_per_text`, `image_embeds`, `text_embeds`

**Options:**

| Flag                   | Description                                   | Default                |
| ---------------------- | --------------------------------------------- | ---------------------- |
| `clip-vit-b32`         | Short name, or `openai/clip-vit-base-patch32` | —                      |
| `--compute-precision`  | `float32`, `float16`, `bfloat16`              | `float32`              |
| `--platform`           | `macOS` or `iOS` (same GPU programs)          | `macOS`                |
| `--fused`              | A single fused asset, including a `main`      | —                      |
| `--text-only`          | Text tower only                               | —                      |
| `--output-dir`         | Parent of the bundle directory                | `<repo-root>/exports/` |
| `--overwrite`          | Replace an existing bundle                    | —                      |
| `--include-debug-info` | Embed debug information in the assets         | —                      |

**Supported models:**

| Model                        | Parameters |
| ---------------------------- | ---------- |
| openai/clip-vit-base-patch32 | 151M       |

## Running

### In your iOS and macOS applications

```swift
import CoreAIEmbeddings

var model = try await CoreAIEmbeddingModel(resourcesAt: URL(filePath: "coreai-models/exports/clip-vit-b32"))
let labels = ["a photo of a cat", "a photo of a dog"]
let texts = try await model.embed(texts: labels)
let image = try await model.embed(images: [cgImage])[0]
let logits = try texts.map { try Float(model.logitScale ?? 1) * image.cosine($0) }  // softmax for label probabilities
```

Texts are cut at 77 tokens, as Hugging Face's `CLIPProcessor` does with truncation; such an embedding's `status` is `.truncated(tokens:limit:)`.

[^1]: [Paper](https://arxiv.org/abs/2103.00020) · [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32)
