# EfficientSAM

Lightweight, promptable image segmentation model that uses a masked autoencoder pretrained ViT-Tiny encoder to reduce compute while preserving accuracy.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run export.py
```

Saves to `<repo-root>/exports/<model>_<dtype>_<static_or_dynamic>[<query-suffix>][<pts-suffix>]/` as a bundle directory containing `<variant>.aimodel` and a `metadata.json` (segmenter bundle, schema 0.2). For example, `--num-queries 4 --num-pts 2` writes to `<repo-root>/exports/efficient_sam_vitt_float16_static_q4_p2/`. Pass `--output-dir <path>` to override the destination.

```sh
uv run export.py --help
```

**Options:**

| Flag            | Description                                                                                                       | Default                |
| --------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `--model`       | Model variant                                                                                                     | `efficient_sam_vitt`   |
| `--output-dir`  | Output directory for the bundle                                                                                   | `<repo-root>/exports/` |
| `--dtype`       | `float16`, `bfloat16`, `float32`                                                                                  | `float32`              |
| `--overwrite`   | Overwrite existing bundle                                                                                         | —                      |
| `--dynamic`     | Dynamic batch size                                                                                                | —                      |
| `--num-queries` | Number of prompt queries (`Q`). Use `1` for single-prompt, or a perfect square (e.g. `64`) for segment-everything | `1`                    |
| `--num-pts`     | Points per query (`P`). Use `1` for a click, `2` for a box (top-left + bottom-right)                              | `1`                    |

> **Note:** `--dynamic` with `--dtype float16` is not supported. The Core AI runtime cannot handle dynamic reshape in attention heads at float16. Use `--dynamic` with float32, or float16 without `--dynamic`.

**Supported models:**

| Model              | Parameters |
| ------------------ | ---------- |
| efficient_sam_vitt | 10M        |

### Common Variants

| Use case                        | Command                                         |
| ------------------------------- | ----------------------------------------------- |
| Single foreground click         | `uv run export.py` (defaults: Q=1, P=1)        |
| Box prompt                      | `uv run export.py --num-pts 2`                  |
| Box + extra click in one query  | `uv run export.py --num-pts 3`                  |
| Segment-everything (8x8 grid)   | `uv run export.py --num-queries 64`             |
| Independent multi-click prompts | `uv run export.py --num-queries 4 --num-pts 1`  |

`Q` is the number of independent masks the model emits per image; `P` is how many points fuse into the same prompt. A box requires both corners in the *same* query (P >= 2), not two separate single-point queries.

## Running

### In your iOS and macOS applications

```swift
import ImageSegmenter

// Load from a segmenter bundle directory (contains metadata.json and *.aimodel).
let segmenter = try await ImageSegmenter(resourcesAt: "coreai-models/exports/efficient_sam_vitt_float32_static")

// Box prompt — one query with two points (EfficientSAM with --num-pts 2):
let box = PointQuery(points: [
    .init(x: 100, y: 100, label: .boxTopLeft),
    .init(x: 400, y: 300, label: .boxBottomRight),
])
let boxSegments = try await segmenter.segment(image: cgImage, pointQuery: box)

// Multiple independent point prompts — Q queries, P=1 each:
let multi = PointQuery(queries: [
    [.init(x: 100, y: 100)],
    [.init(x: 300, y: 300)],
])
let multiSegments = try await segmenter.segment(image: cgImage, pointQuery: multi)

// Segment-everything — the engine substitutes a sqrt(num_queries) × sqrt(num_queries)
// foreground-point grid (e.g. an 8×8 grid for a Q=64 export):
let everything = try await segmenter.segment(image: cgImage, pointQuery: PointQuery())
```

### On your Mac using built-in Command Line Tool

```bash
swift run -c release image-segmenter --model path/to/exported_model_folder --image path/to/image.jpg --point 100,100
```

[^1]: [Paper](https://arxiv.org/abs/2312.00863) · [Repository](https://github.com/yformer/EfficientSAM)
