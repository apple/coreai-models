# YOLOS

YOLOS (You Only Look at One Sequence) applies a plain Vision Transformer directly to image patches and predicts object queries as bounding boxes and class logits.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run export.py
```

Saves to `<repo-root>/exports/<model>_<dtype>_<static_or_dynamic>.aimodel` (e.g. `<repo-root>/exports/hustvl_yolos-base_float32_static.aimodel`). Pass `--output-dir <path>` to override the destination.

```sh
uv run export.py --help
```

**Options:**

| Flag           | Description                                               | Default                |
| -------------- | --------------------------------------------------------- | ---------------------- |
| `--model`      | Model variant                                             | `hustvl/yolos-base`    |
| `--output-dir` | Output directory for `.aimodel`                           | `<repo-root>/exports/` |
| `--dtype`      | `float16`, `bfloat16`, `float32`                          | `float32`              |
| `--overwrite`  | Overwrite existing `.aimodel`                             | —                      |
| `--dynamic`    | Dynamic batch (1–64), spatial (128–1024, multiples of 16) | —                      |

**Supported models:**

| Model             | Parameters |
| ----------------- | ---------- |
| hustvl/yolos-tiny | 6.5M       |
| hustvl/yolos-base | 127M       |

## Running

### In your iOS and macOS applications

```swift
import CoreAIObjectDetector

// Load directly from an exported .aimodel directory.
let detector = try await ObjectDetector(resourcesAt: "coreai-models/exports/yolos-base_float32_static.aimodel")

// Single image, default parameters.
let detections = try await detector.detect(image: cgImage)

// Batched detection. For dynamic-shape exports, optionally override the spatial dims
// on DetectionParameters; for static exports the values are ignored.
var params = DetectionParameters()
params.inputHeight = 800
params.inputWidth = 1024
let batchDetections = try await detector.detect(images: [imageA, imageB], parameters: params)

// Optional: warm up the kernel for the exact (B, H, W) you'll run with.
try await detector.warmup(imageCount: 2, parameters: params)
```

### On your Mac using built-in Command Line Tool

```bash
# Single image, static-shape model.
swift run -c release object-detector --model path/to/exported_model.aimodel --image path/to/image.jpg

# Batched detection on a dynamic-shape export, with optional, explicit input dims and warmup.
swift run -c release object-detector \
  --model path/to/dynamic.aimodel \
  --image a.jpg --image b.jpg \
  --input-height 800 --input-width 1024 \
  --warmup
```

[^1]: [Paper](https://arxiv.org/abs/2106.00666) · [HuggingFace](https://huggingface.co/hustvl/yolos-tiny)
