# FLUX.2

Black Forest Labs' FLUX.2 diffusion models for on-device image generation via Core AI.

## Supported Models

| Model           | Parameters | macOS | iOS |
| --------------- | ---------- | ----- | --- |
| FLUX.2 Klein 4B | 4B         | Yes   | Yes |

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```bash
# Export for iOS (512x512 image resolution by default)
uv run coreai.diffusion.export flux2-klein-4b --platform iOS

# Override resolution (e.g. full 1024 on iOS)
uv run coreai.diffusion.export flux2-klein-4b --platform iOS --resolution 1024

# Export for macOS (1024x1024 image resolution by default)
uv run coreai.diffusion.export flux2-klein-4b --platform macOS

# Include half-resolution VAEs for low-memory tiled decode
# (only affects --single-function exports; the half VAEs are always included otherwise)
uv run coreai.diffusion.export flux2-klein-4b --platform macOS --single-function --low-memory

# Export all components (default -- no --platform flag)
uv run coreai.diffusion.export flux2-klein-4b
```

### Transformer Packaging

There are two ways to export the transformer. By default you get a single asset (`.aimodel`)
covering every supported resolution and reference grid, chosen at runtime. Pass `--single-function`
to get one asset per resolution/grid instead.

|                 | Default                      | `--single-function`           |
| --------------- | ---------------------------- | ----------------------------- |
| Assets          | one `Transformer.aimodel`    | one asset per resolution/grid |
| Disk            | ~2 GB total                  | ~2 GB **each**                |
| Peak memory     | higher                       | lower                         |
| Reference grids | all three, chosen at runtime | only the one you exported     |

The default packages the five variants (`main`, `half`, `img2img_quarter/half/full`) as
entrypoints in one multi-function `.aimodel` that shares one set of weights.
The multi-function transformer saves on disk space by sharing weights, but has a larger
peak memory footprint during runtime.

`--platform iOS` implies `--single-function` (to limit peak memory), 512 resolution, and
the `half` reference grid. Do not use `--platform iOS` if you intend to use multi-function.

Note that `--resolution` and `--low-memory` only apply to `--single-function` exports.
The multi-function asset covers both resolutions and always includes the half VAEs, so the
flags are irrelevant and will produce a warning if passed without `--single-function`.

### Image-to-image

Needs a VAE encoder plus an img2img transformer. Reference tokens from your input image
are concatenated onto the noise sequence, so adherence comes from `--reference-grid`,
not from noise blending (`--strength` is ignored).

The default already includes every grid. `--single-function` needs one chosen at export:

```bash
# iOS: 512 + half grid, included by default
uv run coreai.diffusion.export flux2-klein-4b --platform iOS

# Another grid needs --components, which requires --single-function
uv run coreai.diffusion.export flux2-klein-4b --single-function \
    --components transformer_512 transformer_512_img2img_quarter \
                text_encoder vae_decoder_half vae_encoder_half
```

**Other options:**

```bash
# Full precision (no compression)
uv run coreai.diffusion.export flux2-klein-4b --compression none

# Export specific components only
uv run coreai.diffusion.export flux2-klein-4b --components transformer text_encoder

# Custom output directory
uv run coreai.diffusion.export flux2-klein-4b --output-dir ./my-models/

# Preview resolved config without exporting
uv run coreai.diffusion.export flux2-klein-4b --dry-run
```

## Components

| Component                         | Description                                        | Platform           |
| --------------------------------- | -------------------------------------------------- | ------------------ |
| `transformer`                     | DiT (25 blocks), 1024x1024 image resolution        | macOS              |
| `transformer_512`                 | DiT (25 blocks), 512x512 image resolution          | iOS                |
| `transformer_img2img_<grid>`      | img2img at 1024, one asset per reference grid      | macOS, single-fn   |
| `transformer_512_img2img_<grid>`  | img2img at 512, one asset per reference grid       | iOS, single-fn     |
| `text_encoder`                    | Qwen3 encoder (intermediate layers 9, 18, 27)      | all                |
| `vae_decoder`                     | Latent to 1024x1024 pixel image                    | macOS              |
| `vae_decoder_half`                | Latent to 512x512 pixel image                      | iOS, macOS+low-mem |
| `vae_encoder`                     | 1024x1024 pixel image to latent (image-to-image)   | macOS              |
| `vae_encoder_half`                | 512x512 pixel image to latent (image-to-image)     | iOS, macOS+low-mem |

`<grid>` is `full`, `half`, or `quarter`: the reference token grid relative to the
output grid. At 1024x1024 decode resolution, that ends up being 4096 / 1024 / 256 reference tokens. At 512x512, it will be 1024 / 256 / 64.
Fewer tokens is faster and lighter, but with coarser guidance. The `transformer*_img2img_*`
assets exist only for single-function exports, whereas multi-function carries these as named
entrypoints inside `Transformer.aimodel`.

## Running

### In your iOS and macOS applications

```swift
import CoreAIDiffusionPipeline

// Pipeline auto-detects the best mode from available components
let pipeline = try await Flux2Pipeline(from: modelURL)

let config = PipelineConfiguration(
    prompt: "a photo of a cat",
    stepCount: 4,
    guidanceScale: 1.0,
    schedulerType: .discreteFlow
)

let result = try await pipeline.generateImages(
    configuration: config,
    progressHandler: { progress in
        print("Step \(progress.step)/\(progress.totalSteps)")
        return true
    }
)

let image = result.images.first!
```

For image-to-image, set `startingImage` and pick a grid the bundle has:

```swift
let config = PipelineConfiguration(
    prompt: "a cat wearing a hat",
    stepCount: 4,
    guidanceScale: 1.0,
    schedulerType: .discreteFlow,
    startingImage: inputCGImage,
    referenceGrid: .half            // .full | .half | .quarter
)
```

Asking for a grid the bundle doesn't have throws and lists what it does have. img2img is
unsupported with `.tiled` decode.

### On your Mac using built-in Command Line Tool

```bash
# Text-to-image
swift run -c release diffusion-runner --model path/to/exported_model_folder --prompt "a photo of a cat" --steps 4 --guidance-scale 1.0

# Image-to-image
swift run -c release diffusion-runner --model path/to/exported_model_folder --prompt "a cat wearing a hat" --steps 4 --guidance-scale 1.0 \
    --input-image cat.png --reference-grid half
```
