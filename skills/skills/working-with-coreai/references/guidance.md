# Guidance

General and platform-level guidance for preparing and deploying models on Apple platforms. Read this when resolving decisions around platform targeting, model sizing, and compression strategy for your use case.

______________________________________________________________________

## General guidance

Applies regardless of target OS or platform.

### Model sources

Prefer established sources to reduce integration risk:

- **Apple's coreai-models repo** — export recipes and runtime utilities for many model families; check here first
- **Hugging Face** — broadly validated ready-to-use models
- **Custom PyTorch model** — convert directly via TorchConverter; the most well-supported path for custom models

### Model compression and optimization

Float16 is the recommended default precision. Beyond precision, explore quantization and palettization to find the best tradeoff between model size and output quality for your use case.

### Model size

A model that consumes too much memory can degrade system performance or be terminated by the OS. Choose model sizes that leave a reasonable buffer for your app and the broader system. Use `os_proc_available_memory()` at runtime to query available memory and make informed loading decisions.

______________________________________________________________________

## Platform guidance

### Use cases

| Platform | Suitable workloads |
| -------- | ------------------ |
| iOS | Foreground AI experiences. Background execution is subject to iOS resource management policies — comply with OS guidelines and use entitlements only where applicable. |
| macOS | Foreground and background workloads. Well suited for real-time interactive use cases as well as batch and offline processing. |

### Model sizes

| Platform | Recommendation |
| -------- | -------------- |
| iOS | Keep models under 2 GB |
| macOS | Leave at least 6 GB of RAM headroom for the system and other processes |

### Compression and optimization

**iOS** — optimize for energy efficiency:

- Static shaped inputs, outputs, and intermediate tensors wherever possible
- Limited or no control flow or branching
- Int8/Int4 linear quantized with per-channel granularity or 2/4/6/8 bit palettized weights with per-tensor or per-group-channel granularity. Avoid linear per-block quantization (for example the macOS `4bit` preset) for a model that runs on the Neural Engine: it has produced wrong output with no error ([issue #279](https://github.com/apple/coreai-models/issues/279)). The in-tree iOS presets are palettization only.

Models with variable sequence lengths can be transformed and chunked into a collection of multiple static shaped functions. In some cases a fixed max shape is required — for example, picking a maximum context length and using it to set a fixed-size KV cache.

**macOS** — optimize for scale with available compute and memory:

- Models need not be restricted to static shapes and can have data dependencies and control flow
- Int4 linear per-block quantization is recommended for weight compression

### Specialization options

Use `.default` specialization options at runtime for each platform — this gives Core AI the most flexibility to optimize execution on device.

If you override the default and set a preferred compute unit explicitly, align the model representation to match:

| Preferred compute unit | Recommended model representation |
| ---------------------- | --------------------------------- |
| Neural Engine | Static shapes, palettized weights (not linear per-block quantization), optimized for energy efficiency |
| GPU | Linear quantization, no chunked dynamic shapes, optimized to scale with available compute |

Whichever unit you pin, compare the compressed model's output against an fp32 reference on that unit. A per-block quantized `Linear` has been reported to return wrong values on the GPU as well as the Neural Engine on macOS (see [issue #279](https://github.com/apple/coreai-models/issues/279)), and the runtime gives no warning.
