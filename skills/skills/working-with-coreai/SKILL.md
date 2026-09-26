---
name: working-with-coreai
description: Use this skill whenever the user mentions coreai-torch, TorchConverter, coreai-build, AIModel, AIProgram, .aimodel, or wants to export/compile/run a PyTorch model on Apple silicon (iPhone, iPad, Mac). Also triggers for "deploy on device", "optimize for on-device performance", onboarding new models to Core AI, or choosing between iOS and macOS deployment paths.
---

# Working with Core AI

Deploy PyTorch models on Apple silicon: export with coreai-torch, compile with coreai-build, run with the Core AI runtime (Swift or Python).

**Related skills**: `Skill("coreai-skills:model-authoring")` (Neural Engine and GPU authoring patterns, use when re-structuring model architecture) | `Skill("coreai-skills:model-compression-exploration")` (quantization/palettization sweeps — use when exploring compression tradeoffs)

______________________________________________________________________

## Documentation and reference material

The Core AI toolchain has extensive documentation. Use these as reference — **do not read all pages upfront**. Instead, consult the relevant docs when you need specifics about a particular step.

| Resource | What it covers | When to consult |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [coreai-torch](https://apple.github.io/coreai-torch/index.html) | TorchConverter API, externalization, composite ops, custom lowerings, Metal kernels, debugging | Export questions, API details, custom op registration |
| [CoreAI framework](https://developer.apple.com/documentation/coreai) | AIModel, InferenceFunction, NDArray, specialization, caching | Swift runtime API, on-device integration |
| [coreai-build (AOT compilation)](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time) | Ahead-of-time compilation flags and options | Compilation questions |
| [coreai Python API](https://apple.github.io/coreai-torch/main/coreai-core) | Python runtime: AIModel, InferenceFunction, NDArray, state management | Python runtime questions |
| [coreai-models repo](https://github.com/apple/coreai-models) | Export recipes, Swift runtime utilities, reusable primitives | Export patterns, running models, reference implementations |
| [`guidance.md`](references/guidance.md) | Platform and general guidance: use cases, model sizing, compression strategy | Resolving decisions around platform targeting, model sizing, and compression strategy |

### coreai-models: the reference implementation

The [coreai-models](https://github.com/apple/coreai-models) repo is the canonical source for how to export and run models with Core AI. **Before writing export code from scratch, always explore this repo** — it has working export recipes for many model families, Swift and Python runtime utilities, and reusable primitives. If the user has a local clone, explore it. If not, suggest cloning it.

Explore these directories to find relevant patterns:

- **`models/`** — Per-model export recipes with READMEs and CLI commands for many popular model families (LLMs, vision, audio, diffusion).
- **`python/src/coreai_models/export/`** — Export pipeline code covering macOS and iOS export paths, compression presets, and custom MLIR lowerings.
- **`swift/Sources/`** — Runtime utilities for LLMs (engines, text generation, KV cache, sampling, decode loops), diffusion pipelines, object detection, image segmentation, and constrained decoding.

______________________________________________________________________

## Pipeline overview

The Core AI pipeline transforms a PyTorch model into an optimized on-device asset:

```text
1. AUTHOR        Re-structure model for target platform
                  → Skill("coreai-skills:model-authoring")

2. COMPRESS      Explore quantization/palettization tradeoffs
                  → Skill("coreai-skills:model-compression-exploration")

3. EXPORT        Convert PyTorch → AIProgram via TorchConverter
                  → coreai-torch docs

4. COMPILE       Ahead-of-time compilation for target platform
                  → coreai-build CLI

5. RUN           Load and run on device (Swift or Python)
                  → CoreAI framework / coreai Python API
```

Steps 1 and 2 are optional — many models export directly without re-authoring or compression. Start with export, then add authoring or compression if needed (poor accuracy, poor performance, too large).

For models already in [coreai-models](https://github.com/apple/coreai-models), the export recipes handle all steps. Check the `models/` directory first — if the user's model family is there, point them to the recipe.

______________________________________________________________________

## Export (Python — coreai-torch)

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table

model = MyModel().eval()
ep = torch.export.export(model, args=(torch.randn(1, 3, 224, 224),))
ep = ep.run_decompositions(get_decomp_table())

program = (
    TorchConverter()
    .add_exported_program(ep, input_names=["image"], output_names=["logits"])
    .to_coreai()
)
program.save_asset("model.aimodel")
```

This is the simplest export pattern. Real models often need more — consult the [coreai-torch docs](https://apple.github.io/coreai-torch/index.html) and explore the export code in the coreai-models repo for patterns around:

- **Externalization** of composite ops via `add_pytorch_module()` with `externalize_modules`
- **Mutable state** (e.g. KV cache) via `state_names`
- **Custom Metal kernels** via `TorchMetalKernel` and `register_torch_lowering()`
- **iOS static shape specialization** via `set_static_shape_config()`
- **Compression presets** for macOS vs iOS (different default strategies per platform)

______________________________________________________________________

## Compile (coreai-build CLI)

Ahead of time (AOT) compilation of models can optionally be performed with:
```bash
xcrun coreai-build compile model.aimodel --platform iOS
```

**Docs**: [Ahead-of-time compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time)

______________________________________________________________________

## Run (Swift)

```swift
import CoreAI

let model = try await AIModel(contentsOf: modelURL)
guard let fn = try model.loadFunction(named: "main") else { return }

var input = NDArray(shape: [1, 3, 224, 224], scalarType: .float32)
var view = input.mutableView(as: Float32.self)
// fill view with data...

var outputs = try await fn.run(inputs: ["image": input])
let result = outputs.remove("logits")?.ndArray
```

For LLMs, diffusion, and other complex models, explore the Swift runtime utilities in the coreai-models repo — they provide complete inference engines, decode loops, sampling, and KV cache management that handle the complexity beyond basic `AIModel` usage.

**Docs**: [CoreAI framework](https://developer.apple.com/documentation/coreai)

## Run (Python)

```python
from coreai.runtime import AIModel, NDArray
import numpy as np

model = await AIModel.load("model.aimodel")
fn = model.load_function("main")
outputs = await fn(
    {"image": NDArray(np.random.randn(1, 3, 224, 224).astype(np.float32))}
)
logits = outputs["logits"].numpy()
```

**Docs**: [coreai Python API](https://apple.github.io/coreai-torch/main/coreai-core)

______________________________________________________________________

## Verifying outputs (PSNR)

Compare on-device outputs against PyTorch reference:

| Scenario | Expected PSNR | Investigate if below |
| ------------------ | ------------- | -------------------- |
| float32 end-to-end | > 70 dB | 60 dB |
| fp16 on-device | > 50 dB | 40 dB |
| 4-bit palettized | ~40 dB | 30 dB |

______________________________________________________________________

## Onboarding new models

When helping a user deploy a model they haven't deployed before, follow this protocol. If the user already knows what they want (specific export question, compilation flag, etc.), skip directly to the relevant section above.

### First response: conversation, not code

Your first response is always a conversation. Even if the deployment path seems obvious, ask first.

**Rules:**

- Summarize what you understood (2-3 sentences)
- Ask 1-2 clarifying questions about what's unclear
- Do not recommend or dive into compute unit specific optimizations unless you have already reviewed the general and platform level guidance in [`guidance.md`](references/guidance.md)
- Do not produce compilation commands, code, or technical recommendations yet
- Only discuss what's on the user's path — don't mention irrelevant options
- Always check coreai-models for existing recipes before suggesting a from-scratch approach

**What to learn:**

- What are they building? (real-time camera, batch processing, background service, LLM chat)
- What platform? (iOS, macOS, multi-platform)
- What does the model look like? (architecture, size, input shapes, dynamic vs static)
- What matters most? (speed, energy efficiency, quality, broad device reach)

**Example interaction:**

> **User**: I want to deploy my image classifier on iPhone.
>
> **Assistant**: Got it — you're looking to run an image classification model on iPhone. A couple of questions to help me give you the best guidance:
>
> 1. Roughly how large is the model (parameter count or file size)?
> 2. Does it need to run in real-time (e.g., processing camera frames), or is batch/offline processing fine?
>
> **User**: It's a ResNet-50, about 25M params. Needs to run at 30fps on camera frames.
>
> **Assistant**: Perfect — a 25M param ResNet-50 for real-time camera on iPhone is a great fit for efficient on-device inference. Here's the plan...

### After the conversation

1. Check if the model family exists in [coreai-models](https://github.com/apple/coreai-models) — explore the `models/` directory for a matching recipe
2. If the user's needs involve platform targeting, model sizing, or compression strategy, read [`guidance.md`](references/guidance.md) to resolve the deployment path — present the outcome, not the reasoning
3. Walk through the pipeline steps relevant to their situation, consulting the docs above as needed
4. If the model needs architectural changes, invoke `Skill("coreai-skills:model-authoring")`
5. If compression tradeoffs need exploration, invoke `Skill("coreai-skills:model-compression-exploration")`
