# coreai-models — guidance for agents and reviewers

Core AI Python exports of models (`python/src/coreai_models/`, `models/`), the Swift runtime and tools
that run them (`swift/Sources/`), and agent skills (`skills/`).

## Check your change

See `CONTRIBUTING.md` for setup. CI runs these checks; run them locally first:

```bash
uv run --frozen --only-group lint ruff check python/src/ python/tests/ models/
uv run --frozen --only-group lint ruff format --check python/src/ python/tests/ models/
swift format lint --strict --recursive swift/Sources/ swift/Tests/
uv run --frozen pytest python/tests/test_model_units -x -q
swift test --arch arm64 --disable-sandbox --filter <the suites you touched>   # needs Xcode 27
```

`--disable-sandbox` turns off SwiftPM's own sandbox, which cannot start inside another one (an agent's
sandbox, a CI task). `--frozen` keeps uv from re-locking. Do not commit `uv.lock` unless a
`pyproject.toml` changed; discard an accidental change with `git checkout -- uv.lock`.

New functionality and bug fixes need tests, and every commit must be signed.

## Where things live

- `python/src/coreai_models/` is the export library. `primitives/` holds the building blocks
  (attention, norms, RoPE, caches, MLPs), `models/` the architectures built from them, and `export/`
  the LLM and VLM export pipeline plus the shared compression, presets and bundle metadata. `llm/`,
  `vlm/`, `embedding/`, `diffusion/` and `segmentation/` are the per-task front ends
  (`coreai.llm.export` and the like; segmentation runs through `models/sam3/export.py` and
  `models/sam3_video/export.py`). `model_registry.py` (`coreai.model.registry`) is the catalogue of
  supported models and presets.
- `primitives/` and `models/` each have a `macos/` and an `ios/` folder, and `export/` has
  `macos.py` and `ios.py`. The macOS code builds models for the GPU. The iOS code rewrites the LLMs
  and the EmbeddingGemma 2 encoders for the Neural Engine, with float16 math and channels-first
  layouts. iOS diffusion, CLIP and CLAP exports stay on the GPU. A layer in both folders is two
  separate implementations, not shared code: a fix to model behavior (config, RoPE scaling,
  masking) usually belongs in both, while a numerics or layout fix usually does not.
- `models/<name>/` at the repo root holds per-model READMEs, compression configs and, where needed,
  an `export.py`.
- `swift/Sources/` has one library per task (`CoreAILanguageModels`, `CoreAIDiffusionPipeline`,
  `CoreAISpeech`, ...), shared code in `CoreAIShared`, the `llm-server` request types in
  `CoreAILMCommon`, and the command-line tools in `Tools/`.

## Reuse before you add

A common review finding is a second copy of something the repo already has. Search first; extend the
existing one, or say in the PR why it cannot be reused.

| Need                           | Already here (`python/src/coreai_models/`)                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| RoPE, rotation, cos/sin        | `primitives/macos/rope.py`, `primitives/ios/rope.py`                                                                 |
| RMSNorm, gated MLP             | `primitives/{macos,ios}/rms_norm.py`, `primitives/{macos,ios}/mlp.py`                                                |
| Attention, masks               | `primitives/{macos,ios}/sdpa.py`, `primitives/ios/bidirectional_sdpa.py`; masks in `primitives/{macos,ios}/cache.py` |
| KV and SSM caches              | `primitives/{macos,ios}/cache.py`, `primitives/ios/ssm_cache.py`                                                     |
| Mixture-of-experts layers      | `primitives/macos/switch.py`                                                                                         |
| Causal-LM base, loading        | `models/base.py`                                                                                                     |
| A decoder layer to start from  | `models/macos/mistral.py`, `models/ios/mistral.py`; each family keeps its own `Attention` and `TransformerBlock`     |
| Hugging Face download, weights | `hub.py` (download), `_hf.py` (weight files, RoPE config)                                                            |
| Constants, metadata            | `_constants.py`, `export/metadata.py` (per-asset), `export/bundle.py` (bundle `metadata.json`)                       |

| Need                                   | Already here (`swift/Sources/`)                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Bundle loading, `metadata.json`        | `CoreAIShared/Bundle/ModelBundle.swift`, `CoreAILanguageModels/Bundle/LanguageModelBundle.swift` |
| Loading a model on the right device    | `CoreAIShared/Runtime/ModelStructure.swift` (`PreparedModel`)                                    |
| Seeded RNG that matches NumPy or Torch | `CoreAIShared/RNG/`                                                                              |
| Image preprocessing, `.npy` files      | `CoreAIShared/Image/ImagePreprocessor.swift`, `CoreAIShared/Runtime/NpyArray.swift`              |
| Sampling, stop sequences               | `CoreAILanguageModels/Samplers/`, `StopSequences` in `DecodingStrategies/DecodingStrategy.swift` |
| CLI logging                            | `CoreAIShared/Logger/Logger.swift` (`CLILogger`)                                                 |

The tables name today's files. If one has moved, search the same folders before writing a new
helper. Some look-alikes differ on purpose: the iOS `rotate_half` takes its axis as an argument
(channels-first callers pass `dim=2`), and the iOS `RMSNorm` computes in the input dtype where the
macOS one upcasts to float32. If two helpers look identical but you cannot tell why both exist, ask
before merging them.

Dependency direction in the Python package: the task front ends → `export/` → `models/` →
`primitives/`, never the reverse. Below them, `_hf.py` → `hub.py` → `model_registry.py` →
`_constants.py`; none of these imports `models/` or `export/`.

## Change every sibling

A fix that lands in one of several parallel paths and not the others is the most common bug here.
When you change one, check and change its siblings in the same PR, or say why not:

- the three LLM engines in `swift/Sources/CoreAILanguageModels/InferenceEngines/`
  (`CoreAISequentialEngine`, `StaticShapeEngine`, `CoreAIPipelinedEngine`) and the batched lane that
  `llm-server` uses (`SchedulerSession`, `BatchSessionAdapter`);
- the decoding strategies (vanilla, constrained, pipelined constrained, `ConstrainedGenerator`) and
  the speculative decoder, each of which runs its own stop check;
- `llm-server` and `llm-runner` (`swift/Sources/Tools/`) for anything both do with a request:
  tokenization, stop strings, sampling options;
- the `metadata.json` writers in each exporter and the Swift readers `ModelBundle` and
  `LanguageModelBundle`, including the metadata version on both sides;
- when adding or renaming a model, `model_registry.py` (presets and the catalogue) and
  `models/registry.py` (Hugging Face `model_type` → macOS or iOS class).

## Swift rules that have caused bugs

- `tokenizer.encode(text:)` adds the tokenizer's special tokens (often a BOS). For text that is not a
  whole prompt (stop strings, forced or jump-forward text), use `encode(text:addSpecialTokens: false)`.
- A request field that is decoded must be used or refused; do not decode and ignore it.
- Prefer an exhaustive `switch` with no `default:` over the repo's own enums, so a new case does not
  compile until every place decides what it means.
- A cache keyed on one field of a configuration goes stale when another field changes; key on the
  whole value (`SamplingConfiguration` is `Hashable`).
- A reply that ends at a stop sequence can leave the engine's generation task running. Await
  `cancel()` on the engine before the process exits or the engine is released.

## Words

"Export" is the whole Hugging Face model → `.aimodel` process; "conversion" is only the torch → Core AI
program step. Say "Neural Engine" in text, not "ANE"; name it `neural_engine` in Python and
`neuralEngine` in Swift. Platform values are `macOS` and `iOS`.
