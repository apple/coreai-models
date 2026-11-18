# coreai-models — guidance for agents and reviewers

Core AI exports of models (Python, `python/src/coreai_models/`, `models/`), the Swift runtime and
tools that run them (`swift/Sources/`), and agent skills (`skills/`).

## Check your change

See `CONTRIBUTING.md` for setup. CI runs these checks; run them locally first:

```bash
uv run --frozen --only-group lint ruff check python/src/ python/tests/ models/
uv run --frozen --only-group lint ruff format --check python/src/ python/tests/ models/
swift format lint --strict --recursive swift/Sources/ swift/Tests/
uv run pytest python/tests/test_model_units -x -q
swift test --arch arm64 --filter <the suites you touched>   # needs Xcode 27
```

New functionality and bug fixes need tests, and every commit must be signed.

## Where things live

- `python/src/coreai_models/` is the export library. `primitives/` holds the building blocks
  (attention, norms, RoPE, caches, MLPs); `models/` holds the architectures built from them
  (`base.py` has the shared causal-LM base classes; `registry.py` maps a Hugging Face `model_type`
  to its macOS or iOS class). `export/` is the LLM and VLM export pipeline plus the shared
  compression, presets and bundle metadata; the other tasks convert through `converter.py`.
  `llm/`, `vlm/`, `embedding/`, `diffusion/` and `segmentation/` are the per-task front ends
  (`coreai.llm.export` and the like), and `model_registry.py` (`coreai.model.registry`) is the
  catalogue of supported models and presets.
- `primitives/` and `models/` each have a `macos/` and an `ios/` folder, and `export/` has
  `macos.py` and `ios.py`. The macOS code builds models for the GPU. The iOS code rewrites the LLMs
  and the EmbeddingGemma 2 encoders for the Neural Engine: float16 math, channels-first
  `(B, C, 1, S)` layouts, and an embedding table (int8 by default) behind separate load and gather
  entrypoints. iOS diffusion, CLIP and CLAP exports stay on the GPU. A layer in both folders is two
  separate implementations, not shared code: a fix to model behavior (config, RoPE scaling,
  masking) usually belongs in both, while a numerics or layout fix usually does not.
- `models/<name>/` at the repo root holds per-model READMEs, compression configs and, where needed,
  an `export.py`: either a self-contained script for a model outside the package, or a recipe built
  on `coreai_models`.
- `swift/Sources/` has one library per task (`CoreAILanguageModels`, which also covers VLMs,
  `CoreAIDiffusionPipeline`, `CoreAISpeech`, ...), shared code in `CoreAIShared`, the `llm-server`
  request types in `CoreAILMCommon`, the xgrammar bridge in `lib/CXGrammar`, and the command-line
  tools (`llm-runner`, `llm-server`, `benchmark`, ...) in `Tools/`.

## Reuse before you add

A common review finding is a second copy of something the repo already has. Search first; extend the
existing one, or say in the PR why it cannot be reused.

| Need                           | Already here (`python/src/coreai_models/`)                                                                                                                  |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RoPE, rotation, cos/sin        | `primitives/macos/rope.py` (`initialize_rope`, `RoPE`, `YarnRoPE`, `LongRoPE`), `primitives/ios/rope.py` (`apply_rope`, `rope_cos_sin`, `RoPECache`)        |
| RMSNorm, gated MLP             | `primitives/{macos,ios}/rms_norm.py`, `primitives/{macos,ios}/mlp.py`                                                                                       |
| Attention, masks               | `primitives/{macos,ios}/sdpa.py`, `primitives/ios/bidirectional_sdpa.py`; masks in `primitives/{macos,ios}/cache.py`                                        |
| KV and SSM caches              | `primitives/{macos,ios}/cache.py`, `primitives/ios/ssm_cache.py`                                                                                            |
| Mixture-of-experts layers      | `primitives/macos/switch.py` (`SwitchGLU`)                                                                                                                  |
| Causal-LM base, loading        | `models/base.py` (`BaseForCausalLM`, `BaseForCausalLMForiOS`)                                                                                               |
| A decoder layer to start from  | `models/macos/mistral.py`, `models/ios/mistral.py`; each family keeps its own `Attention` and `TransformerBlock`                                            |
| Hugging Face download, weights | `hub.py` (download), `_hf.py` (weight files, RoPE config)                                                                                                   |
| Constants, metadata            | `_constants.py` (graph and cache names, metadata version), `export/metadata.py` (per-asset author and license), `export/bundle.py` (bundle `metadata.json`) |

The table names today's files. If one has moved, look through `primitives/` and `models/` before
writing a new helper. Some look-alikes differ on purpose: the iOS `rotate_half` takes its axis as an
argument (channels-first callers pass `dim=2`), and the iOS `RMSNorm` computes in the input dtype
where the macOS one upcasts to float32. If two helpers look identical but you cannot tell why both
exist, ask before merging them.

Dependency direction in the Python package: the task front ends → `export/` → `models/` →
`primitives/`, never the reverse. `_constants.py` is a leaf module; `_hf.py` depends only on `hub.py`.

## Change every sibling

A fix that lands in one of several parallel paths and not the others is the most common bug here.
When you change one, check and change its siblings in the same PR, or say why not:

- the three LLM engines in `swift/Sources/CoreAILanguageModels/InferenceEngines/`:
  `CoreAISequentialEngine`, `StaticShapeEngine` and `CoreAIPipelinedEngine` (prefill, reset and
  rewind, sampling, max-token and cancel handling). Sequential and static-shape share `DecodeLoop`;
  the pipelined engine has its own copy of each. The batched lane that `llm-server` uses
  (`SchedulerSession`, `BatchSessionAdapter`) has its own stop and EOS checks;
- the decoding strategies (vanilla, constrained, pipelined constrained, `ConstrainedGenerator`) and
  the speculative decoder: each runs its own stop check, with different single- and multi-token
  behavior;
- `llm-server` and `llm-runner` (`swift/Sources/Tools/`) for anything both do with a request:
  tokenization, stop strings, sampling options;
- the `metadata.json` writers (`export/bundle.py` and the diffusion, embedding, segmentation and VLM
  exporters) and the Swift readers `ModelBundle` and `LanguageModelBundle`. A metadata version
  change touches `_constants.py` and `ModelBundle.supportedMetadataVersions`.

## Swift rules that have caused bugs

- `tokenizer.encode(text:)` adds the tokenizer's special tokens (often a BOS). For text that is not a
  whole prompt (stop strings, forced or jump-forward text), use `encode(text:addSpecialTokens: false)`.
- A request field that is decoded must be used or refused; do not decode and ignore it.
- Prefer an exhaustive `switch` with no `default:` over the repo's own enums, so a new case does not
  compile until every place decides what it means.
- A cache keyed on one field of a configuration goes stale when another field changes; key on the
  whole value (`SamplingConfiguration` is `Hashable`).

## Words

"Export" is the whole Hugging Face model → `.aimodel` process; "conversion" is only the torch → Core AI
program step (`coreai_torch.TorchConverter`). Say "Neural Engine" in text, not "ANE"; name it
`neural_engine` in Python and `neuralEngine` in Swift. Platform values are `macOS` and `iOS`; folder
names are lowercase.
