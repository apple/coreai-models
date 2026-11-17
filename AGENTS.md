# coreai-models — guidance for agents and reviewers

Core AI exports of ML models (Python, `python/src/coreai_models/`, `models/`) and the Swift runtime and
tools that run them (`swift/Sources/`).

## Check your change

See `CONTRIBUTING.md` for setup. The checks a PR must pass locally:

```bash
uv run --frozen --only-group lint ruff check python/src/ python/tests/ models/
uv run --frozen --only-group lint ruff format --check python/src/ python/tests/ models/
uv run pytest python/tests/test_model_units -x -q
swift build && swift test --filter <the suites you touched>
```

## Reuse before you add

A common review finding is a second copy of something the repo already has. Search first; extend the
existing one, or say in the PR why it cannot be reused.

| Need                          | Already here (`python/src/coreai_models/`)                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| RoPE, rotation, cos/sin       | `primitives/macos/rope.py` (`RoPE`, `YarnRoPE`), `primitives/ios/rope.py`                             |
| RMSNorm, MLP                  | `primitives/{macos,ios}/rms_norm.py`, `primitives/macos/mlp.py`                                       |
| Attention, masks              | `primitives/{macos,ios}/sdpa.py`                                                                      |
| A whole decoder layer         | `models/macos/mistral.py` (`Attention`, `TransformerBlock`) and the other decoders in `models/macos/` |
| Hugging Face download, config | `_hf.py`                                                                                              |
| Constants, bundle metadata    | `_constants.py`, `export/metadata.py`                                                                 |

Some look-alikes differ on purpose (for example the iOS `rotate_half` rotates a different axis, and the
iOS `RMSNorm` has no float16 prescale). If two helpers look identical but you cannot tell why both exist,
ask before merging them.

Dependency direction in the Python package: `export/` → `models/` → `primitives/`, never the reverse.
`_hf.py` and `_constants.py` are leaf modules.

## Change every sibling

A fix that lands in one of several parallel paths and not the others is the most common bug here.
When you change one, check and change its siblings in the same PR, or say why not:

- the three LLM engines in `swift/Sources/CoreAILanguageModels/InferenceEngines/`: sequential,
  static-shape and pipelined (prefill, rewind, reset, sampler and stop handling);
- the decoding strategies (vanilla, constrained, pipelined constrained) and their stop checks;
- `llm-server` and `llm-runner` (`swift/Sources/Tools/`) for anything both do with a request:
  tokenization, stop strings, sampling options;
- the Python metadata writer (`export/metadata.py`) and the Swift bundle reader that loads it.

## Swift rules that have caused bugs

- `tokenizer.encode(text:)` adds the tokenizer's special tokens (often a BOS). For text that is not a
  whole prompt (stop strings, forced or jump-forward text), use `encode(text:addSpecialTokens: false)`.
- A request field that is decoded must be used or refused; do not decode and ignore it.
- Prefer an exhaustive `switch` with no `default:` over the repo's own enums, so a new case does not
  compile until every place decides what it means.
- A cache keyed on one field of a configuration goes stale when another field changes; key on the
  whole value (`SamplingConfiguration` is `Hashable`).

## Words

"Export" is the whole HF model → `.aimodel` process; "conversion" is only the torch → MLIR step. Say
"Neural Engine" in text and use `neural_engine` in identifiers. Platform values are `macOS` and `iOS`.
