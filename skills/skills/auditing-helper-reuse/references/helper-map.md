# Helper map

Paths are relative to `python/src/coreai_models/`. Checked against apple/coreai-models `f5d220b8`
(2026-11-16); confirm each entry in the tree before relying on it.

## Where helpers live

| Family | Look in |
| --- | --- |
| A decoder layer to start from | `models/macos/mistral.py`, `models/ios/mistral.py`; each family keeps its own `Attention` and `TransformerBlock`, so copy one rather than import it |
| RoPE, rotation, cos/sin | `primitives/macos/rope.py` (`RoPE`, `YarnRoPE`), `primitives/ios/rope.py`, `models/macos/diffusion_gemma.py` (`apply_rotary_pos_emb`), `models/macos/muse_glimmer_vision.py` (private `_rotate_half`: promote, don't import) |
| Norms, MLP, embeddings | `models/macos/diffusion_gemma.py` (`DiffusionGemmaRMSNorm`, `DiffusionGemmaDenseMLP`, `ScaledEmbedding`), `primitives/ios/rms_norm.py` |
| Attention and masks | `primitives/{macos,ios}/sdpa.py`, `primitives/ios/bidirectional_sdpa.py` |
| Weights and snapshots | `models/base.py` (`resolve_safetensors_files`), `_hf.py`; `vlm/export.py` still keeps its own `_get_safetensors_files` |
| Export boilerplate | `converter.py` (`export_stateless`, `make_converter`, `add_function`, private `_decomp_empty_permuted`), `diffusion/gpu.py` (`export_multifunction`, built on `add_function`), `register_custom_torch_lowering`, `coreai_torch.get_decomp_table()` |
| Constants and metadata | `_constants.py` (`METADATA_VERSION`, `DEFAULT_INCLUDE_DEBUG_INFO`), `export/metadata.py` |
| CLI | each exporter's `_default_output_dir` (a per-exporter copy is the existing pattern) |
| Tests | shared tiny configs and comparison helpers next to the model's tests |

## Look-alikes (each looked identical and was not)

| Looks reusable | Why it is not |
| --- | --- |
| iOS `rotate_half` / `apply_rope` for a BC1S tower | rotates the last dim; BC1S rotates the channel axis, adding transposes to the Neural Engine graph |
| `primitives/ios/rope.rotate_half` for `x[..., mid:]` | slices `mid : x.shape[-1]`, a different slice constant in the graph |
| `RoPECache` for in-forward cos/sin | same values, different graph (baked tables, gather op) |
| `primitives/ios/rms_norm.RMSNorm` | no max-abs prescale (float16 overflow); weight init differs |
| `gelu_ane` | sigmoid approximation, not tanh GELU |
| `BidirectionalSDPA` | fixed `head_dim**-0.5` scale, per-head loop, different mask constant |
| Existing mask builders | all causal; a bidirectional band with key padding is not one of them |
| SDPA's built-in GQA in place of `repeat_interleave` | same outputs, different ops (`index_select`) |
| `export_multifunction` | static shapes only, one input and output name set for every function; build with `make_converter` + `add_function` for dynamic shapes or per-function names |
| `add_function` with its default table | Core AI's decompositions without the `empty_permuted` one that `export_stateless` adds; pass `decompositions=` if the graph needs it |
| A shared safetensors loader or snapshot download | different file discovery or `allow_patterns` = behaviour change |
| A peer model's `ForCausalLM` | state-dict names can differ (e.g. a top-level `embed_tokens`) |
| A peer model's vision MLP or attention | exact vs tanh GELU; RoPE or 2D layouts differ |

## Spec pitfalls

- Module-level flags (e.g. `USE_FUSED_KV`) are read from the module that defines the code after the
  change; toggle them there.
- Metadata writers stamp dates; drop them or snapshot the underlying table.
- MLIR resource names are random per process; hash with `mlir_hash`, never the raw text.
- A spec whose test helpers live under `python/tests` should find them relative to `--src`.
