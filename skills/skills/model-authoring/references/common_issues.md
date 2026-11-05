# Common Issues and Fixes

## General rules

- **Float32 constants**: Any Python float literal (e.g., `x * 2.0`) creates an f32 constant Neural Engine rejects. Cast to float16.
- **Always use float16 weights**.
- **Layout mismatch in comparisons**: Apply the appropriate transform before PSNR. Never compare raw tensors across layouts.
- **Non-contiguous tensors**: Call `.contiguous()` on ALL tensors before wrapping in `NDArray`.

______________________________________________________________________

## Neural Engine SDPA PSNR very low (~15-30 dB)

**Cause**: Causal mask orientation is `(1, query, 1, key)` instead of `(1, key, 1, query)`.
**Fix**: Transpose mask or use `create_ane_causal_mask()` from `neural_engine_rules.md`.

______________________________________________________________________

## Input data type mismatch — "Data type int32 does not match"

**Cause**: Input JSON descriptor uses wrong type specifier.
**Fix**: Use `"si32"` (signed int32), not `"i32"`.

______________________________________________________________________

## Core AI import error about input counts

**Cause**: Input names include PARAMETER and CONSTANT_TENSOR entries folded away after `run_decompositions()`.
**Fix**: Filter to only USER_INPUT and BUFFER kinds:

```python
from torch.export.graph_signature import InputKind

live_kinds = {InputKind.USER_INPUT, InputKind.BUFFER}
input_names = [
    s.arg.name for s in ep.graph_signature.input_specs if s.kind in live_kinds
]
```

______________________________________________________________________

## Core AI export fails with "op has no known lowering"

**Cause**: Model uses vanilla PyTorch ops with no Core AI lowering.
**Fix**: Use Core AI-compatible primitives or re-author layers as Conv2d for Neural Engine. See `neural_engine_rules.md` for supported patterns.

______________________________________________________________________

## Neural Engine MLP — 3 invalid ops from `mps.swish`

**Cause**: `nn.functional.silu(x)` lowers to `mps.cast(→f32) + mps.swish(f32) + mps.cast(→f16)`.
**Fix**: `gate_pre * torch.sigmoid(gate_pre)` instead of `silu()`.

______________________________________________________________________

## Neural Engine RoPE — `gather_nd` produces 3D output

**Cause**: Indexing a 2D cos/sin table with `position_ids: [B, S]` produces 3D output Neural Engine rejects.
**Fix**: Compute cos/sin outside the model, pass as 4D `(1, head_dim, 1, S)` BC1S inputs.

______________________________________________________________________

## Neural Engine M-RoPE PSNR very low (~18 dB)

**Cause**: GPU M-RoPE pattern not reproduced exactly.
**Fix**: Match `torch.cat([cos, cos], dim=-1)` then index with `::2`.

______________________________________________________________________

## HF model fails during `post_init()` — missing `rope_parameters`

**Fix**: Patch `ROPE_INIT_FUNCTIONS` before instantiation:

```python
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

if "default" not in ROPE_INIT_FUNCTIONS:

    def _default_rope(config=None, device=None, seq_len=None, **kwargs):
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _default_rope
```

______________________________________________________________________

## Neural Engine wrong logits — non-contiguous tensors

**Cause**: The runtime reads raw memory as if contiguous, ignoring tensor strides.
**Fix**: Call `.contiguous()` on ALL tensors before wrapping in `NDArray`.

______________________________________________________________________

## Neural Engine causal mask with `float('-inf')`

**Cause**: Neural Engine does not handle IEEE `-inf` correctly in softmax.
**Fix**: Use `-40000.0` — representable in fp16, `exp(-40000)` is zero.

______________________________________________________________________

## Neural Engine model compiles but runs on CPU

**Cause**: Model was compiled without Neural Engine preference (e.g., using default compute selection which routed to CPU/GPU).
**Fix**: Compile with `xcrun coreai-build compile model.aimodel --preferred-compute neural-engine`.

______________________________________________________________________

## `embed_tokens()` — `.detach()` before `.numpy()`

**Cause**: `nn.Embedding.__call__` returns tensor with `requires_grad=True`.
**Fix**: `model.embed_tokens(torch.tensor([[token_id]])).half().detach()`

______________________________________________________________________

## Neural Engine sequential q=1 decode diverges for long prefill

**Cause**: fp16 rounding errors compound across many sequential per-token passes.
**Fix**: Use chunked prefill (S_q=64) or fp32 KV cache tensors in Python.

______________________________________________________________________

## AdaLN conditioning — invalid Neural Engine ops from Python float literals

**Cause**: `1.0 + scale_msa` — Python float `1.0` creates f32 constant.
**Fix**: `one = torch.tensor(1.0, dtype=x.dtype, device=x.device)`
For `torch.tanh`: replace with `2 * torch.sigmoid(2 * x) - 1`.

______________________________________________________________________

## Old `.aimodel` not loadable

**Cause**: Asset was created with an older toolchain and uses an incompatible format.
**Fix**: Re-export and save via `deployable.save_asset()`, then recompile with `coreai-build compile`.

______________________________________________________________________

## Stateful transforms — KV cache resets between Python calls

**Cause**: Stateful transform APIs mark buffers stateful within one invocation, but state resets between inference calls.
**Fix**: Use readonly KV I/O pattern. Pass caches as explicit inputs/outputs.

______________________________________________________________________

## `runner(**inputs)` fails — wrong call signature

**Cause**: `InferenceFunction.__call__` takes one `inputs` dict, so `runner(**inputs)` raises `TypeError: ... got an unexpected keyword argument`.
**Fix**: Use `await runner(inputs)`, not `await runner(**inputs)`.

______________________________________________________________________

## Output dict key order non-deterministic — k/v swapped

**Cause**: Output dicts have non-deterministic key ordering.
**Fix**: Identify outputs by shape, not index. Distinguish k vs v by MSE against known-zero input.

______________________________________________________________________

## Activation function mismatch

**Cause**: Wrong activation type (SiLU vs QuickGELU vs GELU vs SwiGLU) gives PSNR ~20-30 dB.
**Fix**: Print `type()` from source model before re-authoring.

______________________________________________________________________

## State dict key mismatch during weight loading

**Cause**: Re-authored model uses different attribute names.
**Fix**: Print source state dict keys before writing remap.
