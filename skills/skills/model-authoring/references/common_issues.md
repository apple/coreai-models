# Common Issues and Fixes

## General rules

- **Float32 constants**: Any Python float literal (e.g., `x * 2.0`) creates an f32 constant Neural Engine rejects. Cast to float16.
- **Always use float16 weights**.
- **4-bit weights for Neural Engine**: Palettize; do not quantize to INT4. A static Qwen3-0.6B export with per-block INT4 weights compiles for h18p with exit 0 and no Neural Engine region. With the iOS default preset, `4bit_weight_palettized_group32`, it compiles to 30 regions.
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

## Neural Engine MLP — SiLU less precise than on the GPU

**Cause**: With coreai-build 3600.83.1, `silu()` compiles to one Neural Engine region (h16c, h18p) with no invalid ops, and `x * torch.sigmoid(x)` returns the same bits there. Conv → SiLU → conv with random fp16 weights reaches 47 dB against fp32 below, the GPU 83 dB. Over the 28 MLP blocks of Qwen3-0.6B on one prompt, the median (lowest) PSNR is 67.6 (55.2) dB.
**Fix**: Keep `silu()`. If a model needs more, `x * (c * torch.tanh(x * c) + c)`, with `c` an fp16 buffer holding 0.5, reaches 71 dB and 81.6 (69.7) dB. On an M4 Max Neural Engine one Qwen3-0.6B MLP block then takes 2.74 ms, not 0.40 ms, at 64 tokens (equal at 1 token).

```python
import asyncio
import tempfile
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
from coreai_torch import TorchConverter, get_decomp_table
from torch import nn


class TanhSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Buffer(torch.tensor(0.5))

    def forward(self, x):
        return x * (self.c * torch.tanh(x * self.c) + self.c)


async def run(path, x, unit):
    options = SpecializationOptions.from_preferred_compute_unit_kind(unit)
    model = await AIModel.load(path, specialization_options=options)
    return (await model.load_function("main")({"x": NDArray(x)}))["y"].numpy()


x = np.random.default_rng(0).standard_normal((1, 256, 1, 64)).astype(np.float16)
names = {"input_names": ["x"], "output_names": ["y"]}
for act in (nn.SiLU(), TanhSiLU()):
    torch.manual_seed(0)
    module = nn.Sequential(nn.Conv2d(256, 1024, 1), act, nn.Conv2d(1024, 256, 1))
    module = module.eval().half()
    ep = torch.export.export(module, (torch.from_numpy(x),))
    ep = ep.run_decompositions(get_decomp_table())
    path = Path(tempfile.mkdtemp()) / "m.aimodel"
    TorchConverter().add_exported_program(ep, **names).to_coreai().save_asset(path)
    ref = module.float()(torch.from_numpy(x).float()).detach().numpy()
    for unit in (ComputeUnitKind.gpu(), ComputeUnitKind.neural_engine()):
        mse = np.mean((asyncio.run(run(path, x, unit)) - ref) ** 2)
        print(type(act).__name__, unit, 10 * np.log10(np.abs(ref).max() ** 2 / mse))
```

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

## Neural Engine wrong for tiny divisors, clamp floors and eps

**Cause**: In three cases the Neural Engine departs from CPU and GPU. `x / v` with `|v|` below 2^-16 (about 1.5e-5), including 0, returns `x * 65536` with the sign of `v`. `v.clamp(min=c)` with `c` below 2^-14, the smallest normal fp16 (`1e-6`, `2e-5`, also `6.1e-5`, which rounds below it), returns 0 where it clamps. `torch.rsqrt(v + 1e-6)` returns inf at `v = 0` and 256, not 301, at `v = 1e-5`.
**Fix**: Floor with `v.clamp(min=torch.finfo(torch.float16).tiny)` (2^-14) or higher; both units agree below. Adding 2^-14 instead was 7% off at `v = 1e-5`. On the Neural Engine, `x / torch.sqrt(...)` and `x * torch.rsqrt(...)` give the same bits, and the iOS `RMSNorm` primitive (eps 1e-6) is right, all-zero rows included.

```python
import asyncio
import tempfile
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
from coreai_torch import TorchConverter, get_decomp_table

TINY = torch.finfo(torch.float16).tiny  # 2^-14, the smallest normal fp16


class Small(torch.nn.Module):
    def forward(self, x, v):
        floors = (x / v.clamp(min=2e-5), x / v.clamp(min=TINY))
        return x / v, *floors, torch.rsqrt(v + 1e-6), torch.rsqrt(v.clamp(min=TINY))


async def run(path, feeds, unit):
    options = SpecializationOptions.from_preferred_compute_unit_kind(unit)
    model = await AIModel.load(path, specialization_options=options)
    outputs = await model.load_function("main")(feeds)
    return [outputs[f"y{i}"].numpy()[0, 0, 0, :5] for i in range(5)]


v = np.tile(np.array([0, 1e-6, 1e-5, 2e-5, 1e-3], np.float16), (1, 64, 1, 8))
x = np.full_like(v, 1e-6)
names = {"input_names": ["x", "v"], "output_names": [f"y{i}" for i in range(5)]}
inputs = (torch.from_numpy(x), torch.from_numpy(v))
ep = torch.export.export(Small().eval().half(), inputs)
ep = ep.run_decompositions(get_decomp_table())
path = Path(tempfile.mkdtemp()) / "m.aimodel"
TorchConverter().add_exported_program(ep, **names).to_coreai().save_asset(path)
forms = ["x / v", "x / v.clamp(min=2e-5)", "x / v.clamp(min=TINY)"]
forms += ["rsqrt(v + 1e-6)", "rsqrt(v.clamp(min=TINY))"]
feeds = {"x": NDArray(x), "v": NDArray(v)}
for unit in (ComputeUnitKind.gpu(), ComputeUnitKind.neural_engine()):
    for form, y in zip(forms, asyncio.run(run(path, feeds, unit))):
        print(f"{unit}, {form}:", y)
```

______________________________________________________________________

## Neural Engine model compiles but runs on CPU or GPU

**Cause**: Model was compiled without Neural Engine preference. Without `--preferred-compute neural-engine`, a macOS (h16c) compile of a static fp16 graph has no Neural Engine region; an iOS (h18p) compile has one. The flag is only a preference: with an fp32 graph or a dynamic dimension, `coreai-build compile` still exits 0, prints no error, and nothing runs on the Neural Engine.
**Fix**: Export fp16 with static shapes, and compile with `xcrun coreai-build compile model.aimodel --preferred-compute neural-engine`. When nothing runs on the Neural Engine, a load that prefers `ComputeUnitKind.neural_engine()` returns the same bits as one that prefers `gpu()`, so different bits mean some or all of it does.

```python
import asyncio
import tempfile
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
from coreai_torch import TorchConverter, get_decomp_table
from torch import nn


async def run(path, x, unit):
    options = SpecializationOptions.from_preferred_compute_unit_kind(unit)
    model = await AIModel.load(path, specialization_options=options)
    return (await model.load_function("main")({"x": NDArray(x)}))["y"].numpy()


def same_bits_as_gpu(module, x, dynamic_shapes=None):
    inputs = (torch.from_numpy(x),)
    ep = torch.export.export(module.eval(), inputs, dynamic_shapes=dynamic_shapes)
    ep = ep.run_decompositions(get_decomp_table())
    path = Path(tempfile.mkdtemp()) / "m.aimodel"
    names = {"input_names": ["x"], "output_names": ["y"]}
    TorchConverter().add_exported_program(ep, **names).to_coreai().save_asset(path)
    gpu = asyncio.run(run(path, x, ComputeUnitKind.gpu()))
    neural_engine = asyncio.run(run(path, x, ComputeUnitKind.neural_engine()))
    return np.array_equal(gpu, neural_engine)


torch.manual_seed(0)
module = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 1))
x = np.random.default_rng(0).standard_normal((1, 64, 1, 64)).astype(np.float16)
dim = {3: torch.export.Dim("S", min=2, max=512)}
print("fp16:", same_bits_as_gpu(module.half(), x))
print("fp32:", same_bits_as_gpu(module.float(), x.astype(np.float32)))
print("fp16, dynamic width:", same_bits_as_gpu(module.half(), x, (dim,)))
```

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
`torch.tanh` needs no replacement: with coreai-build 3600.83.1 it compiles to one Neural Engine region, and `2 * torch.sigmoid(2 * x) - 1` gives the same PSNR there (82.7 dB in `x * (one + torch.tanh(s))`).

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

**Cause**: `InferenceFunction.__call__` uses `**kwargs`, not a positional dict.
**Fix**: Use `await runner(**inputs)` with keyword arguments, not `runner(inputs_dict)`.

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
