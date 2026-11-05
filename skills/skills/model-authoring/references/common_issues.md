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

## Neural Engine SiLU PSNR low (~47 dB) — `silu()` or `x * torch.sigmoid(x)`

**Cause**: On the Neural Engine these two forms return the same bits. For `|x| <= 8` they are up to 0.015 off, near `x = 0.5`; the GPU is at most 0.002 off. Conv → SiLU → conv with random fp16 weights reaches 47 dB against fp32 below. The GPU reaches 83 dB.
**Fix**: Write SiLU through `tanh`, with `c` an fp16 buffer holding 0.5: `x * (c * torch.tanh(x * c) + c)`. The same block reaches 71 dB and compiles to one Neural Engine region. `torch.sigmoid(x)` alone stays within 0.001.

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

## Neural Engine `x / v` wrong for `|v|` below 2^-16

**Cause**: For `|v|` below 2^-16 (about 1.5e-5), including 0, the Neural Engine returns `x * 65536` with the sign of `v`, not the quotient. With `x = v = 1e-6` it returns 0.0664; CPU and GPU return 1.0.
**Fix**: Floor the divisor at 6.2e-5 or above, for example `x / v.clamp(min=1e-4)`. `v.clamp(min=2e-5)` still returns 0.0664.

```python
import asyncio
import tempfile
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
from coreai_torch import TorchConverter, get_decomp_table


class Divide(torch.nn.Module):
    def __init__(self, floor=None):
        super().__init__()
        self.floor = floor

    def forward(self, x, v):
        return x / (v if self.floor is None else v.clamp(min=self.floor))


async def run(path, feeds, unit):
    options = SpecializationOptions.from_preferred_compute_unit_kind(unit)
    model = await AIModel.load(path, specialization_options=options)
    return (await model.load_function("main")(feeds))["y"].numpy()


v = np.tile(np.array([1e-6, 1e-5, 2e-5, 1e-3], np.float16), (1, 64, 1, 8))
x = np.full_like(v, 1e-6)
names = {"input_names": ["x", "v"], "output_names": ["y"]}
for floor in (None, 2e-5, 1e-4):
    inputs = (torch.from_numpy(x), torch.from_numpy(v))
    ep = torch.export.export(Divide(floor).eval().half(), inputs)
    ep = ep.run_decompositions(get_decomp_table())
    path = Path(tempfile.mkdtemp()) / "m.aimodel"
    TorchConverter().add_exported_program(ep, **names).to_coreai().save_asset(path)
    for unit in (ComputeUnitKind.gpu(), ComputeUnitKind.neural_engine()):
        y = asyncio.run(run(path, {"x": NDArray(x), "v": NDArray(v)}, unit))
        print(f"floor {floor}, {unit}:", y[0, 0, 0, :4])
```

______________________________________________________________________

## Neural Engine model compiles but runs on CPU

**Cause**: Model was compiled without Neural Engine preference (e.g., using default compute selection which routed to CPU/GPU).
**Fix**: Compile with `xcrun coreai-build compile model.aimodel --preferred-compute neural-engine`.

______________________________________________________________________

## Neural Engine compile exits 0 with no Neural Engine regions

**Cause**: `--preferred-compute neural-engine` is a preference. With an fp32 graph or a dynamic dimension, for example, `coreai-build compile` exits 0, prints no error, and writes a bundle with no Neural Engine region. A static Qwen3-0.6B export with linear INT4 weights also exits 0 with no region, with `ANECCompileOffline() failed` inside about 160 KB of warnings. `coreai-build inspect --compute` lists data types, not placement.
**Fix**: Export fp16 with static shapes. Then count regions in the bundle: `find model.h16c.aimodelc -name '*ANE_region_*.mlir.bc' | wc -l`. Zero means the bundle has no Neural Engine code.

```python
import subprocess
import tempfile
from pathlib import Path

import torch
from coreai_torch import TorchConverter, get_decomp_table
from torch import nn

names = {"input_names": ["x"], "output_names": ["y"]}


def compile_for_neural_engine(module, x, dynamic_shapes=None):
    ep = torch.export.export(module.eval(), (x,), dynamic_shapes=dynamic_shapes)
    ep = ep.run_decompositions(get_decomp_table())
    out = Path(tempfile.mkdtemp())
    converter = TorchConverter().add_exported_program(ep, **names)
    converter.to_coreai().save_asset(out / "m.aimodel")
    command = ["xcrun", "coreai-build", "compile", str(out / "m.aimodel")]
    command += ["--preferred-compute", "neural-engine", "--architecture", "h16c"]
    done = subprocess.run(command + ["--output", str(out)], capture_output=True)
    regions = len(list(out.rglob("*ANE_region_*.mlir.bc")))
    printed = (done.stdout + done.stderr).decode().strip()
    return f"exit {done.returncode}, printed {printed!r}, regions {regions}"


x = torch.zeros(1, 64, 1, 64)
dim = {3: torch.export.Dim("S", min=2, max=512)}
print("fp16:", compile_for_neural_engine(nn.Conv2d(64, 64, 1).half(), x.half()))
print("fp32:", compile_for_neural_engine(nn.Conv2d(64, 64, 1), x))
dynamic = compile_for_neural_engine(nn.Conv2d(64, 64, 1).half(), x.half(), (dim,))
print("fp16, dynamic width:", dynamic)
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
