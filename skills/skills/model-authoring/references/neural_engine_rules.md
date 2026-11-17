# Neural Engine Rules

These rules apply when authoring PyTorch models compatible with the Neural Engine. They are organized by topic, from general principles to transformer-specific patterns.

## Neural Engine Programming Constraints

- **Max tensor rank: 5.** Rank-6+ intermediates are rejected. If rank > 5, reshape to remove unused dimensions (e.g., singleton dims of size 1) to bring rank to ≤ 5.
- **Supported dtypes**: fp16, int8, int16. fp32 falls back to GPU/CPU.
- **Fully static shapes**: Export one function per static shape config.

______________________________________________________________________

### Neural Engine residency

For best inference performance, keep the entire model on Neural Engine. Switching between accelerators (Neural Engine \<-> GPU \<-> CPU) introduces overhead that dominates small-model inference. If an op cannot run on Neural Engine, the compiler segments the graph and inserts transfers. Use the `working-with-coreai` skill to compile and check residency, then re-author those ops using Neural Engine-compatible alternatives.

______________________________________________________________________

## Data Layout & Alignment

### Tensor memory alignment

Neural Engine processes data in fixed-size blocks along the last tensor dimension (which Neural Engine treats as width). The last axis must be contiguous and aligned to 64 bytes for the model's inputs and outputs. When this dimension is not well-aligned, it gets padded — and the penalty is severe. A singleton last axis (dimension = 1) gets padded to 64 bytes, resulting in **32x memory cost at fp16** and **64x at int8**. Intermediate tensors that reside in L2 cache may tolerate smaller widths, but at authoring time you cannot predict which tensors will be L2-resident, so design for the worst case.

**Rules:**

- Use power-of-2 sizes for the last dimension whenever possible — they align well with Neural Engine processing granularities.
- Ensure the last dimension contains at least 32 FP16 elements (64 bytes).
- Never use the last axis as a singleton dimension — this is the worst case for padding waste.
- Reshape or transpose tensors to move larger dimensions to the last position.
- Design layer dimensions with this alignment in mind from the start.

```python
# BAD: last dimension is 7 — maps to width on Neural Engine, heavy padding overhead
bad_tensor = torch.randn(1, 16, 32, 224, 7)  # NDCHW layout

# GOOD: last dimension is 64 — well-aligned on Neural Engine
good_tensor = torch.randn(1, 16, 32, 224, 64)  # NDCHW layout
```

______________________________________________________________________

### BC1S format

The Neural Engine operates on tensors in `(Batch, Channels, 1, Sequence)` format. Matrix multiplications are implemented as 1x1 Conv2d.

```python
# Standard to Neural Engine: (B, S, D) → (B, D, 1, S)
x = x.permute(0, 2, 1).unsqueeze(2)

# Neural Engine to standard: (B, D, 1, S) → (B, S, D)
x = x.squeeze(2).permute(0, 2, 1)


# Multi-head GPU to Neural Engine: (B, H, S, D) → (B, H*D, 1, S)
def gpu_to_bc1s(x):
    B, H, S, D = x.shape
    return x.permute(0, 1, 3, 2).reshape(B, H * D, 1, S)


# Neural Engine to multi-head GPU: (B, H*D, 1, S) → (B, H, S, D)
def bc1s_to_gpu(x, n_heads, head_dim):
    B, _, _, S = x.shape
    return x.reshape(B, n_heads, head_dim, S).permute(0, 1, 3, 2)
```

______________________________________________________________________

### Avoid unnecessary memory copies

Unnecessary casts, reshapes, and transposes may introduce memory copies in the compiled graph. Reshapes and transposes that touch the width (innermost) dimension are especially expensive because they force a full data rewrite in memory. While the compiler cancels out some redundant operations during optimization, it does not catch all patterns. Minimize these operations at the source level — the fewer you have, the fewer survive to the compiled binary.

______________________________________________________________________

### Transpose bookkeeping around Conv2d

Conv2d expects data with channels in the right position. When your data flows in BC1S format, transpose into and out of Conv2d projection calls:

```python
# BC1S data → transpose for Conv2d → project → transpose back
x = x.transpose(-3, -1)
projected = self.proj(x)
projected = projected.transpose(-3, -1)
```

This transpose pair appears at every projection site. Keep it consistent — mismatched transposes are a common source of silent correctness bugs.

______________________________________________________________________

## Operations & Projections

### Conv2d instead of Linear

Neural Engine hardware natively accelerates Conv2d — `nn.Linear` gets decomposed into less efficient ops that may fall back to CPU. Using 1x1 Conv2d maps directly to the Neural Engine's convolution engine, keeping everything on-chip.

```python
# GPU: nn.Linear(in_features, out_features)
# Neural Engine: nn.Conv2d(in_features, out_features, kernel_size=1)
```

**State dict weight conversion** — when loading weights from a source model that uses `nn.Linear`, reshape for Conv2d:

```python
# Linear weight [O, I] → Conv2d weight [O, I, 1, 1]
conv.weight.data = linear.weight.unsqueeze(-1).unsqueeze(-1)

# Norm weight: (D,) → (1, D, 1, 1)
norm.weight.data = source_norm.weight.reshape(1, -1, 1, 1)
```

______________________________________________________________________

### Prefer high-level ops

The compiler maps high-level ops (e.g., `nn.LayerNorm`, `nn.RMSNorm`) more efficiently than their manually decomposed equivalents (reduce → multiply → add). Using the high-level op gives the compiler better visibility into intent and more optimization opportunities.

If you manually decompose an op and export it, the compiler may or may not reassemble it — do not rely on this. Use the highest-level PyTorch op available when an Neural Engine-supported lowering exists.

______________________________________________________________________

### Float32 intermediates

Any Python float literal or fp32 op creates an f32 buffer that Neural Engine cannot execute — it falls back to GPU/CPU:

```python
# BAD
x = hidden * (1.0 + scale)  # 1.0 is f32
h = torch.exp(self.conv(x))  # exp upcasts to f32

# GOOD
one = torch.ones(1, dtype=hidden.dtype, device=hidden.device)
x = hidden * (one + scale)
h = torch.exp(self.conv(x)).to(torch.float16)
```

______________________________________________________________________

### Softmax placement

Softmax on a spatial dimension (height, width) limits the compiler's ability to split work across spatial dimensions. This matters when input + output tensor size is large. If you need softmax and want efficient spatial processing, apply softmax on the **channel** dimension instead.

______________________________________________________________________

## Layer Design Guidelines

Architecture choices that improve energy efficiency on Apple silicon. These change the model structure, so **retraining is required**. The trade-off is between mathematical equivalence and computational efficiency — these recommendations prioritize Neural Engine efficiency while maintaining comparable model quality through retraining.

### Convolution stride optimization

Stride values that factor cleanly into 2s and 3s map efficiently to Neural Engine. Other values introduce overhead.

**Rules:**

- For equal strides > 2: use 4, 6, 8, 9, 12, 16, 24, or 32 (prime factors of 2 and 3 only).
- For mixed strides where one is 2: set the other to 3, 4, 8, or 9.
- Avoid unequal large strides — make them equal or set one to 2.
- With palettized kernels, stride support is very limited — use up to 2.
- Avoid large kernel sizes especially along the width dimension. Substitute with mathematically equivalent layers that use Neural Engine-compatible strides — use pixel shuffle or transpose insertion tricks if necessary.

```python
# BAD: stride 11 has prime factor 11; unequal large strides
self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=11)
self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=(7, 5))

# GOOD: prime factors of 2 and 3 only
self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=12)  # 12 = 2² x 3
self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=(8, 8))  # equal strides
self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=(2, 4))  # mixed
```

______________________________________________________________________

### Large kernel decomposition

Large convolution kernels are expensive on Neural Engine. Replace a single large kernel with consecutive smaller kernels that produce the same receptive field:

**Formula:** `k_fused = k1 + k2 - 1`

```python
# BAD: single 9x9 kernel
self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=9, padding=4)

# GOOD: two 5x5 kernels (5 + 5 - 1 = 9)
self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2)
```

______________________________________________________________________

### Convolution fusion

Consecutive small convolutions **without activation between them** can be fused into a single larger convolution. This reduces overhead at the cost of increased per-op computation:

```python
# BEFORE: two consecutive 3x3 convs, no activation between them
self.conv1 = nn.Conv2d(6, 7, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(7, 8, kernel_size=3, padding=1)

# AFTER: fused into one 5x5 conv (3 + 3 - 1 = 5)
self.conv_fused = nn.Conv2d(6, 8, kernel_size=5, padding=2)
```

Only fuse when there is no nonlinearity between the convolutions. Activation functions between convolutions break the linear algebra that makes fusion valid.

______________________________________________________________________

### Dilated convolution factorization

Large dilation rates are expensive. Factor them into chains of smaller dilation rates using prime factors of 2 and 3:

```python
# BAD: single dilation 8
self.conv = nn.Conv2d(ch, ch, 3, dilation=8, padding=8)

# GOOD: three dilation-2 convolutions (8 = 2 x 2 x 2)
self.conv1 = nn.Conv2d(ch, ch, 3, dilation=2, padding=2)
self.conv2 = nn.Conv2d(ch, ch, 3, dilation=2, padding=2)
self.conv3 = nn.Conv2d(ch, ch, 3, dilation=2, padding=2)

# Another example: dilation 6 = 2 x 3
self.conv_a = nn.Conv2d(ch, ch, 3, dilation=2, padding=2)
self.conv_b = nn.Conv2d(ch, ch, 3, dilation=3, padding=3)
```

______________________________________________________________________

### Pooling stride

Use stride 2 or 4 for pooling layers. Other stride values (3, 5, etc.) introduce overhead:

```python
# GOOD
pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
pool = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

# AVOID
pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
pool = nn.MaxPool2d(kernel_size=3, stride=5, padding=1)
```

______________________________________________________________________

## Model Compression

Neural Engine supports palettization (also known as clustering) natively. Compression reduces model size but does not always improve performance — the benefit depends on whether the layer is bottlenecked by weight loading rather than computation. Use profiling tools to identify which layers would benefit most from compression.

**Key points:**

- Palettization is the compression scheme to use on Neural Engine. Linear per-block integer weight quantization (for example the `4bit` macOS preset, `block_size=32`) has produced wrong output with no error when the model runs on the Neural Engine: on an iPhone 16 Pro Max an 8-bit per-block vision encoder scored 0.367 cosine against its fp32 reference on the Neural Engine and 0.999 on the GPU, while the same weights palettized to 8 bits scored 0.9975 (see [issue #279](https://github.com/apple/coreai-models/issues/279)). That report did not test per-channel linear quantization.
- A report on macOS found the same kind of error from a bare int4 per-block `Linear` under both GPU and Neural Engine preference, with an exact result only on the CPU, and found that casting around the matmul to fp32 removed it. Treat per-block linear quantization as unverified on any unit other than the CPU, and compare against an fp32 reference on the compute unit you ship before relying on it.
- Lookup tables can cover multiple output channels rather than one per kernel, which may improve accuracy since the model experiences less compression.
- Newer hardware generations support vector-valued lookup table entries rather than scalar values.
- Compression is most effective for layers where weight transfer time dominates computation time. Layers that are already compute-bound will not see performance gains from compression alone.
- Apply compression after authoring and verifying the float16 model, before final Core AI export.

______________________________________________________________________

## Transformer / LLM Patterns

These patterns apply specifically to transformer-based models (LLMs, ASR encoders, etc.).

### Embedding shape

Neural Engine embeddings use shape `(vocab_size, 1, hidden_size)` to maintain BC1S-compatible output:

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, 1, hidden_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]  # Returns (batch, 1, hidden_size)
```

When loading from a source model: `embedding_weight = source_weight.unsqueeze(1)  # (V, D) → (V, 1, D)`

______________________________________________________________________

### RMSNorm

Norm over `dim=1` (channels in BC1S). Weight shape `(1, D, 1, 1)`.

______________________________________________________________________

### MLP reshape for Conv2d

MLP layers reshape to fuse batch and sequence dimensions before Conv2d, then reshape back.

Match the source model's activation exactly:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, query_len = x.shape[0], x.shape[1]
    dim = x.shape[-1]

    x = x.reshape(batch_size * query_len, dim, 1, 1)  # Fuse for Conv2d
    up = self.up_proj(x)
    gate = self.gate_proj(x)
    gate = nn.functional.silu(gate)
    out = self.down_proj(up * gate)
    return out.reshape(batch_size, query_len, 1, dim)
```

If the source uses GELU with tanh approximation:

```python
gate = nn.functional.gelu(gate_pre, approximate="tanh")
```

The key principle: verify which activation the source model uses, then express it in a form that maps to Neural Engine-supported ops. When in doubt, check the op against the Core AI dialect reference.

______________________________________________________________________

### Float32 intermediates (in MLP context)

Any Python float literal or fp32 op creates an f32 buffer that Neural Engine cannot execute:

```python
# BAD
x = hidden * (1.0 + scale)  # 1.0 is f32
h = torch.exp(self.conv(x))  # exp upcasts to f32

# GOOD
one = torch.ones(1, dtype=hidden.dtype, device=hidden.device)
x = hidden * (one + scale)
h = torch.exp(self.conv(x)).to(torch.float16)
```

______________________________________________________________________

### Per-head attention

Neural Engine cannot fuse multi-head attention into a single operation. Split Q/K/V into explicit per-head tensors and compute each head individually. Beyond correctness, this chunking produces smaller intermediate tensors that are more likely to stay in L2 cache, improving both throughput and multicore utilization.

Use the `bchq,bkhc->bkhq` einsum pattern for the Q@K matmul — it maps directly to hardware without intermediate transpose or reshape operations. This avoids memory copies that would otherwise be triggered by reshaping the attention dimensions.

```python
queries = query.split(head_dim, dim=1)
keys = key.split(head_dim, dim=1)
values = value.split(head_dim, dim=1)

outputs = []
for h in range(n_heads):
    kv_idx = h // kv_group_size  # For GQA
    # bchq,bkhc->bkhq: no transpose/reshape needed
    attn = torch.einsum("bchq,bkhc->bkhq", queries[h], keys[kv_idx])
    attn = attn * scale
    attn = attn + mask
    attn = torch.softmax(attn, dim=-1)
    outputs.append(torch.einsum("bkhq,bkhc->bchq", attn, values[kv_idx]))
```

This is fundamental to Neural Engine hardware — there is no fused SDPA path.

______________________________________________________________________

### Causal mask

Neural Engine mask shape is `(1, key_seq, 1, query_seq)` — **transposed from GPU**.

```python
def create_ane_causal_mask(seq_len):
    key_idx = torch.arange(seq_len).unsqueeze(1)
    query_idx = torch.arange(seq_len).unsqueeze(0)
    mask = key_idx > query_idx
    mask = mask.float().masked_fill(mask, -40000.0)  # NOT float('-inf')
    return mask.unsqueeze(0).unsqueeze(2)  # (1, key_seq, 1, query_seq)
```

> **Why -40000.0**: Neural Engine hardware does not handle IEEE `-inf` correctly in softmax. `-40000.0` is representable in fp16 and drives `exp(-40000)` to zero.

Neural Engine also uses `K @ Q` (transposed from GPU's `Q @ K^T`) together with this transposed mask.

______________________________________________________________________

### RoPE

Precompute cos/sin outside the exported model; pass as `(1, head_dim, 1, S)` 4D inputs. Do not index a 2D table inside the graph with `position_ids` — `gather_nd` produces 3D output.

______________________________________________________________________

### KV cache — readonly pattern

Neural Engine KV cache shape: `[n_layers, B, H_kv*D, 1, max_S]`, sequence on **dim 4**. Functional I/O (not buffer).

The model contains **no KV writes**. Each call receives the full past cache, concatenates current K/V for attention, and returns new K/V tokens as outputs. Python updates the cache externally.

**Per-layer forward:**

```python
k_full = torch.cat([k_cache_layer, key_rope], dim=-1)
v_full = torch.cat([v_cache_layer, value], dim=-1)
# Attention uses k_full/v_full with causal mask
# Return key_rope and value as new_k/new_v outputs
```

> **CRITICAL: return `key_rope`, not raw `new_k`.** If you cache pre-RoPE K, the next call attends to stale non-RoPE-encoded keys → PSNR collapses to ~20 dB.

**Python cache update:**

```python
k_cache[layer_idx, :, :, :, t : t + S_q] = outputs["new_k"]
v_cache[layer_idx, :, :, :, t : t + S_q] = outputs["new_v"]
```

**Causal mask for readonly pattern (offset `t`, query len `S_q`):**

```python
past_key_idx = torch.arange(max_S).view(max_S, 1, 1)
query_j = torch.arange(S_q).view(1, 1, S_q)
past_mask = (past_key_idx >= t + query_j).float() * -1e4

new_key_j = torch.arange(S_q).view(S_q, 1, 1)
new_mask = (new_key_j > query_j).float() * -1e4

# Combined: (1, max_S+S_q, 1, S_q)
mask = torch.cat([past_mask, new_mask], dim=0).unsqueeze(0)
```

**Checklist:**

- [ ] No `mutable_slice_update` / cache writes inside the model
- [ ] `k_full = cat([k_cache, key_rope], dim=-1)`
- [ ] Model returns `key_rope`/`value` as `new_k`/`new_v` outputs
- [ ] Mask: strict `k < t+j` for past; causal `m <= j` for new
- [ ] Python writes `new_k`/`new_v` → cache slots `[t : t+S_q]`

______________________________________________________________________

### Model decomposition

Neural Engine models typically separate the embedding table from the transformer body. The embedding is exported separately because Neural Engine quantizes it independently and passes the table as an explicit input:

```python
class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        self.embed_tokens = Embedding(config)  # Exported separately
        self.extend = ModelExtend(config)  # Main export target


class ModelExtend(nn.Module):
    def __init__(self, config):
        self.model = TransformerModel(config)  # Transformer layers
        # LM head (tied or separate)
```

This decomposition enables separate embedding quantization and lookup programs.

______________________________________________________________________

### Chunked prefill

```python
CHUNK = 64
for chunk_start in range(0, prefill_len, CHUNK):
    S_q = min(CHUNK, prefill_len - chunk_start)
    mask = create_ane_causal_mask_readonly(S_q, max_seq_len, offset=chunk_start)
    # run model with chunk_embeds, mask, k_cache, v_cache
    k_cache[:, :, :, :, chunk_start : chunk_start + S_q] = new_k
    v_cache[:, :, :, :, chunk_start : chunk_start + S_q] = new_v
```

**Seam rule**: offset = `chunk_start`, not `chunk_end`.

**fp16 drift warning**: Sequential per-token prefill (S_q=1 per call) accumulates fp16 rounding errors across many steps x many layers. For prefill > ~50 tokens, use chunked prefill (S_q=64) or fp32 KV cache tensors in Python.

______________________________________________________________________

### Neural Engine functions

On Neural Engine, the compiled artifact exposes multiple static entrypoints per `(context_len, extend_num_tokens)` shape pair.

| Entrypoint | When used |
| ------------------------ | ---------------------------------------------- |
| `extend_{ctx}_{len}` | Token generation (returns logits + updated KV) |
| `prompt_opt_{ctx}_{len}` | Fast prefill (returns KV only, no logits) |
| `gather_embeddings_{N}` | Embedding lookup before each extend |

All functions compile from **one dynamic `torch.export`** via Core AI shape specialization.
