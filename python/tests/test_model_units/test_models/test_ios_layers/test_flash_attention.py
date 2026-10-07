# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity for the blocked / flash global-attention path used by iOS Gemma4.

``BlockedSDPA`` (``primitives/ios/sdpa.py``) walks the key axis in blocks and keeps
an online-softmax recurrence instead of materializing the full score matrix. It is
pinned from two directions:

  * ``TestBlockedSDPA`` -- parity with the flat per-head ``SDPA`` (the flat-cache
    reference, which takes one big softmax), proving the online-softmax recurrence
    is algebraically identical. ``BlockedSDPA`` reads the *same* flat cache
    ``(1, C, 1, ctx)`` and flat mask ``(1, ctx, 1, q)`` as ``SDPA`` (no re-layout),
    so those are passed straight through -- across MQA/GQA/MHA head configs,
    ``n_blocks`` in {1, 2, 16}, ragged final blocks, and the ctx>65536 hoist-safe
    value matmul -- and every compute op in the traced graph must have rank <= 4
    (the on-device 4D compute-tensor limit) with a dynamic ``seq_len``.

  * ``TestBlockedSDPALongContext`` -- at a full 65536-key context, numerics against
    ``torch.scaled_dot_product_attention`` in both float32 and float16. This checks
    the cross-block rescale and the fp16 overflow guards (the ``inv_block`` weight
    scaling, the -40000 pseudo -inf) against an out-of-repo implementation rather
    than against another primitive in this package.
"""

import pytest
import torch
import torch.nn.functional as F

from coreai_models.primitives.ios.sdpa import SDPA, BlockedSDPA

DTYPE = torch.float32
NEG = float("-inf")
ATOL = 1e-5


def _build_causal_mask(
    ctx: int, q_len: int, start: int, dtype: torch.dtype = DTYPE
) -> torch.Tensor:
    """Flat causal mask in SDPA layout ``(1, ctx, 1, q_len)``.

    Query column ``i`` is at absolute position ``start + i`` and attends every
    key ``0 .. start+i`` (full causal). Matches ``test_gemma4._global_mask``.
    """
    m = torch.full((1, ctx, 1, q_len), NEG, dtype=dtype)
    for i in range(q_len):
        m[0, : start + i + 1, 0, i] = 0.0
    return m


def _make_inputs(n_heads, n_kv, head_dim, ctx, q_len, start, dtype: torch.dtype = DTYPE):
    query = torch.randn(1, n_heads * head_dim, 1, q_len, dtype=dtype)
    key = torch.randn(1, n_kv * head_dim, 1, ctx, dtype=dtype)
    value = torch.randn(1, n_kv * head_dim, 1, ctx, dtype=dtype)
    mask = _build_causal_mask(ctx, q_len, start, dtype)
    return query, key, value, mask


def _torch_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal_mask: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """``torch.scaled_dot_product_attention`` on the flat BC1S inputs ``SDPA`` takes.

    An independent reference: torch fuses the whole score matrix in one shot, so it
    shares no code (and no blocking) with the primitive under test. Converts the
    flat cache layout to torch's ``(B, n_heads, seq, head_dim)`` and the flat mask
    ``(1, ctx, 1, q_len)`` to an additive ``(1, 1, q_len, ctx)`` ``attn_mask``, then
    converts the output back.
    """
    q_len = query.shape[-1]
    ctx = key.shape[-1]
    n_heads = query.shape[1] // head_dim
    n_kv = key.shape[1] // head_dim

    q = query.reshape(1, n_heads, head_dim, q_len).transpose(2, 3)
    k = key.reshape(1, n_kv, head_dim, ctx).transpose(2, 3)
    v = value.reshape(1, n_kv, head_dim, ctx).transpose(2, 3)
    attn_mask = causal_mask.squeeze(2).transpose(1, 2).unsqueeze(1)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        scale=head_dim**-0.5,
        enable_gqa=(n_kv != n_heads),
    )
    return out.transpose(2, 3).reshape(1, n_heads * head_dim, 1, q_len)


def _max_op_rank(module, args, dynamic_shapes=None) -> int:
    """Export ``module`` (optionally with ``dynamic_shapes``) and return the max
    rank across every *compute* tensor (placeholders/outputs excluded).

    ``BlockedSDPA`` takes only flat rank-4 inputs, so even the placeholders are
    <=4D. Exporting with a dynamic ``seq_len`` additionally guards against a
    reshape / split that would emit a query-length guard and tie the graph to one
    ``q_len``.
    """
    from torch.export import export

    gm = export(module, args, dynamic_shapes=dynamic_shapes).graph_module
    max_rank = 0
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        val = node.meta.get("val")
        vals = val if isinstance(val, (list, tuple)) else [val]
        for v in vals:
            shape = getattr(v, "shape", None)
            if shape is not None:
                max_rank = max(max_rank, len(shape))
    return max_rank


# (n_heads, n_kv): MQA 8:1 is the real Gemma4 global config the head-stacking
# targets; GQA 4:2 and MHA 4:4 guard the grouping/unstacking logic.
_HEAD_CFGS = [(8, 1), (4, 2), (4, 4)]

# (block_size, ctx, start, label) -- n_blocks = ceil(ctx/block_size).
#   ctx==block_size              -> 1 block
#   ctx==2*block_size            -> 2 blocks
#   ctx==16*block_size           -> 16 blocks
#   ctx not a multiple of block  -> ragged final block
# start chosen to exercise both a mid-prompt chunk (future blocks fully masked, held
# at the fp16-safe -inf floor) and an end chunk (diagonal in the last block).
_SHAPE_CASES = [
    (16, 16, 8, "n_blocks=1 end"),
    (16, 16, 0, "n_blocks=1 start"),
    (16, 32, 24, "n_blocks=2 end"),
    (16, 32, 0, "n_blocks=2 start"),
    (16, 256, 248, "n_blocks=16 end"),
    (16, 256, 0, "n_blocks=16 start"),
    (16, 40, 24, "ragged ctx=40 (3 blks, last=8)"),
    (16, 100, 0, "ragged ctx=100 (7 blks, last=4)"),
    (8, 53, 30, "ragged ctx=53 (7 blks, last=5)"),
]


class TestBlockedSDPA:
    """The real ``BlockedSDPA`` primitive == flat ``SDPA`` + stays on-device (rank<=4)."""

    @pytest.mark.parametrize("n_heads,n_kv", _HEAD_CFGS, ids=["MQA8:1", "GQA4:2", "MHA4:4"])
    @pytest.mark.parametrize(
        "block_size,ctx,start,label",
        _SHAPE_CASES,
        ids=[c[3] for c in _SHAPE_CASES],
    )
    def test_parity_head_and_shape_configs(self, n_heads, n_kv, block_size, ctx, start, label):
        """Head-stacking + block walk matches flat SDPA across head configs and ragged tails."""
        torch.manual_seed(0)
        head_dim = 16
        q_len = 8
        query, key, value, mask = _make_inputs(n_heads, n_kv, head_dim, ctx, q_len, start)
        with torch.no_grad():
            ref = SDPA(head_dim=head_dim)(query, key, value, mask)
            got = BlockedSDPA(head_dim=head_dim, block_size=block_size)(query, key, value, mask)
        torch.testing.assert_close(got, ref, atol=ATOL, rtol=0.0)

    def test_parity_hoist_safe_value_matmul(self):
        """ctx > 65536 takes the (v @ p^T)^T value matmul; it must also match SDPA."""
        torch.manual_seed(0)
        head_dim = 8
        q_len = 8
        ctx = 32768 * 2 + 1  # 65537 > 65536 -> hoist_safe path (last block ragged)
        query, key, value, mask = _make_inputs(8, 1, head_dim, ctx, q_len, ctx - 8)
        with torch.no_grad():
            ref = SDPA(head_dim=head_dim)(query, key, value, mask)
            got = BlockedSDPA(head_dim=head_dim, block_size=32768)(query, key, value, mask)
        torch.testing.assert_close(got, ref, atol=ATOL, rtol=0.0)

    @pytest.mark.parametrize(
        "head_dim,ctx,block_size",
        [(64, 32, 16), (8, 65537, 32768)],
        ids=["common", "hoist_safe"],
    )
    def test_export_ranks_le_4_dynamic_seq_len(self, head_dim, ctx, block_size):
        """Every compute op stays rank <= 4 (iOS 4D limit) with a dynamic ``seq_len``.

        ctx is static; only ``seq_len`` is a ``Dim``. The export traces each query
        length statically, but keeping it dynamic here checks the primitive has no
        q_len-dependent reshape/split, alongside the 4D compute-tensor limit.
        """
        from torch.export import Dim

        torch.manual_seed(0)
        q_len = 8
        query, key, value, mask = _make_inputs(8, 1, head_dim, ctx, q_len, ctx - 8)
        blocked = BlockedSDPA(head_dim=head_dim, block_size=block_size)
        sl = Dim("seq_len", max=131072)
        ds = {"query": {3: sl}, "key": None, "value": None, "causal_mask": {3: sl}}
        max_rank = _max_op_rank(blocked, (query, key, value, mask), dynamic_shapes=ds)
        assert max_rank <= 4, f"max compute-op rank {max_rank} > 4 (busts iOS 4D limit)"


# Long-context tolerances, per dtype. Measured max abs error for the case below
# (ctx=65536, block_size=8192, head_dim=8, MQA 8:1): float32 2.2e-7, float16
# 9.2e-4. The tolerances leave ~46x and ~5x headroom respectively.
_LONG_CTX_ATOL = {torch.float32: 1e-5, torch.float16: 5e-3}


class TestBlockedSDPALongContext:
    """65536-key context, checked against ``torch.scaled_dot_product_attention``."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
    def test_parity_with_torch_sdpa_ctx_65536(self, dtype):
        """Full 65536-key context, 8 blocks of 8192, vs unblocked torch SDPA.

        ctx == 65536 sits just *under* the ``hoist_safe`` cutoff (``ctx > 65536``),
        so this is the widest context that still takes the direct ``p @ v`` matmul,
        and ``xpose_span`` collapses to a single 65536-wide super-chunk.

        float16 is the dtype the iOS export actually runs in, and it is where the
        block walk earns its keep: a plain flash denominator overflows fp16 past
        ~15k keys, so at 65536 keys the ``inv_block`` weight scaling has to hold.
        """
        torch.manual_seed(0)
        # MQA 8:1 -- the real Gemma4 global config, and the head config with the
        # largest measured error of the three (head-config coverage itself is in
        # TestBlockedSDPA).
        n_heads, n_kv = 8, 1
        head_dim = 8
        q_len = 8
        ctx = 65536
        block_size = 8192  # 8 whole blocks, the primitive's default
        # start == ctx - q_len: the diagonal lands in the final block, so every
        # query row attends across the entire 65536-key context.
        query, key, value, mask = _make_inputs(
            n_heads, n_kv, head_dim, ctx, q_len, ctx - q_len, dtype
        )
        with torch.no_grad():
            ref = _torch_sdpa(query, key, value, mask, head_dim)
            got = BlockedSDPA(head_dim=head_dim, block_size=block_size)(query, key, value, mask)

        assert got.dtype == dtype
        assert torch.isfinite(got).all(), "BlockedSDPA produced NaN/Inf at ctx=65536"
        torch.testing.assert_close(got, ref, atol=_LONG_CTX_ATOL[dtype], rtol=0.0)
