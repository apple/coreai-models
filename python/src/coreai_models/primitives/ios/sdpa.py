# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPA(nn.Module):
    """iOS-optimized Scaled Dot-Product Attention.

    Unlike PyTorch's fused SDPA, iOS requires each attention head to be computed
    individually to meet hardware constraints and ensure efficient compilation.
    This implementation processes heads sequentially rather than in parallel.
    """

    def __init__(
        self,
        head_dim: int | None = None,
        scale: float | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        with torch.device("cpu"):
            if scale is None:
                self._scale_factor = nn.Buffer(torch.tensor(head_dim**-0.5), persistent=False)
            else:
                self._scale_factor = (
                    nn.Buffer(scale, persistent=False)
                    if isinstance(scale, torch.Tensor)
                    else nn.Buffer(torch.tensor(scale), persistent=False)
                )
        self._use_hf_impl = os.environ.get("USE_HF_IMPL", "").lower() == "true"

    # Efficient implementation equivalent to the following:
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention for iOS.

        Args:
            query: Query tensor with shape (batch_size, n_heads*head_dim, 1, seq_len)
            key: Key tensor with shape (batch_size, n_kv_heads*head_dim, 1, max_seq_len)
            value: Value tensor with shape (batch_size, n_kv_heads*head_dim, 1, max_seq_len)
            causal_mask: Causal attention mask with shape (1, max_seq_len, 1, seq_len)

        Returns:
            torch.Tensor: Attention output with shape (batch_size, n_heads*head_dim, 1, seq_len)
        """

        # use FlashAttention to avoid
        # materializing the full attention score matrix.
        # Trim K/V from max_pos to seq_len (cache positions beyond seq_len
        # are zeros masked by -inf) so we can use is_causal=True, which is
        # required for the FlashAttention kernel.
        if query.is_cuda and self._use_hf_impl:
            B, _, _, S = query.shape
            n_heads = query.shape[1] // self.head_dim
            n_kv_heads = key.shape[1] // self.head_dim

            # This path is currently prefill-only: it trims K/V to the first S positions
            # and relies on is_causal=True. In prefill every valid KV position
            # lies within [0, S)- a valid (unmasked, == 0) entry at KV index
            # >= S means this is an extend/decode call, which the trim and
            # is_causal=True below would silently mishandle.
            assert not (causal_mask[:, S:] == 0).any(), (
                "CUDA/HF SDPA path is prefill-only. Got a causal_mask with "
                "valid KV positions beyond query length S (extend/decode not "
                "supported)."
            )

            q = query.reshape(B, n_heads, self.head_dim, S).transpose(2, 3).contiguous()
            k = key[..., :S].reshape(B, n_kv_heads, self.head_dim, S).transpose(2, 3).contiguous()
            v = value[..., :S].reshape(B, n_kv_heads, self.head_dim, S).transpose(2, 3).contiguous()

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                # is_causal=True is required by the FlashAttention kernel we
                # target here, so causal_mask is intentionally not passed as
                # attn_mask. This path is only taken for prefill, where the
                # mask is guaranteed causal (asserted above), so is_causal=True
                # and the provided causal_mask are equivalent.
                is_causal=True,
                scale=self._scale_factor,
                enable_gqa=(n_kv_heads != n_heads),
            )

            return out.transpose(2, 3).reshape(B, n_heads * self.head_dim, 1, S)

        # Apply the scale factor before QK^T for numerical stability
        key = key.transpose(-3, -1) * self._scale_factor
        queries = query.split(self.head_dim, dim=1)
        keys = list(key.split(self.head_dim, dim=-1))

        n_heads = len(queries)

        # permute key heads in advance
        for kv_idx in range(len(keys)):
            keys[kv_idx] = keys[kv_idx].permute(0, 2, 3, 1)

        kv_group_size = len(queries) // len(keys)

        scores = []

        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            q = queries[head_idx].permute(0, 2, 3, 1)
            k = keys[kv_idx]
            attn_score = q @ k
            attn_score = attn_score.permute(0, 3, 1, 2)
            scores.append(attn_score)

        full_scores = torch.cat(scores, dim=2)
        masked_scores = full_scores + torch.cat([causal_mask] * n_heads, dim=2)
        full_scores = masked_scores.softmax(1)

        scores = full_scores.split(1, dim=2)

        values = list(value.split(self.head_dim, dim=1))

        # transpose values in advance
        for kv_idx in range(len(values)):
            values[kv_idx] = values[kv_idx].permute(0, 2, 3, 1).squeeze(1)

        weights = []
        for head_idx in range(n_heads):
            kv_idx = head_idx // kv_group_size
            s = scores[head_idx].permute(0, 2, 3, 1).squeeze(1)
            v = values[kv_idx]
            weight = (s @ v).unsqueeze(1)
            weight = weight.permute(0, 3, 1, 2)
            weights.append(weight)

        final_score = torch.cat(weights, dim=1)
        return final_score


#: Largest dimension the accelerator's transposes handle; BlockedSDPA keeps every
#: transposed key axis within it.
_MAX_TRANSPOSE_DIM = 65536

#: fp16-safe stand-in for -inf in the online-softmax running max.
_FP16_NEG_INF = -40000.0


class BlockedSDPA(nn.Module):
    """Blocked / flash global attention for large-context iOS."""

    def __init__(
        self,
        head_dim: int,
        scale: float | torch.Tensor | None = None,
        block_size: int = 8192,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size
        with torch.device("cpu"):
            if isinstance(scale, torch.Tensor):
                self._scale_factor = nn.Buffer(scale, persistent=False)
            else:
                if scale is None:
                    scale = head_dim**-0.5
                self._scale_factor = nn.Buffer(torch.tensor(scale), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Online-softmax attention over a flat cache slot (batch size 1).

        Args:
            query: ``(1, n_heads*head_dim, 1, q_len)`` (BC1S).
            key/value: ``(1, n_kv*head_dim, 1, ctx)`` — the flat global cache slot.
            causal_mask: ``(1, ctx, 1, q_len)`` — same flat mask as ``SDPA``.

        Returns:
            ``(1, n_heads*head_dim, 1, q_len)`` — identical layout to ``SDPA``.
        """
        head_dim, block_size = self.head_dim, self.block_size
        ctx = key.shape[-1]

        queries = query.split(head_dim, dim=1)  # each (1, head_dim, 1, q_len)
        n_heads = len(queries)
        n_kv = key.shape[1] // head_dim
        kv_group_size = n_heads // n_kv

        # Fold 1/sqrt(block_size) into the exp weights so no fp16 accumulator grows with the key
        # count (plain flash sums overflow fp16 past ~15k keys on the accelerator).
        inv_block = 1.0 / float(block_size) ** 0.5
        # Above the transpose limit, form p@v as (v @ pᵀ)ᵀ (transpose the small p, not the
        # value cache); the direct p@v is a faster kernel at/below it, so keep it there.
        hoist_safe = ctx > _MAX_TRANSPOSE_DIM
        # Slice K/V/mask into super-chunks no wider than the limit so the transpose the
        # compiler hoists for the scores matmul stays within it.
        xpose_span = max(block_size, (_MAX_TRANSPOSE_DIM // block_size) * block_size)

        # Pre-permute each Q head to (1, 1, q_len, head_dim) once (block-independent).
        qp = [q.permute(0, 2, 3, 1) for q in queries]

        outs = []
        for kv_idx in range(n_kv):
            c0 = head_dim * kv_idx  # this KV head's K/V channel base
            group = range(kv_idx * kv_group_size, (kv_idx + 1) * kv_group_size)
            # Fold the group's Q heads onto one matmul's row axis so the K/V block streams
            # through the matmul unit once for all of them.
            q_stack = torch.cat([qp[h] for h in group], dim=2)  # (1, 1, G·q_len, head_dim)
            rows = q_stack.shape[2]
            # den is the block-scaled denominator, o the output.
            m = torch.full((1, 1, rows, 1), _FP16_NEG_INF, dtype=query.dtype, device=query.device)
            den = torch.zeros((1, 1, rows, 1), dtype=query.dtype, device=query.device)
            o = torch.zeros((1, rows, head_dim), dtype=query.dtype, device=query.device)
            for c_lo in range(0, ctx, xpose_span):
                c_hi = min(c_lo + xpose_span, ctx)
                key_c = key[:, c0 : c0 + head_dim, :, c_lo:c_hi]
                val_c = value[:, c0 : c0 + head_dim, :, c_lo:c_hi]
                mask_c = causal_mask[:, c_lo:c_hi]
                for lo in range(0, c_hi - c_lo, block_size):
                    hi = min(lo + block_size, c_hi - c_lo)
                    # squeeze (not permute) to (1, head_dim, B): a reshape can't be hoisted
                    # into a transpose, so the matmul is rank-3 (1,rows,hd)@(1,hd,B).
                    k = (key_c[:, :, :, lo:hi] * self._scale_factor).squeeze(2)
                    vraw = val_c[:, :, :, lo:hi].squeeze(2)  # (1, head_dim, B)
                    mblock = mask_c[:, lo:hi].permute(0, 2, 3, 1)  # (1, 1, q_len, B)
                    mask = torch.cat([mblock] * kv_group_size, dim=2)  # (1, 1, rows, B)
                    s = (q_stack.squeeze(1) @ k).unsqueeze(1) + mask  # (1, 1, rows, B)
                    m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
                    corr = torch.exp(m - m_new)
                    p = torch.exp(s - m_new) * inv_block
                    prev = den * corr
                    den = prev + p.sum(dim=-1, keepdim=True)
                    if hoist_safe:
                        pv = (vraw @ p.squeeze(1).transpose(1, 2)).transpose(1, 2)
                    else:
                        pv = p.squeeze(1) @ vraw.transpose(1, 2)
                    o = (o * prev.squeeze(1) + pv) / den.squeeze(1)
                    m = m_new
            # Un-fold the G·q_len rows to per-head channels with tensor_split (plain slices;
            # unflatten/reshape overflow the compiler's instruction-reorder window).
            for chunk in torch.tensor_split(o, kv_group_size, dim=1):
                outs.append(chunk.transpose(1, 2).unsqueeze(2))

        return torch.cat(outs, dim=1)  # (1, n_heads·head_dim, 1, q_len)
