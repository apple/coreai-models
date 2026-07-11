# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import coreai_torch
import coreai_torch.composite_ops
import torch
from typing_extensions import Self


class SwitchLinear(torch.nn.Module):
    def __init__(
        self: Self,
        input_dims: int,
        output_dims: int,
        num_weight_sets: int,
        num_experts: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.gather_mm = coreai_torch.composite_ops.GatherMM(num_batch_axes=1)
        rand_weight = torch.rand(
            *(num_weight_sets, num_experts, output_dims, input_dims),
        )
        self.weight = torch.nn.Parameter(rand_weight)
        if bias:
            rand_bias = torch.rand(
                *(num_weight_sets, num_experts, output_dims),
            )
            self.bias = torch.nn.Parameter(rand_bias)

    def forward(
        self: Self,
        x: torch.Tensor,  # batch size mul query length x 1 x 1 x input dims
        indices: torch.IntTensor,  # batch size mul query length x num active experts
    ) -> torch.Tensor:
        # num weight sets x num experts x input dims x output dims
        weight_transpose = self.weight.transpose(-1, -2)
        # intermediate shapes
        #   x       : bsql x 1 x 1 x in_dims
        #   gathered : nws x bsql x nae x in_dims x out_dims
        # result shape
        #   y       : nws x bsql x nae x 1 x out_dims
        y = self.gather_mm(x, weight_transpose, rhs_indices=indices)
        if hasattr(self, "bias"):
            # num weight sets x batch size mul query length x num active experts x output dims
            active_experts_bias = coreai_torch.composite_ops._gather_mm._gather(
                self.bias, indices, num_batch_axes=1
            )
            # num weight sets x batch size mul query length x num active experts x 1 x output dims
            active_experts_bias = active_experts_bias.unsqueeze(-2)
            y = y + active_experts_bias
        return y


class SwiGLU(torch.nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()
        self._activate = torch.nn.SiLU()

    def forward(self: Self, up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        activated_gate = self._activate(gate)
        return activated_gate * up


class SwitchGLU(torch.nn.Module):
    def __init__(
        self: Self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        bias: bool = False,
        activation: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = SwitchLinear(hidden_size, moe_intermediate_size, 1, num_experts, bias=bias)
        self.up_proj = SwitchLinear(hidden_size, moe_intermediate_size, 1, num_experts, bias=bias)
        self.down_proj = SwitchLinear(moe_intermediate_size, hidden_size, 1, num_experts, bias=bias)
        self._activate = activation if activation is not None else SwiGLU()
        # Eager-only optimization. When set, tokens are processed in chunks of
        # this size to bound the peak GatherMM intermediate. Left None for
        # export/production so the traced
        # graph carries no data-dependent control flow on the token dimension.
        self.eager_chunk_size: int | None = None

    def forward(
        self: Self,
        x: torch.Tensor,  # batch size x query length x hidden size
        indices: torch.IntTensor,  # batch size x query length x num active experts
    ) -> torch.Tensor:
        batch_size, query_length, hidden_size = x.shape
        num_active_experts = indices.shape[-1]
        # batch size mul query length x 1 x 1 x hidden size
        x = x.reshape((-1, 1, 1, hidden_size))
        # batch size mul query length x num active experts
        indices = indices.reshape((-1, num_active_experts))
        bsql = x.shape[0]

        chunk_size = self.eager_chunk_size
        if chunk_size is None or bsql <= chunk_size:
            gate = self.gate_proj(x, indices)
            up = self.up_proj(x, indices)
            gated_up = self._activate(up, gate)
            # nws x bsql x nae x 1 x hidden size
            x = self.down_proj(gated_up, indices)
        else:
            chunks = []
            for start in range(0, bsql, chunk_size):
                x_c = x[start : start + chunk_size]
                idx_c = indices[start : start + chunk_size]
                gate_c = self.gate_proj(x_c, idx_c)
                up_c = self.up_proj(x_c, idx_c)
                gated_c = self._activate(up_c, gate_c)
                chunks.append(self.down_proj(gated_c, idx_c))
            # nws x bsql x nae x 1 x hidden size
            x = torch.cat(chunks, dim=1)

        # batch size x query length x num active experts x hidden size
        x = x.reshape((batch_size, query_length, num_active_experts, hidden_size))
        return x
