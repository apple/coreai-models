# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Mutable weights: tensors that persist on device and change after export.

A ``MutableWeight`` is a registered buffer that the module's ``forward`` updates in
place. Exported with ``coreai_models.export.mutable_weight``, the runtime holds each
mutable weight on device between calls: the graph can update it, and the host can seed,
read and replace it between calls without re-exporting the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import Self


@dataclass(frozen=True)
class MutableWeightSpec:
    """The name, shape and dtype of a :class:`MutableWeight`.

    ``name`` identifies the weight to the exporter and the runtime, so it must match
    between export and runtime.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32


class MutableWeight(nn.Module):
    """A tensor that persists on device and is updated in place across calls.

    Stored as a buffer rather than an ``nn.Parameter``, so ``torch.export`` captures
    the update as a buffer mutation instead of freezing the tensor as a constant.
    Inside the traced ``forward``, write it with :meth:`update`, or read it with
    :meth:`read` when the graph uses it without changing it.

    Args:
        name: The weight's name; see :class:`MutableWeightSpec`.
        shape: Tensor shape.
        dtype: Element dtype.
        init: Initial value, zeros when omitted. Its shape must equal ``shape``.

    Raises:
        ValueError: If ``init`` does not have shape ``shape``.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        init: Tensor | None = None,
    ) -> None:
        super().__init__()
        if init is None:
            init = torch.zeros(shape, dtype=dtype)
        elif tuple(init.shape) != tuple(shape):
            raise ValueError(
                f"MutableWeight {name!r}: init shape {tuple(init.shape)} != {tuple(shape)}"
            )
        self._name = name
        self._shape = tuple(shape)
        self._dtype = dtype
        self.register_buffer("_data", init.to(dtype))

    @property
    def name(self) -> str:
        """The weight's name."""
        return self._name

    @property
    def spec(self) -> MutableWeightSpec:
        """The :class:`MutableWeightSpec` of this weight."""
        return MutableWeightSpec(name=self._name, shape=self._shape, dtype=self._dtype)

    def value(self) -> Tensor:
        """The current tensor. Reading with this alone does not keep the weight on device."""
        return cast(Tensor, self._data)

    def read(self) -> Tensor:
        """The current tensor, kept on device so the host can replace it between calls.

        A weight stays on device only when the graph writes it; one that ``forward`` only
        reads is exported as a constant. This writes the weight back unchanged, so use it
        for weights that the host sets between calls, such as an adapter.
        """
        return self.update(self.value())

    def update(self, new_value: Tensor) -> Tensor:
        """Copy ``new_value`` into the weight in place and return the weight."""
        data = cast(Tensor, self._data)
        data.copy_(new_value)
        return data


class MutableWeightStore(nn.Module):
    """A named collection of :class:`MutableWeight` s, with host-side snapshots.

    Args:
        weights: Initial weights; names must be unique.

    Raises:
        ValueError: If two weights share a name.
    """

    def __init__(self, weights: list[MutableWeight] | None = None) -> None:
        super().__init__()
        self._weights = nn.ModuleDict()
        for weight in weights or []:
            self.add(weight)

    def add(self, weight: MutableWeight) -> MutableWeight:
        """Register ``weight`` and return it. Raises ``ValueError`` on a duplicate name."""
        if weight.name in self._weights:
            raise ValueError(f"duplicate MutableWeight name {weight.name!r}")
        self._weights[weight.name] = weight
        return weight

    def __getitem__(self, name: str) -> MutableWeight:
        return cast(MutableWeight, self._weights[name])

    def __contains__(self, name: object) -> bool:
        return name in self._weights

    def weights(self) -> list[MutableWeight]:
        """The weights, in registration order."""
        return [cast(MutableWeight, w) for w in self._weights.values()]

    def specs(self) -> list[MutableWeightSpec]:
        """The spec of each weight, in registration order."""
        return [w.spec for w in self.weights()]

    def weight_names(self) -> tuple[str, ...]:
        """The weights' names, in registration order, as the exporter's ``weight_names``."""
        return tuple(w.name for w in self.weights())

    def snapshot(self) -> dict[str, Tensor]:
        """A copy of every weight's current value, keyed by name."""
        return {w.name: w.value().detach().clone() for w in self.weights()}

    def load(self, snapshot: dict[str, Tensor]) -> Self:
        """Copy the values in ``snapshot`` into the weights of the same name.

        Raises:
            KeyError: If ``snapshot`` names a weight this store does not have.
        """
        for name, value in snapshot.items():
            self[name].update(value)
        return self
