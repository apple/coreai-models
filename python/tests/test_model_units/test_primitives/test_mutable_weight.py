# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for mutable weights: the primitive, and export plus runtime round trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from coreai_models.primitives.mutable_weight import (
    MutableWeight,
    MutableWeightSpec,
    MutableWeightStore,
)
from tests._runner_infra._deps import _HAS_COREAI, _MSG_COREAI_NOT_FOUND

if _HAS_COREAI:
    from coreai_models.export.mutable_weight import (
        MutableWeightRunner,
        export_mutable_weight_module,
    )


class TestMutableWeight:
    def test_defaults_to_zeros_with_spec(self) -> None:
        weight = MutableWeight("w", (2, 3), dtype=torch.float16)
        assert torch.equal(weight.value(), torch.zeros(2, 3, dtype=torch.float16))
        assert weight.spec == MutableWeightSpec("w", (2, 3), torch.float16)

    def test_init_is_cast_to_dtype(self) -> None:
        weight = MutableWeight("w", (2,), dtype=torch.float16, init=torch.tensor([1.5, 2.5]))
        assert weight.value().dtype == torch.float16
        assert weight.value().tolist() == [1.5, 2.5]

    def test_init_with_wrong_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="init shape"):
            MutableWeight("w", (2, 3), init=torch.zeros(3, 2))

    def test_update_writes_in_place_and_read_keeps_the_value(self) -> None:
        weight = MutableWeight("w", (3,))
        buffer = weight.value()
        returned = weight.update(torch.tensor([1.0, 2.0, 3.0]))
        assert returned is buffer
        assert weight.read().tolist() == [1.0, 2.0, 3.0]


class TestMutableWeightStore:
    def test_names_keep_registration_order(self) -> None:
        store = MutableWeightStore([MutableWeight("b", (1,)), MutableWeight("a", (2,))])
        assert store.weight_names() == ("b", "a")
        assert [spec.name for spec in store.specs()] == ["b", "a"]
        assert "a" in store and "c" not in store

    def test_duplicate_name_is_refused(self) -> None:
        store = MutableWeightStore([MutableWeight("w", (1,))])
        with pytest.raises(ValueError, match="duplicate"):
            store.add(MutableWeight("w", (1,)))

    def test_snapshot_is_a_copy_and_load_restores_it(self) -> None:
        store = MutableWeightStore([MutableWeight("w", (2,), init=torch.tensor([1.0, 2.0]))])
        snapshot = store.snapshot()
        store["w"].update(torch.tensor([9.0, 9.0]))
        assert snapshot["w"].tolist() == [1.0, 2.0]
        store.load(snapshot)
        assert store["w"].value().tolist() == [1.0, 2.0]

    def test_load_with_an_unknown_name_is_refused(self) -> None:
        store = MutableWeightStore([MutableWeight("w", (1,))])
        with pytest.raises(KeyError):
            store.load({"other": torch.zeros(1)})


class _Accumulator(nn.Module):
    """Adds its input to a mutable weight each call and returns the new total."""

    def __init__(self) -> None:
        super().__init__()
        self.total = MutableWeight("total", (4,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.total.update(self.total.value() + x) * 1.0


class _Projection(nn.Module):
    """``x @ weight``, where the host sets ``weight`` between calls."""

    def __init__(self, read_with_value: bool = False) -> None:
        super().__init__()
        self.weight = MutableWeight("weight", (4, 3))
        self._read_with_value = read_with_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.value() if self._read_with_value else self.weight.read()
        return x @ weight


@pytest.mark.skipif(not _HAS_COREAI, reason=_MSG_COREAI_NOT_FOUND)
class TestMutableWeightRoundTrip:
    def _export(self, module: nn.Module, x: torch.Tensor, name: str, tmp_path: Path) -> Path:
        _, path = export_mutable_weight_module(
            module,
            (x,),
            input_names=["x"],
            output_names=["y"],
            weight_names=[name],
            output_path=tmp_path / "model.aimodel",
        )
        return path

    def test_a_weight_written_in_the_graph_persists_across_calls(self, tmp_path: Path) -> None:
        x = torch.ones(4)
        path = self._export(_Accumulator(), x, "total", tmp_path)
        with MutableWeightRunner(path, {"total": torch.zeros(4)}) as runner:
            first = runner.step({"x": x})["y"]
            second = runner.step({"x": x})["y"]
            np.testing.assert_allclose(first, np.ones(4))
            np.testing.assert_allclose(second, np.full(4, 2.0))
            np.testing.assert_allclose(runner.weights_snapshot()["total"], np.full(4, 2.0))

    def test_the_host_replaces_a_read_weight_between_calls(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        x, first_weight, second_weight = torch.randn(2, 4), torch.randn(4, 3), torch.randn(4, 3)
        path = self._export(_Projection(), x, "weight", tmp_path)
        with MutableWeightRunner(path, {"weight": first_weight}) as runner:
            np.testing.assert_allclose(
                runner.step({"x": x})["y"], (x @ first_weight).numpy(), rtol=1e-5, atol=1e-5
            )
            runner.set_weight("weight", second_weight)
            np.testing.assert_allclose(
                runner.step({"x": x})["y"], (x @ second_weight).numpy(), rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(runner.weights_snapshot()["weight"], second_weight.numpy())

    def test_setting_an_unknown_weight_is_refused(self, tmp_path: Path) -> None:
        x = torch.ones(4)
        path = self._export(_Accumulator(), x, "total", tmp_path)
        with (
            MutableWeightRunner(path, {"total": torch.zeros(4)}) as runner,
            pytest.raises(KeyError),
        ):
            runner.set_weight("other", torch.zeros(4))

    def test_a_weight_only_read_with_value_is_refused_at_export(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"MutableWeight\.read\(\)"):
            self._export(_Projection(read_with_value=True), torch.randn(2, 4), "weight", tmp_path)


class _Weighted(nn.Module):
    """``x @ weight`` for one ``(n, m)`` mutable weight of a given dtype."""

    def __init__(self, n: int, m: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = MutableWeight("weight", (n, m), dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.read()


class _ManyWeights(nn.Module):
    """Row ``i`` of the output is ``x * w_i``, so a weight bound to the wrong name shows."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.store = MutableWeightStore([MutableWeight(f"w{i:02d}", (4,)) for i in range(count)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x * weight.read() for weight in self.store.weights()])


_COMPUTE_UNITS = [
    "cpu",
    "gpu",
    pytest.param(
        "neural_engine",
        marks=pytest.mark.xfail(reason="mutable weights are not yet verified on the ANE"),
    ),
]


@pytest.mark.skipif(not _HAS_COREAI, reason=_MSG_COREAI_NOT_FOUND)
@pytest.mark.parametrize("compute_unit", _COMPUTE_UNITS)
class TestMutableWeightStress:
    """Round trips under load, on each preferred compute unit."""

    @staticmethod
    def _options(compute_unit: str):
        from coreai.runtime import ComputeUnitKind, SpecializationOptions

        return SpecializationOptions.from_preferred_compute_unit_kind(
            getattr(ComputeUnitKind, compute_unit)()
        )

    @staticmethod
    def _export(module: nn.Module, x: torch.Tensor, names: list[str], tmp_path: Path) -> Path:
        _, path = export_mutable_weight_module(
            module,
            (x,),
            input_names=["x"],
            output_names=["y"],
            weight_names=names,
            output_path=tmp_path / "model.aimodel",
        )
        return path

    def test_a_written_weight_persists_over_many_calls(
        self, compute_unit: str, tmp_path: Path
    ) -> None:
        x = torch.ones(4)
        path = self._export(_Accumulator(), x, ["total"], tmp_path)
        with MutableWeightRunner(
            path, {"total": torch.zeros(4)}, options=self._options(compute_unit)
        ) as runner:
            for step in range(1, 101):
                np.testing.assert_allclose(runner.step({"x": x})["y"], np.full(4, float(step)))
            np.testing.assert_allclose(runner.weights_snapshot()["total"], np.full(4, 100.0))

    def test_the_host_replaces_a_weight_on_every_call(
        self, compute_unit: str, tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 16)
        weights = [torch.randn(16, 8) for _ in range(20)]
        path = self._export(_Weighted(16, 8, torch.float32), x, ["weight"], tmp_path)
        with MutableWeightRunner(
            path, {"weight": weights[0]}, options=self._options(compute_unit)
        ) as runner:
            # Every value once, then back to the first: an earlier value is not cached.
            for weight in [*weights, weights[0]]:
                runner.set_weight("weight", weight)
                np.testing.assert_allclose(
                    runner.step({"x": x})["y"], (x @ weight).numpy(), rtol=1e-3, atol=1e-3
                )

    def test_many_weights_keep_their_names(self, compute_unit: str, tmp_path: Path) -> None:
        count = 24
        module = _ManyWeights(count)
        x = torch.arange(1.0, 5.0)
        path = self._export(module, x, list(module.store.weight_names()), tmp_path)
        names = module.store.weight_names()
        init = {name: torch.full((4,), float(i)) for i, name in enumerate(names)}
        with MutableWeightRunner(path, init, options=self._options(compute_unit)) as runner:
            expected = torch.stack([x * i for i in range(count)]).numpy()
            np.testing.assert_allclose(runner.step({"x": x})["y"], expected)

    def test_a_large_weight(self, compute_unit: str, tmp_path: Path) -> None:
        # 2048 x 4096 float32: 32 MiB.
        torch.manual_seed(0)
        x, weight = torch.randn(1, 2048), torch.randn(2048, 4096) / 64
        path = self._export(_Weighted(2048, 4096, torch.float32), x, ["weight"], tmp_path)
        with MutableWeightRunner(
            path, {"weight": weight}, options=self._options(compute_unit)
        ) as runner:
            np.testing.assert_allclose(
                runner.step({"x": x})["y"], (x @ weight).numpy(), rtol=1e-2, atol=1e-2
            )

    def test_a_float16_weight(self, compute_unit: str, tmp_path: Path) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 16, dtype=torch.float16)
        first, second = (torch.randn(16, 8, dtype=torch.float16) for _ in range(2))
        path = self._export(_Weighted(16, 8, torch.float16), x, ["weight"], tmp_path)
        with MutableWeightRunner(
            path, {"weight": first}, options=self._options(compute_unit)
        ) as runner:
            for weight in (first, second):
                runner.set_weight("weight", weight)
                expected = (x.float() @ weight.float()).numpy()
                np.testing.assert_allclose(
                    runner.step({"x": x})["y"].astype(np.float32), expected, rtol=1e-2, atol=1e-2
                )
