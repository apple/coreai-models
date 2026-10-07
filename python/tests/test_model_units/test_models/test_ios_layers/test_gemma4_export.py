# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Gemma 4 export recipe (``models/gemma4/export.py``).

The recipe is not part of the installed package, so it is loaded by path. Its CLI
checks and metadata are tested directly; a tiny synthetic Gemma 4 is traced over a
non-power-of-two ladder, and exported as a one-bucket ladder through Core AI
conversion, saved, and loaded back, with the emitted functions checked against the
names the runner looks up.
"""

import argparse
import asyncio
import importlib.util
import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from coreai_models._constants import (  # noqa: E402
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
)
from coreai_models.models.ios.gemma4_text import Gemma4ForCausalLMForiOS  # noqa: E402
from tests._runner_infra._deps import _HAS_COREAI, _MSG_COREAI_NOT_FOUND  # noqa: E402
from tests.test_model_units.test_models.test_ios_layers.test_gemma4 import (  # noqa: E402
    Gemma4ForCausalLM,
    _build_quantized_ios_model,
    _make_config,
)

pytestmark = [
    pytest.mark.skipif(Gemma4ForCausalLM is None, reason="gemma4 requires transformers>=5.5"),
    pytest.mark.skipif(not _HAS_COREAI, reason=_MSG_COREAI_NOT_FOUND),
]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_spec = importlib.util.spec_from_file_location(
    "gemma4_export", _REPO_ROOT / "models/gemma4/export.py"
)
export_g4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_g4)


def test_ladder_converts_and_loads():
    """Every ladder function converts, and loads under the name and I/O the runner uses."""
    torch.manual_seed(0)
    cfg = _make_config()
    cfg.sliding_window = Gemma4ForCausalLMForiOS.SLIDING_WINDOW
    # Quantized embeddings, as shipped: the gather then lowers to the fused
    # dequant-gather, which only converts when traced dynamic in the query length.
    model = _build_quantized_ios_model(cfg, Gemma4ForCausalLM(cfg).state_dict()).half()
    ctx = 1024

    program = asyncio.run(export_g4._export_blocked_ladder(model, model.config, ctx))

    inputs = Gemma4ForCausalLMForiOS.export_input_names()
    states = Gemma4ForCausalLMForiOS.export_state_names()
    expected = {
        "load_embeddings": ((), ()),
        **{
            f"gather_embeddings_{q}": (inputs[GATHER_EMBEDDINGS_FUNCTION_NAME], ()) for q in (8, 64)
        },
        f"extend_{ctx}_8": (inputs[EXTEND_FUNCTION_NAME], states[EXTEND_FUNCTION_NAME]),
        f"prompt_opt_{ctx}_64": (inputs[EXTEND_FUNCTION_NAME], states[EXTEND_FUNCTION_NAME]),
    }

    async def load() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            asset = program.save_asset(Path(tmpdir) / "gemma4.aimodel")
            async with asset.executable() as aimodel:
                loaded = {}
                for name in aimodel.function_names:
                    desc = aimodel.load_function(name).desc
                    loaded[name] = (tuple(desc.input_names), tuple(desc.state_names))
                return loaded

    loaded = asyncio.run(load())
    assert {name: loaded.get(name) for name in expected} == expected
    # Anything else is a composite the lowering emits (the fused dequant-gather).
    assert all(name.startswith("fused_") for name in set(loaded) - set(expected)), sorted(loaded)


def _tiny_export_model():
    torch.manual_seed(0)
    cfg = _make_config()
    cfg.sliding_window = Gemma4ForCausalLMForiOS.SLIDING_WINDOW
    return _build_quantized_ios_model(cfg, Gemma4ForCausalLM(cfg).state_dict()).half()


def test_short_context_ladder_traces_every_rung():
    """A --max-context-length below the top shipping bucket becomes the ladder's last
    rung, and every rung traces."""
    model = _tiny_export_model()
    buckets = export_g4.context_ladder(2048)
    assert buckets == [1024, 2048]
    names = [name for name, *_ in export_g4._export_programs(model, model.config, buckets)]
    assert names == [
        "load_embeddings",
        "gather_embeddings",
        "extend_1024_8",
        "prompt_opt_1024_64",
        "extend_2048_8",
        "prompt_opt_2048_64",
    ]


def _cli_args(*argv: str) -> argparse.Namespace:
    return export_g4.build_parser().parse_args(["--model", "google/gemma-4-E2B-it", *argv])


@pytest.mark.parametrize(
    "argv, message",
    [
        (("--max-context-length", "64"), "must exceed the prefill query length"),
        (("--max-context-length", "2000"), "power of two"),
        (("--max-context-length", "0"), "power of two"),
        (("--max-context-length", "262144"), "supports at most"),
    ],
)
def test_cli_rejects_invalid_arguments(argv, message):
    with pytest.raises(SystemExit, match=message):
        export_g4._resolve_defaults(_cli_args(*argv))


def test_cli_accepts_small_and_default_contexts():
    for argv in ((), ("--max-context-length", "128"), ("--max-context-length", "32768")):
        export_g4._resolve_defaults(_cli_args(*argv))


def test_metadata_extras_require_what_the_runner_needs():
    cfg = _make_config()
    cfg.final_logit_softcapping = 30.0
    overrides = export_g4._ios_metadata_extras(cfg)["overrides"]
    assert overrides["sliding_window"] == cfg.sliding_window
    assert overrides["final_logit_softcapping"] == 30.0
    assert set(overrides["rope"]) == {
        "sliding_head_dim",
        "global_head_dim",
        "sliding_rope_theta",
        "global_rope_theta",
        "partial_rotary_factor",
    }
    del cfg.rope_parameters["full_attention"]["partial_rotary_factor"]
    with pytest.raises(ValueError, match="partial_rotary_factor"):
        export_g4._ios_metadata_extras(cfg)
