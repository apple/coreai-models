# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the DiffusionGemma export script.

The export script is co-located at ``models/diffusion_gemma/export.py`` (not part
of the installed package), so it is loaded by path. Pure helpers are tested
directly; the weight-loading / export paths are exercised with mocks so no 26B
checkpoint is required.
"""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import torch

# The export script is co-located under models/, loaded by path.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_spec = importlib.util.spec_from_file_location(
    "diffusion_gemma_export", _REPO_ROOT / "models/diffusion_gemma/export.py"
)
export_dg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_dg)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_resolve_dtype() -> None:
    assert export_dg._resolve_dtype("float16") is torch.float16
    assert export_dg._resolve_dtype("bfloat16") is torch.bfloat16
    assert export_dg._resolve_dtype("float32") is torch.float32


def test_rm_removes_existing_dir_only_when_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "asset"
        target.mkdir()
        export_dg._rm(target, overwrite=False)
        assert target.exists()  # not removed without overwrite
        export_dg._rm(target, overwrite=True)
        assert not target.exists()


def test_build_parser_parses_diffusion_args() -> None:
    args = export_dg._build_parser().parse_args(
        ["--model", "m", "--enc-len", "17", "--static-encoder", "--compression", "4bit"]
    )
    assert args.model == "m"
    assert args.enc_len == 17
    assert args.static_encoder is True
    assert args.compression == "4bit"


def test_write_bundle_metadata_contents() -> None:
    from coreai_models.models.macos.diffusion_gemma_config import (
        DiffusionGemmaConfig,
        DiffusionGemmaGenerationConfig,
    )

    full_cfg = DiffusionGemmaConfig()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        export_dg._write_bundle_metadata(
            bundle,
            "dg",
            "some/model",
            full_cfg.text_config,
            DiffusionGemmaGenerationConfig(),
            full_cfg,
            max_ctx=4096,
            canvas_length=32,
            compression="4bit",
            num_layers=None,
            encoder_only=False,
        )
        meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["kind"] == "diffusion_llm"
    assert meta["assets"] == {"encoder": "encoder.aimodel", "decoder": "decoder.aimodel"}
    assert meta["diffusion"]["canvas_length"] == 32
    assert meta["language"]["vocab_size"] == full_cfg.text_config.vocab_size


def test_write_bundle_metadata_encoder_only_omits_decoder() -> None:
    from coreai_models.models.macos.diffusion_gemma_config import (
        DiffusionGemmaConfig,
        DiffusionGemmaGenerationConfig,
    )

    full_cfg = DiffusionGemmaConfig()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        export_dg._write_bundle_metadata(
            bundle,
            "dg",
            "some/model",
            full_cfg.text_config,
            DiffusionGemmaGenerationConfig(),
            full_cfg,
            max_ctx=64,
            canvas_length=32,
            compression="none",
            num_layers=2,
            encoder_only=True,
        )
        meta = json.loads((bundle / "metadata.json").read_text())
    assert "decoder" not in meta["assets"]


# ---------------------------------------------------------------------------
# Export orchestration (mocked; no weights / Core AI compile)
# ---------------------------------------------------------------------------


def test_export_encoder_only_orchestration_mocked() -> None:
    from coreai_models.models.macos.diffusion_gemma import DiffusionGemmaEncoder
    from coreai_models.models.macos.diffusion_gemma_config import (
        DiffusionGemmaConfig,
        DiffusionGemmaGenerationConfig,
        DiffusionGemmaTextConfig,
    )

    tiny = DiffusionGemmaTextConfig(num_hidden_layers=2, vocab_size=100)
    encoder = DiffusionGemmaEncoder(tiny)
    prog = mock.Mock()  # stands in for the exported AIProgram

    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.object(export_dg, "load_diffusion_gemma_encoder", return_value=encoder),
        mock.patch.object(
            export_dg.DiffusionGemmaConfig,
            "from_pretrained",
            return_value=DiffusionGemmaConfig(text_config=tiny),
        ),
        mock.patch.object(
            export_dg.DiffusionGemmaGenerationConfig,
            "from_pretrained",
            return_value=DiffusionGemmaGenerationConfig(),
        ),
        mock.patch.object(export_dg, "export_macos_model", return_value=prog),
        mock.patch.object(export_dg, "build_aimodel_metadata", return_value={}),
        mock.patch.object(export_dg, "_save_tokenizer"),
    ):
        out = export_dg.export_diffusion_gemma(
            "some/model", output_dir=tmp, output_name="dg", encoder_only=True, num_layers=2
        )
        assert (Path(out) / "metadata.json").exists()
        prog.save_asset.assert_called_once()


def _tiny_text_config():
    from coreai_models.models.macos.diffusion_gemma_config import DiffusionGemmaTextConfig

    return DiffusionGemmaTextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        global_head_dim=16,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_experts=4,
        top_k_experts=2,
        vocab_size=40,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
    )


def test_export_full_orchestration_mocked() -> None:
    from coreai_models.models.macos.diffusion_gemma import (
        DiffusionGemmaDecoder,
        DiffusionGemmaEncoder,
    )
    from coreai_models.models.macos.diffusion_gemma_config import (
        DiffusionGemmaConfig,
        DiffusionGemmaGenerationConfig,
    )

    tiny = _tiny_text_config()
    enc = DiffusionGemmaEncoder(tiny)
    dec = DiffusionGemmaDecoder(tiny)
    prog = mock.Mock()
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.object(export_dg, "load_diffusion_gemma_encoder", return_value=enc),
        mock.patch.object(export_dg, "load_diffusion_gemma_decoder", return_value=dec),
        mock.patch.object(
            export_dg.DiffusionGemmaConfig,
            "from_pretrained",
            return_value=DiffusionGemmaConfig(text_config=tiny),
        ),
        mock.patch.object(
            export_dg.DiffusionGemmaGenerationConfig,
            "from_pretrained",
            return_value=DiffusionGemmaGenerationConfig(),
        ),
        mock.patch.object(export_dg, "export_macos_model", return_value=prog),
        mock.patch.object(export_dg, "export_to_coreai", return_value=prog),
        mock.patch.object(export_dg, "build_aimodel_metadata", return_value={}),
        mock.patch.object(export_dg, "_quantize_encoder", side_effect=lambda e, *a: e),
        mock.patch.object(export_dg, "_quantize_decoder", side_effect=lambda d, *a: d),
        mock.patch.object(export_dg, "_save_tokenizer"),
    ):
        out = export_dg.export_diffusion_gemma(
            "some/model",
            output_dir=tmp,
            output_name="dg",
            compression="4bit",
            canvas_length=8,
            enc_len=4,
            num_layers=2,
            static_encoder=True,
        )
        meta = json.loads((Path(out) / "metadata.json").read_text())
    assert meta["assets"] == {"encoder": "encoder.aimodel", "decoder": "decoder.aimodel"}


def test_export_save_tokenizer_fallback_copies_files() -> None:
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        (Path(src) / "tokenizer.json").write_text("{}")
        fake_transformers = types.ModuleType("transformers")

        class _AutoTok:
            @staticmethod
            def from_pretrained(_m):
                raise ValueError("list has no keys")  # trigger the fallback

        fake_transformers.AutoTokenizer = _AutoTok
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = lambda *a, **k: src
        with mock.patch.dict(
            sys.modules, {"transformers": fake_transformers, "huggingface_hub": fake_hub}
        ):
            export_dg._save_tokenizer("some/model", Path(dst))
        assert (Path(dst) / "tokenizer.json").exists()


def test_export_main_mocked() -> None:
    with (
        mock.patch.object(export_dg, "export_diffusion_gemma", return_value="/tmp/out") as ex,
        mock.patch.object(sys, "argv", ["e", "--model", "m", "--num-layers", "2"]),
    ):
        export_dg.main()
    ex.assert_called_once()


def test_export_quantize_encoder_decoder_mocked() -> None:
    import coreai_models.export.compression as compression
    import coreai_models.export.presets as presets
    from coreai_models.models.macos.diffusion_gemma import (
        DiffusionGemmaDecoder,
        DiffusionGemmaEncoder,
    )

    tiny = _tiny_text_config()
    enc = DiffusionGemmaEncoder(tiny)
    dec = DiffusionGemmaDecoder(tiny)
    canvas, enc_len, n_kv, hd = 8, 4, tiny.cache_num_key_value_heads, tiny.cache_head_dim
    dec_inputs = {
        "decoder_input_ids": torch.zeros(1, canvas, dtype=torch.int32),
        "prev_soft_embeds": torch.zeros(1, canvas, tiny.hidden_size),
        "position_ids": torch.arange(canvas, dtype=torch.int32).unsqueeze(0),
        "encoder_k": torch.zeros(tiny.num_hidden_layers, 1, n_kv, enc_len, hd),
        "encoder_v": torch.zeros(tiny.num_hidden_layers, 1, n_kv, enc_len, hd),
        "temperature": torch.tensor([0.8]),
    }
    with (
        mock.patch.object(presets, "get_preset", return_value={"torch_quantization_config": {}}),
        mock.patch.object(
            compression, "quantize_pytorch_model", side_effect=lambda m, *a, **k: m
        ) as quant,
    ):
        assert export_dg._quantize_encoder(enc, "4bit", torch.float32) is enc
        assert export_dg._quantize_decoder(dec, "4bit", dec_inputs) is dec

    # Both calls must pass the calibration-contract kwargs the quantizer now requires.
    enc_kwargs = quant.call_args_list[0].kwargs
    assert set(enc_kwargs) >= {"cache_seq_len", "state_indices"}
    assert enc_kwargs["state_indices"] == (2, 3)
    dec_kwargs = quant.call_args_list[1].kwargs
    assert dec_kwargs["state_indices"] == ()
    assert dec_kwargs["cache_seq_len"] == 0

    # Preset without a torch_quantization_config -> quantization is skipped.
    with mock.patch.object(presets, "get_preset", return_value={}):
        assert export_dg._quantize_encoder(enc, "weird", torch.float32) is enc
        assert export_dg._quantize_decoder(dec, "weird", dec_inputs) is dec


def test_export_save_tokenizer_success_path() -> None:
    with tempfile.TemporaryDirectory() as dst:
        saved = {}

        class _Tok:
            def save_pretrained(self, path):
                saved["path"] = path

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda _m: _Tok())
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            export_dg._save_tokenizer("some/model", Path(dst))
        assert saved["path"] == dst


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    for fn in _TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(_TESTS)} tests passed")
