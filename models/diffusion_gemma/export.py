# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export DiffusionGemma-26B-A4B to Core AI.

Produces a bundle with .aimodel components plus tokenizer + metadata:

  encoder.aimodel  — autoregressive prompt encoder (fills the KV cache)
  decoder.aimodel  — bidirectional block-diffusion canvas denoiser
                     (self-conditioning is folded into this graph)

Usage:
    uv run models/diffusion_gemma/export.py \\
        --model google/diffusiongemma-26b-a4b-it \\
        --max-context-length 4096 --canvas-length 256 --output-dir ./exports/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch

from coreai_models.export.macos import export_macos_model, export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.pipeline import ExportConfig
from coreai_models.models.macos.diffusion_gemma import (
    load_diffusion_gemma_decoder,
    load_diffusion_gemma_encoder,
)
from coreai_models.models.macos.diffusion_gemma_config import (
    DiffusionGemmaConfig,
    DiffusionGemmaGenerationConfig,
)

logger = logging.getLogger("export_diffusion_gemma")

# Trace-time encoder context length used for the decoder's cross-attention KV.
_TRACE_ENC_CTX = 8


def _resolve_dtype(precision: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[precision]


def _rm(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)


def export_diffusion_gemma(
    hf_model_id: str,
    max_context_length: int = 4096,
    canvas_length: int = 256,
    compression: str = "none",
    output_dir: str = "./exports",
    output_name: str | None = None,
    num_layers: int | None = None,
    overwrite: bool = False,
    compute_precision: str = "bfloat16",
    encoder_only: bool = False,
    encoder_len: int = _TRACE_ENC_CTX,
    static_encoder: bool = False,
    include_debug_info: bool = False,
) -> str:
    """Export DiffusionGemma to a Core AI bundle."""
    dtype = _resolve_dtype(compute_precision)
    name = output_name or "diffusiongemma_26b_a4b"
    bundle = Path(output_dir) / name
    bundle.mkdir(parents=True, exist_ok=True)

    full_cfg = DiffusionGemmaConfig.from_pretrained(hf_model_id)
    gen_cfg = DiffusionGemmaGenerationConfig.from_pretrained(hf_model_id)
    text_cfg = full_cfg.text_config

    meta = build_aimodel_metadata(hf_model_id)

    # -- Encoder (autoregressive, fills the KV cache) ------------------------
    logger.info("Loading + exporting ENCODER ...")
    encoder = load_diffusion_gemma_encoder(
        hf_model_id,
        target_dtype=dtype,
        max_context_length=max_context_length,
        num_layers=num_layers,
    )
    if compression and compression != "none":
        encoder = _quantize_encoder(encoder, compression, dtype)

    n_layers_eff = encoder.config.num_hidden_layers
    cache_n_kv = encoder.num_key_value_heads
    cache_hd = encoder.head_dim
    enc_path = bundle / "encoder.aimodel"
    _rm(enc_path, overwrite)

    if static_encoder:
        # Static-shape prefill encoder (fixed sequence length = encoder_len). MPSGraph's
        # dynamic shape-function inference crashes in the Swift runtime on the dynamic
        # graph, so the Swift runner requires a static export. The cache is sized to
        # encoder_len and surfaced as keyCache/valueCache state.
        ids0 = torch.zeros(1, encoder_len, dtype=torch.int32)
        pos0 = torch.arange(encoder_len, dtype=torch.int32).unsqueeze(0)
        kc = torch.zeros(
            n_layers_eff, 1, cache_n_kv, encoder_len, cache_hd, dtype=dtype
        )
        vc = torch.zeros_like(kc)
        enc_prog = export_to_coreai(
            encoder,
            {"input_ids": ids0, "position_ids": pos0, "k_cache": kc, "v_cache": vc},
            dynamic_shapes=None,
            input_names=("input_ids", "position_ids"),
            output_names=("logits",),
            state_names=("keyCache", "valueCache"),
            include_debug_info=include_debug_info,
        )
    else:
        export_cfg = ExportConfig(
            hf_model_id=hf_model_id,
            max_context_length=max_context_length,
            compute_precision=compute_precision,
            compression=compression,
            include_debug_info=include_debug_info,
        )
        # Pass the encoder as `config` too: _build_reference_inputs reads the unified
        # cache dims (num_key_value_heads=8, head_dim=512, num_hidden_layers) off it.
        enc_prog = export_macos_model(encoder, encoder, export_cfg)
    enc_prog.save_asset(enc_path, meta)
    del encoder

    if encoder_only:
        _save_tokenizer(hf_model_id, bundle / "tokenizer")
        _write_bundle_metadata(
            bundle,
            name,
            hf_model_id,
            text_cfg,
            gen_cfg,
            full_cfg,
            max_context_length,
            canvas_length,
            compression,
            num_layers,
            encoder_only=True,
        )
        logger.info("Encoder-only bundle written to %s", bundle)
        return str(bundle)

    # -- Decoder (bidirectional canvas denoiser, self-conditioning folded in) -
    logger.info("Loading + exporting DECODER ...")
    decoder = load_diffusion_gemma_decoder(
        hf_model_id, target_dtype=dtype, num_layers=num_layers
    )
    h = text_cfg.hidden_size
    decoder_input_ids = torch.zeros(1, canvas_length, dtype=torch.int32)
    prev_soft_embeds = torch.zeros(1, canvas_length, h, dtype=dtype)
    pos = torch.arange(canvas_length, dtype=torch.int32).unsqueeze(0)
    enc_k = torch.zeros(n_layers_eff, 1, cache_n_kv, encoder_len, cache_hd, dtype=dtype)
    enc_v = torch.zeros_like(enc_k)
    temperature = torch.tensor([0.8], dtype=torch.float32)
    dec_inputs = {
        "decoder_input_ids": decoder_input_ids,
        "prev_soft_embeds": prev_soft_embeds,
        "position_ids": pos,
        "encoder_k": enc_k,
        "encoder_v": enc_v,
        "temperature": temperature,
    }

    if compression and compression != "none":
        decoder = _quantize_decoder(decoder, compression, dec_inputs)

    dec_prog = export_to_coreai(
        decoder,
        dec_inputs,
        output_names=("logits", "soft_embeds"),
        include_debug_info=include_debug_info,
    )
    dec_path = bundle / "decoder.aimodel"
    _rm(dec_path, overwrite)
    dec_prog.save_asset(dec_path, meta)
    del decoder

    _save_tokenizer(hf_model_id, bundle / "tokenizer")
    _write_bundle_metadata(
        bundle,
        name,
        hf_model_id,
        text_cfg,
        gen_cfg,
        full_cfg,
        max_context_length,
        canvas_length,
        compression,
        num_layers,
        encoder_only=False,
    )
    logger.info("Bundle written to %s", bundle)
    return str(bundle)


def _write_bundle_metadata(
    bundle,
    name,
    hf_model_id,
    text_cfg,
    gen_cfg,
    full_cfg,
    max_context_length,
    canvas_length,
    compression,
    num_layers,
    encoder_only,
) -> None:
    """Write the bundle-level metadata.json."""
    assets = {"encoder": "encoder.aimodel"}
    if not encoder_only:
        assets["decoder"] = "decoder.aimodel"
    bundle_meta = {
        "metadata_version": "0.2",
        "kind": "diffusion_llm",
        "name": name,
        "assets": assets,
        "language": {
            "tokenizer": "tokenizer",
            "vocab_size": text_cfg.vocab_size,
            "max_context_length": max_context_length,
            "embedded_tokenizer": True,
        },
        "diffusion": {
            "canvas_length": canvas_length,
            "max_denoising_steps": gen_cfg.max_denoising_steps,
            "entropy_bound": gen_cfg.entropy_bound,
            "confidence_threshold": gen_cfg.confidence_threshold,
            "stability_threshold": gen_cfg.stability_threshold,
            "t_max": gen_cfg.t_max,
            "t_min": gen_cfg.t_min,
            "mask_token_id": full_cfg.image_token_id,
        },
        "source": {
            "hf_model_id": hf_model_id,
            "model_definition": "torch",
            "compression": compression,
            "num_layers": num_layers,
            "encoder_only": encoder_only,
        },
    }
    with open(bundle / "metadata.json", "w") as f:
        json.dump(bundle_meta, f, indent=2)


def _quantize_encoder(encoder, compression, dtype):
    """Apply pre-export weight quantization (e.g. 4bit) to the encoder.

    The encoder's forward contract (input_ids, position_ids, k_cache, v_cache) is
    identical to a standard LLM, so the shared quantization preset applies.
    """
    from coreai_models._constants import (
        QUANT_TRACE_OFFSET,
        QUANT_TRACE_QUERY_LEN,
        TRACE_KV_CACHE_SEQ_LEN,
    )
    from coreai_models.export.compression import quantize_pytorch_model
    from coreai_models.export.presets import get_preset

    preset = get_preset(compression)
    quant_cfg = preset.get("torch_quantization_config")
    if quant_cfg is None:
        logger.warning(
            "Compression '%s' has no torch_quantization_config; skipping", compression
        )
        return encoder
    quant_cfg = dict(quant_cfg)

    vocab = encoder.config.vocab_size
    input_ids = torch.randint(1, vocab, (1, QUANT_TRACE_QUERY_LEN), dtype=torch.int32)
    position_ids = torch.arange(
        QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32
    ).unsqueeze(0)
    k_cache = torch.zeros(
        encoder.num_hidden_layers,
        1,
        encoder.num_key_value_heads,
        TRACE_KV_CACHE_SEQ_LEN,
        encoder.head_dim,
        dtype=dtype,
    )
    v_cache = torch.zeros_like(k_cache)

    dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("seq_ids", max=TRACE_KV_CACHE_SEQ_LEN - 2)},
        "position_ids": {
            1: torch.export.Dim(
                "seq_pos", min=QUANT_TRACE_QUERY_LEN, max=TRACE_KV_CACHE_SEQ_LEN - 1
            )
        },
        "k_cache": None,
        "v_cache": None,
    }

    logger.info("Applying %s weight quantization to encoder ...", compression)
    # The caches (k_cache, v_cache) are the last two inputs; cache_seq_len is the
    # traced KV-cache length. Only consumed if the preset calibrates activations
    # (weight-only 4bit does not), but passed for a correct calibration contract.
    return quantize_pytorch_model(
        encoder,
        (input_ids, position_ids, k_cache, v_cache),
        dynamic_shapes,
        quant_cfg,
        cache_seq_len=TRACE_KV_CACHE_SEQ_LEN,
        state_indices=(2, 3),
    )


def _quantize_decoder(decoder, compression, dec_inputs):
    """Apply pre-export weight quantization to the decoder.

    The decoder has a fixed-shape contract (canvas + fixed encoder-context length),
    so the quant trace uses static shapes (``dynamic_shapes=None``).
    """
    from coreai_models.export.compression import quantize_pytorch_model
    from coreai_models.export.presets import get_preset

    preset = get_preset(compression)
    quant_cfg = preset.get("torch_quantization_config")
    if quant_cfg is None:
        logger.warning(
            "Compression '%s' has no torch_quantization_config; skipping", compression
        )
        return decoder
    quant_cfg = dict(quant_cfg)

    example_inputs = (
        dec_inputs["decoder_input_ids"],
        dec_inputs["prev_soft_embeds"],
        dec_inputs["position_ids"],
        dec_inputs["encoder_k"],
        dec_inputs["encoder_v"],
        dec_inputs["temperature"],
    )
    logger.info("Applying %s weight quantization to decoder ...", compression)
    # The decoder is stateless per forward (encoder_k/encoder_v are read-only
    # cross-attention inputs, not mutable caches), so there are no state indices
    # and no KV-cache sequence length to bound calibration against.
    return quantize_pytorch_model(
        decoder,
        example_inputs,
        None,
        quant_cfg,
        cache_seq_len=0,
        state_indices=(),
    )


def _save_tokenizer(hf_model_id: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(hf_model_id)
        tok.save_pretrained(str(dest))
        logger.info("  tokenizer saved to %s", dest)
    except Exception as e:  # noqa: BLE001
        # Some checkpoints (list-valued eos_token_id) break AutoTokenizer.save;
        # fall back to copying the raw tokenizer files from the snapshot.
        logger.warning("  AutoTokenizer.save failed (%s); copying raw files", e)
        from huggingface_hub import snapshot_download

        snap = Path(
            snapshot_download(
                hf_model_id,
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                ],
            )
        )
        for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            src = snap / f
            if src.exists():
                shutil.copy(src, dest / f)
        logger.info("  tokenizer files copied to %s", dest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export.py",
        description="Export DiffusionGemma to Core AI .aimodel (encoder + decoder).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=4096,
        help="Encoder KV-cache max context",
    )
    parser.add_argument("--canvas-length", type=int, default=256, help="Canvas length")
    parser.add_argument(
        "--compression", default="none", help="Compression preset or 'none'"
    )
    parser.add_argument(
        "--compute-precision",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--output-dir", default="./exports")
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Truncate to N layers (smoke)"
    )
    parser.add_argument(
        "--encoder-len",
        type=int,
        default=_TRACE_ENC_CTX,
        help="Fixed encoder-context length baked into the decoder cross-attention (must match "
        "the prompt token length used at inference).",
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Export only the encoder (a complete autoregressive LM).",
    )
    parser.add_argument(
        "--static-encoder",
        action="store_true",
        help="Export a static-shape (fixed encoder-len) prefill encoder. Required for the Swift "
        "llm-runner, whose MPSGraph path does not support the dynamic-shape encoder.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help="Export in the converter's DEBUG mode (embeds full debug info). Defaults to "
        "RELEASE mode so shipped assets stay small.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    result = export_diffusion_gemma(
        hf_model_id=args.model,
        max_context_length=args.max_context_length,
        canvas_length=args.canvas_length,
        compression=args.compression,
        output_dir=args.output_dir,
        output_name=args.output_name,
        num_layers=args.num_layers,
        overwrite=args.overwrite,
        compute_precision=args.compute_precision,
        encoder_only=args.encoder_only,
        encoder_len=args.encoder_len,
        static_encoder=args.static_encoder,
        include_debug_info=args.include_debug_info,
    )
    print(f"Export complete: {result}")


if __name__ == "__main__":
    main()
