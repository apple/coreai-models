# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma 4 (Gemma 3n / E2B) vision-language exporter.

Mirrors the Qwen3-VL VLM exporter (dict-based ``VLMSpec``/``SUPPORTED_MODELS``
in :mod:`coreai_models.vlm.export`) but adapts to Gemma 4's architecture, which
diverges from Qwen3-VL in three load-bearing ways:

  1. **Dual KV cache + PLE.** Gemma 4's decoder has a bounded sliding-window
     cache *and* a growable full-attention cache (4 state tensors, named
     ``slidingKeyCache``/``slidingValueCache``/``keyCache``/``valueCache``), plus
     an externalized INT8 Per-Layer-Embeddings (``ple_embeddings``) graph input
     and a separate ``<name>_ple.safetensors`` sidecar. Qwen3-VL has a single
     2-state cache and no PLE. The ``main`` graph here therefore takes
     ``(inputs_embeds, position_ids, ple_embeddings)`` + 4 states.

  2. **Native ViT vision tower.** Gemma 4 (transformers==5.10.2) uses a
     ``Gemma4VisionModel`` (patch embedder + transformer encoder + 2-D average
     pooler) followed by a soft-token multimodal embedder -- not Qwen3-VL's
     ViT+patchify nor an earlier MobileNet-v5. Vision produces a
     ``[1, vision_soft_tokens_per_image, text_hidden]`` soft-token block that the
     host scatters into ``inputs_embeds`` at the image-placeholder positions.

  3. **Scaled embeddings.** ``embed.aimodel`` bakes in Gemma's
     ``hidden_size**0.5`` embed scale (matching HF ``Gemma3nTextScaledWordEmbedding``)
     so merged text/image embeddings live at the right scale, mirroring HF
     ``masked_scatter``.

The three-asset bundle layout, the host-side scatter-merge of image soft tokens,
and the ``embed.aimodel`` / ``vision.aimodel`` split are all identical in spirit
to Qwen3-VL.

Bundle (``<name>.llmasset/``):
  - ``<name>.aimodel``          decoder, role ``main`` (inputs_embeds, dual KV, PLE)
  - ``embed.aimodel``           token-embedding lookup (role ``embedding``)
  - ``vision.aimodel``          Gemma4VisionModel + soft-token embedder (role ``vision``)
  - ``<name>_ple.safetensors``  externalized INT8 PLE sidecar
  - ``tokenizer/``              embedded HF tokenizer
  - ``metadata.json``           bundle manifest (``kind=vlm``)
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    KEY_CACHE_SLIDING_NAME,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
    VALUE_CACHE_SLIDING_NAME,
)
from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.models.macos.gemma4_text import Gemma4ForCausalLMEmbeddings
from coreai_models.models.macos.gemma4_vision import build_vision_encoder

logger = logging.getLogger(__name__)

# Dual-cache state names, in the exact order of
# Gemma4ForCausalLMEmbeddings.forward's state args (k_cache, v_cache,
# k_cache_full, v_cache_full). Matches Gemma4ForCausalLM._state_names.
DUAL_CACHE_STATE_NAMES = (
    KEY_CACHE_SLIDING_NAME,
    VALUE_CACHE_SLIDING_NAME,
    KEY_CACHE_NAME,
    VALUE_CACHE_NAME,
)


@dataclass(frozen=True)
class Gemma4VLMSpec:
    """Per-model export recipe for a Gemma 4 vision-language checkpoint.

    Carries the HF id, output bundle name, and the vision geometry
    (resolution, soft-token count, image placeholder token, and SigLIP/ImageNet
    normalization stats) that drive the vision-encoder export and the ``vision``
    block of ``metadata.json``.
    """

    short_name: str
    hf_model_id: str
    output_name: str
    image_token_id: int
    image_size: int
    vision_soft_tokens_per_image: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float

    @property
    def num_visual_tokens(self) -> int:
        return self.vision_soft_tokens_per_image


# Gemma 4 E2B defaults. image_token_id and the image geometry come from the HF
# Gemma4Config / image processor. Gemma 4 preprocessing is just rescale-to-[0,1]
# + resize (image_mean=[0,0,0], image_std=[1,1,1], do_normalize=False); the patch
# embedder itself does the 2*(x-0.5) centering. 768x768 -> 48x48 patches ->
# 256 soft tokens (pooling_kernel_size=3).
GEMMA4_E2B = Gemma4VLMSpec(
    short_name="gemma4-e2b",
    hf_model_id="google/gemma-4-E2B-it",
    output_name="gemma_4_e2b_it_vlm",
    image_token_id=258880,  # <|image|> (real E2B tokenizer id; in-range for embed + PLE)
    image_size=768,
    vision_soft_tokens_per_image=256,
    image_mean=(0.0, 0.0, 0.0),
    image_std=(1.0, 1.0, 1.0),
    rescale_factor=1.0 / 255.0,
)


# ---------------------------------------------------------------------------
# safetensors helpers
# ---------------------------------------------------------------------------


def _get_safetensors_files(model_dir: str) -> list[str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        return [os.path.join(model_dir, s) for s in shards]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError(f"No safetensors in {model_dir}")


def _text_prefix() -> str:
    """State-dict prefix under which the Gemma 4 unified checkpoint stores text weights."""
    return "model.language_model."


def load_text_decoder_from_safetensors(
    hf_config,
    model_dir: str,
    max_ctx: int,
    num_layers: int | None,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the Gemma 4 text decoder (embeddings variant) directly from safetensors.

    Keeps only ``model.language_model.*`` (drops ``model.vision_tower.*`` /
    ``model.audio_tower.*`` / ``model.embed_vision.*`` / ``model.embed_audio.*``),
    strips the prefix to ``model.*``, and runs the model's own state-dict
    mutations (q/k/v fusion, qk_norm fusion, MoE reshaping, PLE pop). Bypasses
    ``from_hf_memory_efficient`` for the same reason Qwen3-VL does.
    """
    text_cfg = Gemma4ForCausalLMEmbeddings._get_reauthored_config(
        getattr(hf_config, "text_config", hf_config), max_ctx, num_layers
    )
    model = Gemma4ForCausalLMEmbeddings(text_cfg, model_device="meta")
    model.to(dtype=dtype)

    prefix = _text_prefix()
    state_dict: dict[str, torch.Tensor] = {}
    for path in _get_safetensors_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safe_open has no __iter__
                if not key.startswith(prefix):
                    continue  # skip vision_tower / audio_tower / embed_vision / embed_audio
                stripped = key[len(prefix) :]  # "layers.0.self_attn.q_proj.weight"
                model_key = "model." + stripped
                tensor = f.get_tensor(key)
                if tensor.dtype not in (torch.float16, torch.int8) and "zero_point" not in key:
                    tensor = tensor.to(dtype)
                state_dict[model_key] = tensor

    model._mutate_state_dict(state_dict)
    model.load_state_dict(state_dict, assign=True, strict=False)

    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"Parameters not loaded: {meta}")
    return model.eval()


# ---------------------------------------------------------------------------
# embed.aimodel: token-embedding lookup (embed_scale baked in)
# ---------------------------------------------------------------------------


class ScaledEmbedTokens(nn.Module):
    """Token-embedding lookup that bakes in Gemma's ``hidden_size**0.5`` scale.

    HF ``Gemma3nTextScaledWordEmbedding`` multiplies the gathered rows by
    ``embed_scale``; the exported decoder (``Gemma4TextModelEmbeddings``)
    expects ``inputs_embeds`` to already carry that factor, so it lives here.

    Uses fancy indexing (``weight[input_ids]``) rather than ``nn.Embedding`` so
    the gather lowers with Int32 indices, as the runtime feeds Int32.

    Input:  input_ids  int32 [1, seq]
    Output: embeddings f16   [1, seq, hidden]
    """

    def __init__(self, weight: torch.Tensor, embed_scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (self.weight[input_ids] * self.embed_scale).to(torch.float16)


def _load_embed_weight(model_dir: str) -> torch.Tensor:
    embed_key = _text_prefix() + "embed_tokens.weight"
    for path in _get_safetensors_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            if embed_key in f.keys():  # noqa: SIM118
                return f.get_tensor(embed_key).to(torch.float16)
    raise RuntimeError(f"embed_tokens not found in safetensors (looked for '{embed_key}')")


async def export_embed_model(
    spec: Gemma4VLMSpec,
    bundle_path: Path,
    model_dir: str,
    hidden_size: int,
    max_ctx: int,
    overwrite: bool,
) -> str:
    """Export the (scaled) token-embedding lookup as embed.aimodel."""
    weight = _load_embed_weight(model_dir)
    vocab_size, hidden = weight.shape
    embed_scale = float(hidden_size) ** 0.5
    module = ScaledEmbedTokens(weight, embed_scale).eval()

    input_ids = torch.zeros(1, 64, dtype=torch.int32)
    program = export_to_coreai(
        module,
        {"input_ids": input_ids},
        dynamic_shapes={"input_ids": {1: torch.export.Dim("embed_seq", max=max_ctx - 1)}},
        input_names=("input_ids",),
        output_names=("embeddings",),
        state_names=None,
    )
    program.optimize()

    embed_path = bundle_path / "embed.aimodel"
    if embed_path.exists():
        if not overwrite:
            raise FileExistsError(f"{embed_path} exists. Use --overwrite.")
        shutil.rmtree(embed_path)
    meta = build_aimodel_metadata(spec.hf_model_id, component="embedding")
    await asyncio.to_thread(program.save_asset, embed_path, meta)
    logger.info(f"Saved embed.aimodel: {vocab_size} x {hidden} x f16 (scale={embed_scale:.3f})")
    return "embed.aimodel"


# ---------------------------------------------------------------------------
# Text decoder bundle (decoder + embed + PLE sidecar + tokenizer + metadata)
# ---------------------------------------------------------------------------


def _build_decoder_reference_inputs(
    model: nn.Module, text_cfg, hidden_size: int, max_ctx: int, num_layers: int | None
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build reference inputs + dynamic shapes for the inputs_embeds dual-cache decoder.

    Mirrors ``export.macos._build_reference_inputs`` but with ``inputs_embeds``
    in place of ``input_ids`` (VLM decoder graph contract).
    """
    QUERY_LEN = 16
    OFFSET = 8
    n_layers = num_layers or text_cfg.num_hidden_layers

    # inputs_embeds is a host-fed graph input kept in f16 (matches embed.aimodel
    # / vision.aimodel outputs and the Swift Float16 path); the decoder casts it
    # to bf16 internally. The KV caches are internal bf16 states, matching
    # Gemma 4's bf16 compute (see Gemma4ForCausalLM / the text-only recipe).
    embed_dtype = torch.float16
    compute_dtype = torch.bfloat16
    inputs_embeds = torch.randn(1, QUERY_LEN, hidden_size, dtype=embed_dtype)
    position_ids = torch.arange(QUERY_LEN + OFFSET, dtype=torch.int32).unsqueeze(0)

    sliding_window = getattr(text_cfg, "sliding_window", None) or 512
    k_cache, v_cache = model._build_sliding_kv_cache_tensors(
        text_cfg, sliding_window, compute_dtype
    )
    k_full, v_full = model._build_full_kv_cache_tensors(
        text_cfg, TRACE_KV_CACHE_SEQ_LEN, compute_dtype
    )

    ref: dict[str, torch.Tensor] = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_cache_full": k_full,
        "v_cache_full": v_full,
    }

    # Bounded sliding-window chunk must fit the window (see update_and_fetch_windowed).
    seq_ids_max = min(max_ctx - 2, sliding_window)
    query_dim = torch.export.Dim("query_len", max=seq_ids_max)
    dynamic: dict = {
        "inputs_embeds": {1: query_dim},
        "position_ids": {1: torch.export.Dim("seq_pos", min=QUERY_LEN, max=max_ctx - 1)},
        # Sliding cache is preallocated at its final size -> static seq dim.
        "k_cache": None,
        "v_cache": None,
        # Full-attention cache is preallocated and updated in place; its seq
        # dimension ranges from the trace length up to the full context.
        "k_cache_full": {
            3: torch.export.Dim("k_full_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)
        },
        "v_cache_full": {
            3: torch.export.Dim("v_full_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)
        },
    }

    ple_dim = getattr(text_cfg, "hidden_size_per_layer_input", 0)
    if ple_dim and hasattr(model, "_ple_weight"):
        ref["ple_embeddings"] = torch.randint(
            -128, 127, (1, QUERY_LEN, n_layers * ple_dim), dtype=torch.int8
        )
        dynamic["ple_embeddings"] = {1: query_dim}

    return ref, dynamic


async def export_text_bundle(
    spec: Gemma4VLMSpec,
    *,
    max_ctx: int,
    num_layers: int | None,
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, str, int]:
    """Download weights and write the text portion of the Gemma 4 VLM bundle.

    Returns ``(bundle_path, model_dir, hidden_size)`` so the vision export can
    reuse the download and the metadata patch has the hidden size.
    """
    output_name = spec.output_name

    logger.info(f"Downloading {spec.hf_model_id}...")
    model_dir = snapshot_download(
        spec.hf_model_id,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "special_tokens_map.json",
            "generation_config.json",
        ],
    )
    raw_cfg = AutoConfig.from_pretrained(model_dir)
    text_cfg = raw_cfg.text_config
    hidden_size = text_cfg.hidden_size
    vocab_size = text_cfg.vocab_size
    logger.info(f"Text config: hidden={hidden_size}, vocab={vocab_size}, ctx={max_ctx}")

    logger.info("Loading Gemma4 text decoder (embeddings variant) from safetensors...")
    model = load_text_decoder_from_safetensors(
        raw_cfg, model_dir, max_ctx, num_layers, torch.bfloat16
    )
    # _get_reauthored_config mutated raw_cfg.text_config in place with the
    # overrides; re-read it so the reference inputs use the effective config.
    eff_cfg = model.config

    ref, dynamic = _build_decoder_reference_inputs(model, eff_cfg, hidden_size, max_ctx, num_layers)
    input_names: tuple[str, ...] = ("inputs_embeds", "position_ids")
    if "ple_embeddings" in ref:
        input_names = (*input_names, "ple_embeddings")

    logger.info("Exporting Gemma4 VLM text decoder to Core AI (dual-cache, PLE, inputs_embeds)...")
    program = export_to_coreai(
        model,
        ref,
        dynamic_shapes=dynamic,
        input_names=input_names,
        output_names=("logits",),
        state_names=DUAL_CACHE_STATE_NAMES,
    )
    program.optimize()

    bundle_path = output_dir / (output_name + ".llmasset")
    bundle_path.mkdir(parents=True, exist_ok=True)
    aimodel_path = bundle_path / f"{output_name}.aimodel"
    if aimodel_path.exists():
        if not overwrite:
            raise FileExistsError(f"{aimodel_path} exists. Use --overwrite.")
        shutil.rmtree(aimodel_path)

    logger.info(f"Saving decoder to {aimodel_path}...")
    meta = build_aimodel_metadata(spec.hf_model_id, component="main")
    await asyncio.to_thread(program.save_asset, aimodel_path, meta)

    # PLE sidecar (INT8) -- mmapped by the Swift runtime, gathered per token.
    if hasattr(model, "dump_ple_embedding") and hasattr(model, "_ple_weight"):
        logger.info("Dumping Per-Layer Embeddings (PLE) artifact...")
        ple_path = model.dump_ple_embedding(str(bundle_path), output_name)
        logger.info(f"Wrote PLE artifact to {ple_path}")

    del model

    logger.info("Exporting embed.aimodel (scaled)...")
    embed_rel = await export_embed_model(
        spec, bundle_path, model_dir, hidden_size, max_ctx, overwrite
    )

    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(str(bundle_path / "tokenizer"))

    # metadata.json (vision asset patched in later by export_vision_encoder).
    metadata = {
        "metadata_version": "0.2",
        "kind": "vlm",
        "name": output_name,
        "assets": {
            "main": f"{output_name}.aimodel",
            "embedding": embed_rel,
        },
        "language": {
            "tokenizer": spec.hf_model_id,
            "vocab_size": vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": {
            "architecture": "gemma4",
            "image_size": spec.image_size,
            "image_token_count": spec.num_visual_tokens,
            "image_token_id": spec.image_token_id,
            "image_mean": list(spec.image_mean),
            "image_std": list(spec.image_std),
            "rescale_factor": spec.rescale_factor,
        },
        "source": {
            "hf_model_id": spec.hf_model_id,
            "model_definition": "torch",
        },
    }
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Text bundle complete: {bundle_path}")
    return bundle_path, model_dir, hidden_size


# ---------------------------------------------------------------------------
# vision.aimodel (MobileNet-v5 + soft-token embedder, static 768x768 input)
# ---------------------------------------------------------------------------


async def export_vision_encoder(
    spec: Gemma4VLMSpec,
    bundle_path: Path,
    hidden_size: int,
    overwrite: bool,
) -> str:
    """Export the Gemma 4 vision encoder as vision.aimodel and patch metadata.json."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration as HFModel,
        )
    except ImportError:  # transformers < 5 exposes it as Gemma3n
        from transformers.models.gemma3n.modeling_gemma3n import (  # type: ignore[no-redef]
            Gemma3nForConditionalGeneration as HFModel,
        )

    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}. Export the text decoder first.")

    logger.info(f"Loading {spec.hf_model_id} for vision encoder extraction...")
    hf_model = HFModel.from_pretrained(spec.hf_model_id, dtype=torch.float32).eval()

    export_module = build_vision_encoder(hf_model, image_size=spec.image_size)
    del hf_model

    pixel_shape = (1, 3, spec.image_size, spec.image_size)
    with torch.no_grad():
        out = export_module(torch.rand(*pixel_shape, dtype=torch.float32))
        logger.info(
            f"Vision encoder output {tuple(out.shape)} {out.dtype}; "
            f"expected (1, {spec.num_visual_tokens}, {hidden_size}) float16"
        )

    program = export_to_coreai(
        export_module,
        {"pixel_values": torch.randn(*pixel_shape, dtype=torch.float32)},
        dynamic_shapes=None,
        input_names=("pixel_values",),
        output_names=("image_features",),
    )
    program.optimize()

    vision_path = bundle_path / "vision.aimodel"
    if vision_path.exists():
        if not overwrite:
            raise FileExistsError(f"{vision_path} exists. Use --overwrite.")
        shutil.rmtree(vision_path)
    logger.info(f"Saving vision.aimodel to {vision_path}...")
    meta = build_aimodel_metadata(spec.hf_model_id, component="vision")
    await asyncio.to_thread(program.save_asset, vision_path, meta)

    with open(bundle_path / "metadata.json") as f:
        metadata = json.load(f)
    metadata["assets"]["vision"] = "vision.aimodel"
    with open(bundle_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Updated metadata.json with vision asset")
    return "vision.aimodel"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def export_gemma4_vlm(
    spec: Gemma4VLMSpec,
    *,
    max_ctx: int,
    num_layers: int | None,
    output_dir: Path,
    overwrite: bool,
    skip_vision: bool,
) -> Path:
    bundle_path, _model_dir, hidden_size = await export_text_bundle(
        spec,
        max_ctx=max_ctx,
        num_layers=num_layers,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if not skip_vision:
        logger.info("Exporting vision encoder...")
        await export_vision_encoder(spec, bundle_path, hidden_size, overwrite)
    return bundle_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coreai_models.vlm.gemma4",
        description="Export a Gemma 4 (E2B) vision-language checkpoint to a Core AI bundle.",
    )
    parser.add_argument(
        "--hf-model-id",
        default=GEMMA4_E2B.hf_model_id,
        help=f"HuggingFace model id to export (default: {GEMMA4_E2B.hf_model_id}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the bundle into (default: ./exports).",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=32768,
        help="Max context length baked into the decoder cache (default: 32768).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Truncate the text decoder to N layers (debugging only).",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Export only the text decoder + embedding (skip the vision encoder).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing bundle at the output path.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _build_parser().parse_args()

    spec = GEMMA4_E2B
    if args.hf_model_id and args.hf_model_id != spec.hf_model_id:
        spec = type(spec)(**{**spec.__dict__, "hf_model_id": args.hf_model_id})

    output_dir = args.output_dir or (Path.cwd() / "exports")
    bundle_path = asyncio.run(
        export_gemma4_vlm(
            spec,
            max_ctx=args.max_context_length,
            num_layers=args.num_layers,
            output_dir=output_dir,
            overwrite=args.overwrite,
            skip_vision=args.skip_vision,
        )
    )
    print(f"\nExported Gemma4 VLM bundle: {bundle_path.resolve()}")


if __name__ == "__main__":
    main()
