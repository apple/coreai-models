# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pack the Gemma 4 MTP/EAGLE drafter for a runtime that runs the whole draft head on the CPU.

There is no Core AI program here, only one safetensors file, ``draft_head.safetensors``, that the
runtime maps and reads in place.

- The output table (``lm_head.weight`` when the checkpoint has one, else the embedding) is permuted
  by ``masked_embedding.token_ordering``, so each centroid's tokens are contiguous, exactly as in
  the ``draft_scan.bin`` that ``export_drafter.py`` writes. It is a pure reordering, so the scan
  is bit exact.
- ``token_ordering`` is narrowed to int32.
- ``__metadata__`` records the head's geometry, its sliding window and its RoPE parameters.
- The centroids, projections and norms are fp16. With ``--int8-attention`` the attention
  projections (``q_proj``, ``o_proj``) are symmetric int8 per output row with an fp16 scale. That
  reads about an eighth fewer weight bytes per draft step and leaves the share of accepted drafts
  where fp16 has it; quantizing the MLP as well costs acceptance, so it is not offered. The
  drafter only proposes and the target verifies every token, so a coarser weight can cost speed,
  never correctness.

The pack is per variant: the assistant reads the target's final hidden state, so its
``post_projection`` is as wide as the target's hidden size (1536 for E2B, 2560 for E4B).

    cd models/gemma4
    uv run export_draft_pack.py --model google/gemma-4-E2B-it [--int8-attention]

The assistant is the target's own (``<model>-assistant`` on the Hugging Face hub), downloaded on
first use; ``--drafter`` takes another repo id or a local directory instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from export_drafter import (
    LAYER_TENSORS,
    REPO_ROOT,
    SIDECAR_VERSION,
    check_checkpoint,
    load_drafter,
    output_table,
    resolve_drafter,
)
from safetensors.torch import save_file

PACK_NAME = "draft_head.safetensors"
# `__metadata__["format"]`, which tells this pack from another safetensors file.
PACK_FORMAT = "gemma4_draft_head"
# The projections `--int8-attention` quantizes.
ATTENTION_PROJECTIONS = ("self_attn.q_proj.weight", "self_attn.o_proj.weight")


def int8_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` as symmetric int8 with one fp16 scale per output row: ``w ≈ q * scale[:, None]``."""
    scale = w.float().abs().amax(dim=1, keepdim=True) / 127.0
    scale[scale == 0] = 1.0  # an all-zero row stays zero under any scale; 1 keeps it finite in fp16
    q = torch.round(w.float() / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.reshape(-1).to(torch.float16)


def pack(drafter: Path, out_dir: Path, int8_attention: bool = False) -> Path:
    """Write ``draft_head.safetensors`` to ``out_dir`` and return its path."""
    t = load_drafter(drafter)
    config = json.loads((drafter / "config.json").read_text())
    text = config["text_config"]
    check_checkpoint(t, text)
    num_layers = text["num_hidden_layers"]
    rope = text["rope_parameters"]

    table = output_table(t)
    vocab = table.shape[0]
    centroids = t["masked_embedding.centroids.weight"].shape[0]
    if centroids <= 0 or vocab % centroids != 0:
        raise SystemExit(
            f"vocabulary {vocab} is not divisible by the {centroids} centroids: the tokens past "
            f"{vocab // max(centroids, 1) * max(centroids, 1)} would never be scored"
        )
    order = t["masked_embedding.token_ordering"].to(torch.int64)
    if not torch.equal(order.sort().values, torch.arange(vocab)):
        raise SystemExit("masked_embedding.token_ordering is not a permutation of the vocabulary")

    tensors: list[tuple[str, torch.Tensor]] = [
        ("embed_perm_f16", table[order].to(torch.float16)),
        ("token_ordering_i32", order.to(torch.int32)),
        ("centroids_f16", t["masked_embedding.centroids.weight"].to(torch.float16)),
    ]
    for name in ("pre_projection.weight", "post_projection.weight", "model.norm.weight"):
        tensors.append((name, t[name].to(torch.float16)))
    for i in range(num_layers):
        for suffix in LAYER_TENSORS:
            name = f"model.layers.{i}.{suffix}"
            if int8_attention and suffix in ATTENTION_PROJECTIONS:
                q, scale = int8_rows(t[name])
                tensors += [(name, q), (name + ".scale", scale)]
            else:
                tensors.append((name, t[name].to(torch.float16)))

    # The geometry travels with the tensors, so the runtime adopts it instead of assuming a
    # variant's. safetensors metadata values are strings.
    metadata = {
        "format": PACK_FORMAT,
        "version": SIDECAR_VERSION,
        "vocab": vocab,
        "centroids": centroids,
        "per_centroid": vocab // centroids,
        "layers": num_layers,
        "hidden": text["hidden_size"],
        "heads": text["num_attention_heads"],
        "head_dim": text["head_dim"],
        "full_head_dim": text.get("global_head_dim", text["head_dim"]),
        "ffn": text["intermediate_size"],
        # The runtime builds the rotary tables and the sliding mask itself.
        "sliding_window": text["sliding_window"],
        "sliding_theta": rope["sliding_attention"]["rope_theta"],
        "full_theta": rope["full_attention"]["rope_theta"],
        "full_partial_rotary_factor": rope["full_attention"].get("partial_rotary_factor", 1.0),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PACK_NAME
    save_file(
        {name: x.contiguous() for name, x in tensors},
        str(path),
        metadata={k: str(v) for k, v in metadata.items()},
    )
    print(f"Packed {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default=None,
        help="Target HuggingFace model ID, which names the default assistant ('<model>-assistant')",
    )
    parser.add_argument(
        "--drafter",
        default=None,
        help="The assistant: a hub repo id or a local directory holding its safetensors",
    )
    parser.add_argument(
        "--int8-attention",
        action="store_true",
        help="Store the attention projections as int8 with a scale per output row",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "exports" / "drafter")
    args = parser.parse_args()
    if args.model is None and args.drafter is None:
        parser.error("pass --model (for its default assistant) or --drafter")
    drafter = resolve_drafter(args.drafter, args.model)
    pack(drafter, args.output_dir, args.int8_attention)


if __name__ == "__main__":
    main()
