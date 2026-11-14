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
- The centroids, projections and norms are fp16; a runtime widens what it needs.

The pack is per variant: the assistant reads the target's final hidden state, so its
``post_projection`` is as wide as the target's hidden size (1536 for E2B, 2560 for E4B).

    cd models/gemma4
    uv run export_draft_pack.py --model google/gemma-4-E2B-it

It writes ``exports/draft_head/<model>/draft_head.safetensors`` unless ``--output-dir`` says
otherwise. With ``--model``, the assistant must match that target's hidden size.

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
    resolve_drafter,
    scan_tensors,
)
from safetensors.torch import save_file
from transformers import AutoConfig

PACK_NAME = "draft_head.safetensors"
# `__metadata__["format"]`, which tells this pack from another safetensors file.
PACK_FORMAT = "gemma4_draft_head"


def default_output_dir(model: str | None, drafter: Path) -> Path:
    """``exports/draft_head/<name>``: one directory per variant, named for the target when given."""
    name = model.rsplit("/", 1)[-1] if model else drafter.name
    return REPO_ROOT / "exports" / "draft_head" / name


def pack(drafter: Path, out_dir: Path, backbone_hidden_size: int | None = None) -> Path:
    """Write ``draft_head.safetensors`` to ``out_dir`` and return its path.

    ``backbone_hidden_size`` is the target's hidden size; when given, an assistant trained for
    another target is refused.
    """
    t = load_drafter(drafter)
    config = json.loads((drafter / "config.json").read_text())
    text = config["text_config"]
    check_checkpoint(t, text, backbone_hidden_size)
    num_layers = text["num_hidden_layers"]
    rope = text["rope_parameters"]

    vocab, centroids, tensors = scan_tensors(t)
    tensors.append(("centroids_f16", t["masked_embedding.centroids.weight"].to(torch.float16)))
    for name in ("pre_projection.weight", "post_projection.weight", "model.norm.weight"):
        tensors.append((name, t[name].to(torch.float16)))
    for i in range(num_layers):
        for suffix in LAYER_TENSORS:
            name = f"model.layers.{i}.{suffix}"
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
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write draft_head.safetensors (default: exports/draft_head/<model or "
        "assistant name>, one directory per variant, since the pack is per variant)",
    )
    args = parser.parse_args()
    if args.model is None and args.drafter is None:
        parser.error("pass --model (for its default assistant) or --drafter")
    drafter = resolve_drafter(args.drafter, args.model)
    out_dir = args.output_dir or default_output_dir(args.model, drafter)
    # With a target named, an assistant trained for another target is refused here.
    target_hidden = None
    if args.model:
        target_hidden = AutoConfig.from_pretrained(args.model).text_config.hidden_size
    pack(drafter, out_dir, target_hidden)


if __name__ == "__main__":
    main()
