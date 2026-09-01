# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Create model bundles from exported .aimodel files."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

METADATA_VERSION = "0.2"


def bundle_llm_asset(
    bundle_path: Path,
    hf_model_id: str,
    hf_config: Any,
    compression: str,
    name: str,
    tokenizer_model_id: str | None = None,
    drafter_name: str | None = None,
    speculative_config: dict[str, Any] | None = None,
) -> None:
    """Add tokenizer and metadata.json (0.2 schema) to an LLM bundle.

    Expects ``{name}.aimodel`` to already exist inside bundle_path.
    When *drafter_name* is set, ``{drafter_name}.aimodel`` must also exist;
    the metadata will list both assets under ``"main"`` and ``"drafter"``
    and include a ``"speculative"`` block.

    Args:
        tokenizer_model_id: HF model ID to download the tokenizer from.
            Defaults to *hf_model_id* when ``None``.  Useful for drafters
            whose checkpoint has no tokenizer (they share the target's).
        drafter_name: Base name of the drafter ``.aimodel`` inside the bundle.
            When ``None``, no drafter is included.
        speculative_config: Runtime config for speculative decoding (e.g.
            ``{"num_draft_tokens": 5, "shared_embeddings": True}``).
            Written as the ``"speculative"`` key in metadata.json.
    """
    tok_id = tokenizer_model_id or hf_model_id
    _write_tokenizer(bundle_path / "tokenizer", tok_id)
    _write_metadata(
        bundle_path,
        hf_model_id,
        hf_config,
        compression,
        name,
        drafter_name=drafter_name,
        speculative_config=speculative_config,
    )


def _write_tokenizer(dest: Path, hf_model_id: str) -> None:
    logger.info(f"Saving tokenizer from {hf_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer.save_pretrained(str(dest))


def _write_metadata(
    bundle_path: Path,
    hf_model_id: str,
    hf_config: Any,
    compression: str,
    name: str,
    drafter_name: str | None = None,
    speculative_config: dict[str, Any] | None = None,
) -> None:
    assets: dict[str, str] = {"main": f"{name}.aimodel"}
    if drafter_name is not None:
        assets["drafter"] = f"{drafter_name}.aimodel"

    metadata: dict[str, Any] = {
        "metadata_version": METADATA_VERSION,
        "kind": "llm",
        "name": name,
        "assets": assets,
        "language": {
            "tokenizer": hf_model_id,
            "vocab_size": getattr(hf_config, "vocab_size", None),
            "max_context_length": getattr(hf_config, "max_position_embeddings", None),
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "source": {
            "model_definition": "torch",
            "hf_model_id": hf_model_id,
        },
        "compression": compression if compression != "none" else None,
        "compilation": {
            "date": datetime.now().astimezone().isoformat(),
            "targets": [],
        },
    }
    if speculative_config is not None:
        metadata["speculative"] = speculative_config

    metadata_path = bundle_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_path}")
