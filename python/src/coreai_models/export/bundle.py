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
    drafter_config: Any | None = None,
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
        drafter_config: The drafter model's own config. When supplied (the
            DFlash path), the speculative structural metadata is sourced
            EXPLICITLY from it and validated against *hf_config* (the target)
            so a drafter/target divergence fails loudly at export time rather
            than silently shipping wrong metadata.
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
        drafter_config=drafter_config,
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
    drafter_config: Any | None = None,
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
        metadata["speculative"] = _speculative_metadata(
            hf_config, speculative_config, drafter_config=drafter_config
        )

    metadata_path = bundle_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_path}")


# Structural params that MUST agree between the drafter and its target because the
# drafter borrows the target's shared embed_tokens / lm_head and is driven by the
# target's hidden-state features. A divergence here is a build error, not a runtime
# surprise: e.g. the drafter checkpoint declaring vocab_size 262144 against a target
# of 202048 previously crashed the standalone drafter. Layer count is deliberately
# NOT here — the drafter is intentionally shallower than the target.
_DRAFTER_TARGET_SHARED_PARAMS = (
    "vocab_size",
    "hidden_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
)


def _validate_drafter_target_geometry(target_config: Any, drafter_config: Any) -> None:
    """Fail loudly if the drafter and target diverge on shared structural params.

    The DFlash drafter shares the target's embeddings/lm_head and consumes the
    target's ``drafter_features``; the params in ``_DRAFTER_TARGET_SHARED_PARAMS``
    must therefore match exactly. Raises ``ValueError`` naming every divergent
    field and both values, rather than silently emitting metadata sourced from the
    wrong model.
    """
    mismatches: list[str] = []
    for param in _DRAFTER_TARGET_SHARED_PARAMS:
        target_val = getattr(target_config, param, None)
        drafter_val = getattr(drafter_config, param, None)
        # Only compare when BOTH configs declare the param; a param absent on one
        # side isn't a divergence (some tiny/test configs omit e.g. head_dim).
        if target_val is None or drafter_val is None:
            continue
        if target_val != drafter_val:
            mismatches.append(f"{param}: drafter={drafter_val!r} vs target={target_val!r}")

    if mismatches:
        raise ValueError(
            "DFlash drafter/target structural divergence detected while writing "
            "speculative metadata — the drafter shares the target's embeddings and "
            "feature space, so these MUST match:\n  "
            + "\n  ".join(mismatches)
            + "\nRefusing to emit metadata that would silently mispair the drafter "
            "with the target (recall the 262144-vs-202048 vocab_size crash)."
        )


def _speculative_metadata(
    hf_config: Any,
    runtime_knobs: dict[str, Any],
    drafter_config: Any | None = None,
) -> dict[str, Any]:
    """Merge runtime knobs with DFlash structural constants.

    Structural keys (``block_size``, ``mask_token_id``, ``target_layer_ids``,
    ``drafter_hidden_size``) describe the drafter, so they are sourced from
    *drafter_config* when it is supplied (the DFlash path) — an EXPLICIT source of
    truth — and only fall back to *hf_config* (the target) when no drafter config is
    given (the legacy ring-drafter path). ``runtime_knobs`` (``num_draft_tokens``,
    ``shared_embeddings``, ``drafter_kind``) are export-time decisions and win on
    conflict. The config attr is ``draft_mask_token_id`` (the HF-reserved
    ``mask_token_id`` is avoided), but it is emitted under the metadata key
    ``mask_token_id`` the runtime reads.

    When *drafter_config* is supplied it is first validated against *hf_config* via
    :func:`_validate_drafter_target_geometry`, so a drafter/target divergence fails
    loudly here rather than shipping wrong metadata.
    """
    # The drafter's config is the explicit source of truth for its own structural
    # constants; fall back to the target config for the legacy ring path.
    if drafter_config is not None:
        _validate_drafter_target_geometry(hf_config, drafter_config)
        structural_source = drafter_config
    else:
        structural_source = hf_config

    block: dict[str, Any] = {}
    target_layer_ids = getattr(structural_source, "target_layer_ids", None)
    if target_layer_ids is not None:
        block["target_layer_ids"] = list(target_layer_ids)
    mask_token_id = getattr(structural_source, "draft_mask_token_id", None)
    if mask_token_id is not None:
        block["mask_token_id"] = mask_token_id
    block_size = getattr(structural_source, "block_size", None)
    if block_size is not None:
        block["block_size"] = block_size
    drafter_hidden = getattr(structural_source, "drafter_hidden_size", None)
    if drafter_hidden is None:
        drafter_hidden = getattr(structural_source, "hidden_size", None)
    if drafter_hidden is not None:
        block["drafter_hidden_size"] = drafter_hidden
    block.update(runtime_knobs)  # runtime knobs override structural defaults
    return block
