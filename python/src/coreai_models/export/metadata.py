# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Per-model AIModel asset metadata for model exports.

Maps a HuggingFace model id to author/license/description fields. The pipeline
calls :func:`build_aimodel_metadata` and passes the result to
``AIProgram.save_asset``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from coreai.runtime import AIModelAssetMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIModelMetadataFields:
    """Static metadata fields for a single HuggingFace model."""

    author: str
    license: str
    model_description: str


# Keyed by HuggingFace model id (as it appears in `ExportConfig.hf_model_id`).
# When you add a new model to one of the preset tables in
# `coreai_models.model_registry`, add a matching entry here.
_METADATA: dict[str, AIModelMetadataFields] = {
    # ---- LLMs ----
    "Qwen/Qwen2.5-1.5B-Instruct": AIModelMetadataFields(
        author="Qwen Team",
        license="Apache-2.0",
        model_description=(
            "Qwen2.5-1.5B-Instruct is a 1.5B-parameter instruction-tuned causal "
            "language model from the Qwen2.5 family. "
            "Source: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"
        ),
    ),
    "Qwen/Qwen3-0.6B": AIModelMetadataFields(
        author="Qwen Team",
        license="Apache-2.0",
        model_description=(
            "Qwen3-0.6B is a 0.6B-parameter causal language model from the Qwen3 "
            "family. Source: https://huggingface.co/Qwen/Qwen3-0.6B"
        ),
    ),
    "Qwen/Qwen3-4B": AIModelMetadataFields(
        author="Qwen Team",
        license="Apache-2.0",
        model_description=(
            "Qwen3-4B is a 4B-parameter causal language model from the Qwen3 "
            "family. Source: https://huggingface.co/Qwen/Qwen3-4B"
        ),
    ),
    "Qwen/Qwen3-8B": AIModelMetadataFields(
        author="Qwen Team",
        license="Apache-2.0",
        model_description=(
            "Qwen3-8B is an 8B-parameter causal language model from the Qwen3 "
            "family. Source: https://huggingface.co/Qwen/Qwen3-8B"
        ),
    ),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": AIModelMetadataFields(
        author="Qwen Team",
        license="Apache-2.0",
        model_description=(
            "Qwen3-Coder-30B-A3B-Instruct is a 30B-parameter mixture-of-experts "
            "instruction-tuned coding model from the Qwen3 family. "
            "Source: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct"
        ),
    ),
    "google/gemma-3-4b-it": AIModelMetadataFields(
        author="Gemma Team",
        license="Gemma Terms of Use",
        model_description=(
            "Gemma 3 4B IT is a 4B-parameter instruction-tuned multimodal model "
            "from Google's Gemma 3 family; this export targets the text decoder. "
            "Source: https://huggingface.co/google/gemma-3-4b-it"
        ),
    ),
    "google/gemma-3-12b-it": AIModelMetadataFields(
        author="Gemma Team",
        license="Gemma Terms of Use",
        model_description=(
            "Gemma 3 12B IT is a 12B-parameter instruction-tuned multimodal model "
            "from Google's Gemma 3 family; this export targets the text decoder. "
            "Source: https://huggingface.co/google/gemma-3-12b-it"
        ),
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": AIModelMetadataFields(
        author="Mistral AI",
        license="Apache-2.0",
        model_description=(
            "Mistral 7B Instruct v0.3 is a 7B-parameter instruction-tuned causal "
            "language model from Mistral AI. "
            "Source: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3"
        ),
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": AIModelMetadataFields(
        author="Mistral AI",
        license="Apache-2.0",
        model_description=(
            "Mixtral 8x7B Instruct v0.1 is a sparse mixture-of-experts "
            "instruction-tuned causal language model from Mistral AI. "
            "Source: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1"
        ),
    ),
    "openai/gpt-oss-20b": AIModelMetadataFields(
        author="OpenAI",
        license="Apache-2.0",
        model_description=(
            "gpt-oss-20b is a 20B-parameter open-weights causal language model "
            "released by OpenAI. "
            "Source: https://huggingface.co/openai/gpt-oss-20b"
        ),
    ),
    # ---- VLMs ----
    "Qwen/Qwen3-VL-2B-Instruct": AIModelMetadataFields(
        author="Qwen Team",
        license="Apache-2.0",
        model_description=(
            "Qwen3-VL-2B-Instruct is a 2B-parameter instruction-tuned "
            "vision-language model from the Qwen3-VL family. "
            "Source: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct"
        ),
    ),
    # ---- Diffusion ----
    "runwayml/stable-diffusion-v1-5": AIModelMetadataFields(
        author="Robin Rombach, Patrick Esser, et al.",
        license="CreativeML Open RAIL-M",
        model_description=(
            "Stable Diffusion v1.5 is a latent text-to-image diffusion model "
            "trained on a subset of LAION-5B that generates images from natural "
            "language prompts. "
            "Source: https://huggingface.co/runwayml/stable-diffusion-v1-5"
        ),
    ),
    "sd2-community/stable-diffusion-2-1": AIModelMetadataFields(
        author="Stability AI",
        license="CreativeML Open RAIL++-M",
        model_description=(
            "Stable Diffusion 2.1 is a latent text-to-image diffusion model from "
            "Stability AI, fine-tuned from SD 2.0 with reduced restrictive "
            "filtering. "
            "Source: https://huggingface.co/sd2-community/stable-diffusion-2-1"
        ),
    ),
    "stabilityai/stable-diffusion-3.5-medium": AIModelMetadataFields(
        author="Stability AI",
        license="Stability AI Community License",
        model_description=(
            "Stable Diffusion 3.5 Medium is a Multimodal Diffusion Transformer "
            "(MMDiT-X) text-to-image model from Stability AI. "
            "Source: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
        ),
    ),
    "black-forest-labs/FLUX.2-klein-4B": AIModelMetadataFields(
        author="Black Forest Labs",
        license="Apache-2.0",
        model_description=(
            "FLUX.2 [klein] 4B is a distilled, open-weights text-to-image "
            "rectified-flow transformer from Black Forest Labs. "
            "Source: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
        ),
    ),
    # ---- Segmentation ----
    "facebook/sam3": AIModelMetadataFields(
        author="N. Carion et al.",
        license="SAM License",
        model_description=(
            "SAM 3 is a unified foundation model for promptable segmentation in "
            "images and videos. It can detect, segment, and track objects using "
            "text or visual prompts such as points, boxes, and masks. This export "
            "targets image segmentation, re-authored for iOS. "
            "Source: https://github.com/facebookresearch/sam3"
        ),
    ),
}


def build_aimodel_metadata(hf_model_id: str, component: str | None = None) -> AIModelAssetMetadata:
    """Build an :class:`AIModelAssetMetadata` for ``hf_model_id``.

    Stamps the current time as ``creation_date``. If ``component`` is given
    (e.g. ``"TextEncoder"``, ``"VAEDecoder"``, ``"vision"``), it is appended
    to ``model_description`` so multi-component bundles can identify which
    sub-model an asset is. If no metadata entry is registered for the given
    id, logs a loud warning and returns metadata with only ``creation_date``
    populated (author/license/description left blank), so the export still
    completes.
    """
    fields = _METADATA.get(hf_model_id)
    if fields is None:
        banner = "!" * 80
        logger.warning(
            "\n%s\n"
            "WARNING: No AIModel metadata registered for hf_model_id '%s'.\n"
            "         The exported asset will ship without author/license/"
            "description fields.\n"
            "         Add an entry to coreai_models/export/metadata.py.\n"
            "%s",
            banner,
            hf_model_id,
            banner,
        )

    metadata = AIModelAssetMetadata()
    if fields is not None:
        metadata.author = fields.author
        metadata.license = fields.license
        description = fields.model_description
        if component:
            description = f"{description} — {component} component."
        metadata.model_description = description
    metadata.creation_date = int(time.time())
    return metadata
