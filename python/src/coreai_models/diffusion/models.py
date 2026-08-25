# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Supported diffusion model families for export.

Diffusion uses HF model IDs directly (no internal registry) — these are
the families known to work end-to-end through the export pipeline.
"""

# Each entry: (family name, example HF model ID, pipeline_type)
SUPPORTED_MODELS: list[tuple[str, str, str]] = [
    ("stable-diffusion-1.x", "runwayml/stable-diffusion-v1-5", "sd"),
    ("stable-diffusion-2.x", "sd2-community/stable-diffusion-2-1", "sd"),
    ("stable-diffusion-3.x", "stabilityai/stable-diffusion-3.5-medium", "sd3"),
    ("flux2", "black-forest-labs/FLUX.2-klein-4B", "flux2"),
    ("wan-t2v-1.3b", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan"),
]


def list_models() -> list[str]:
    """Return the names of supported diffusion model families."""
    return [name for name, _, _ in SUPPORTED_MODELS]


def get_pipeline_type(model_id: str) -> str:
    """Determine the pipeline type for a given HF model ID.

    Returns "sd", "sd3", or "flux2". Raises ValueError for unknown models.
    """
    for _, known_id, ptype in SUPPORTED_MODELS:
        if model_id == known_id:
            return ptype

    raise ValueError(
        f"Unknown diffusion model: '{model_id}'. "
        f"Supported models: {[mid for _, mid, _ in SUPPORTED_MODELS]}"
    )
