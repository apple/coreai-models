# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import glob
from pathlib import Path
from typing import Any

import safetensors
import torch

from coreai_models._download import download_snapshot


def _download_from_huggingface(huggingface_model_id: str) -> Path:
    """
    Download model snapshot then return local path.

    Despite the legacy name, this may route to either the HuggingFace Hub
    (default) or the ModelScope hub — the download goes through the unified
    abstraction in :mod:`coreai_models._download`, which selects the backend
    via the ``COREAI_DOWNLOAD_BACKEND`` env var
    (or an explicit backend). The underlying hub manages local cache, so it
    will not re-download if already cached.
    """
    allow_patterns = [
        "*.json",
        "model*.safetensors",
        "*.py",
        "tokenizer.model",
        "*.tiktoken",
        "tiktoken.model",
        "*.txt",
        "*.jsonl",
        "*.jinja",
    ]
    path_str = download_snapshot(
        huggingface_model_id,
        allow_patterns=allow_patterns,
    )
    return Path(path_str)


def load_named_tensors_from_weight_files(
    huggingface_model_id: str,
) -> dict[str, torch.Tensor]:
    model_path = _download_from_huggingface(huggingface_model_id)
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        with safetensors.safe_open(weight_file, framework="pt") as safe_open:
            for key in safe_open.keys():  # noqa: SIM118
                if key in state_dict:
                    message = "not yet support tensors sharded across multiple files"
                    raise NotImplementedError(message)
                state_dict[key] = safe_open.get_tensor(key)
    return state_dict


def resolve_rope_theta(config: Any, default: float | None = None) -> float | None:
    """Locate RoPE theta across HuggingFace transformers versions.

    Transformers ≥ 4.x moved `config.rope_theta` into `config.rope_parameters`
    (and mirrored in `config.rope_scaling`). This helper checks the legacy
    attribute first, then the new dicts, and falls back to `default` if none
    is found. Callers that need to know whether RoPE scaling is in effect
    should use `is_default_rope_scaling()`.
    """
    theta = getattr(config, "rope_theta", None)
    if theta is not None:
        return theta
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return params["rope_theta"]
    scaling = getattr(config, "rope_scaling", None)
    if isinstance(scaling, dict) and "rope_theta" in scaling:
        return scaling["rope_theta"]
    return default


def is_default_rope_scaling(config: Any) -> bool:
    """Return True if the config has no non-trivial RoPE scaling.

    Newer HuggingFace configs always populate `rope_scaling` with
    `{"rope_type": "default", ...}` even when no scaling is in effect. This
    helper lets callers guard against *actual* scaling (linear, yarn, …)
    without tripping on the benign default-shaped dict.
    """
    scaling = getattr(config, "rope_scaling", None)
    if scaling is None:
        return True
    return isinstance(scaling, dict) and scaling.get("rope_type", "default") == "default"
