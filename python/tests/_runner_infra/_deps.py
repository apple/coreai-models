# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Optional-dependency probes shared across the runner infrastructure."""

import importlib
import importlib.util
import platform


def _find_package(name: str) -> tuple[bool, str]:
    has = importlib.util.find_spec(name)
    msg = f"No module named '{name}'"
    return bool(has), msg


_HAS_MLX, _MSG_MLX_NOT_FOUND = _find_package("mlx")
# even if available, non-mac OS such as linux only has super outdated version
_HAS_MLX = _HAS_MLX and platform.system() == "Darwin"


def _has_coreai_with_ai_program() -> tuple[bool, str]:
    """Probe for the specific ``coreai.authoring.AIProgram`` symbol.

    Even if ``import coreai`` succeeds, the symbols required by the parity
    runners (``AIProgram`` and ``coreai.runtime.NDArray``) may be missing
    on older ``coreai-core`` wheels. Treat any ImportError here as
    "coreai not available" for the purposes of the runner infrastructure,
    so optional-import gates skip the Core AI runner cleanly instead of
    blowing up at collection time.
    """
    has_pkg = importlib.util.find_spec("coreai") is not None
    if not has_pkg:
        return False, "No module named 'coreai'"
    try:
        from coreai.authoring import AIProgram  # noqa: F401
        from coreai.runtime import NDArray  # noqa: F401
    except ImportError as exc:
        return False, f"coreai is installed but missing required symbols: {exc}"
    return True, "coreai and required symbols (AIProgram, NDArray) available"


_HAS_COREAI, _MSG_COREAI_NOT_FOUND = _has_coreai_with_ai_program()


def _hf_hub_reachable(model_id: str = "yujiepan/qwen3-tiny-random") -> bool:
    """Return True if the HuggingFace Hub responds for ``model_id`` within 2s.

    Used as an opt-in gate for tests that call ``Model.from_hf(...)``. Any
    exception (proxy block, DNS failure, timeout, missing dep) is treated as
    "not reachable" so the test is skipped rather than failing.

    Also verifies the model's config.json is downloadable — model_info()
    can succeed while actual file downloads are blocked by proxy/firewall.
    """
    try:
        import huggingface_hub

        huggingface_hub.HfApi().model_info(model_id, timeout=2)
        huggingface_hub.hf_hub_download(model_id, "config.json")
        return True
    except Exception:
        return False
