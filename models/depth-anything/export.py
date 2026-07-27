# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git",
#     "scipy<1.15",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# # xformers is a hard dep in depth-anything-3's pyproject.toml, but DA3's code
# # catches `ImportError` on `from xformers.ops import SwiGLU` and falls back to
# # a pure-torch SwiGLU. xformers has no macOS-arm64 wheel and its sdist build
# # fails on Apple Silicon. The override marks xformers as only needed on an
# # impossible Python version, which makes uv's resolver drop it entirely.
# override-dependencies = ["xformers ; python_version >= '99'"]
# ///
import argparse
import shutil
import time
from pathlib import Path

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.dinov2.layers import rope as _rope


def _patch_depth_anything_for_export() -> None:
    # `RotaryPositionEmbedding2D.forward` uses `int(positions.max()) + 1`, a
    # data-dependent guard that breaks torch.export. Swap in a shape-based
    # equivalent.
    def _rope_forward(
        self, tokens: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0
        assert positions.ndim == 3 and positions.shape[-1] == 2
        feature_dim = tokens.size(-1) // 2
        max_position = positions.shape[-2] + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
        horizontal = self._apply_1d_rope(
            horizontal, positions[..., 1], cos_comp, sin_comp
        )
        return torch.cat((vertical, horizontal), dim=-1)

    _rope.RotaryPositionEmbedding2D.forward = _rope_forward


class DepthAnythingModule(torch.nn.Module):
    # Wraps DepthAnything3 to bypass the autocast/no_grad block in its forward
    # and filter the addict.Dict return down to plain tensors (needed for
    # torch.export).
    def __init__(self, model_name: str):
        super().__init__()
        _patch_depth_anything_for_export()
        self._api = DepthAnything3.from_pretrained(model_name)

    def forward(self, image: torch.Tensor, export_feat_layers: list[int]):
        raw = self._api.model(
            image, None, None, export_feat_layers, False, False, "saddle_balanced"
        )
        return {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}


def reference_inputs(dtype: torch.dtype) -> dict[str, torch.Tensor | list[int]]:
    # depth-anything expects (B, N, 3, H, W) where N is number of views.
    # export_feat_layers=[] selects the "no aux features" path; the default of
    # None trips len(None) / `in None` checks inside the model.
    return {
        "image": torch.randn(1, 2, 3, 224, 224).to(dtype),
        "export_feat_layers": [],
    }


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "exports")


def _variant_name(model_name: str, dtype: torch.dtype) -> str:
    safe_name = Path(model_name).name
    dtype_name = str(dtype).split(".")[-1]
    return f"{safe_name}_{dtype_name}"


def _asset_path(output_dir: str, model_name: str, dtype: torch.dtype) -> Path:
    return Path(output_dir) / f"{_variant_name(model_name, dtype)}.aimodel"


def _save_asset(coreai_program, model_path: Path, overwrite: bool) -> None:
    if model_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{model_path} already exists. Pass --overwrite to replace it."
            )
        if model_path.is_dir():
            shutil.rmtree(model_path)
        else:
            model_path.unlink()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    coreai_program.save_asset(model_path, _build_aimodel_metadata())


def _build_aimodel_metadata() -> AIModelAssetMetadata:
    # Source: https://github.com/ByteDance-Seed/Depth-Anything-3
    metadata = AIModelAssetMetadata()
    metadata.author = "H. Lin et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = "Depth Anything v3 is a monocular depth estimation model that predicts depth, confidence, camera intrinsics, and extrinsics from a batch of image views. Source: https://github.com/ByteDance-Seed/Depth-Anything-3"
    metadata.creation_date = int(time.time())
    return metadata


def create_depth_anything(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
):
    print("[INFO] Sourcing model...")
    model = DepthAnythingModule(model_name)
    model.eval()
    model.to(dtype)
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(dtype)

    exported = torch.export.export(
        model,
        args=(),
        kwargs=example_inputs,
    )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["image"],
        output_names=["depth", "depth_conf", "extrinsics", "intrinsics"],
    )
    coreai_program = converter.to_coreai()
    print("[INFO] Model converted.")
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    model_path = _asset_path(output_dir, model_name, dtype)
    _save_asset(coreai_program, model_path, overwrite)
    print(f"[INFO] Successfully created and saved Core AI model to {model_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Create and save a Core AI AIProgram for Depth Anything v3."
    )
    parser.add_argument(
        "--model",
        choices=["depth-anything/da3-small"],
        default="depth-anything/da3-small",
        help="Model variant to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the .aimodel asset (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32"],
        default="float32",
        help="Torch dtype to use for the model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing .aimodel asset at the output path.",
    )
    args = parser.parse_args()

    dtype = {
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_depth_anything(
        output_dir,
        args.model,
        dtype,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
