# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "efficient-sam @ git+https://github.com/yformer/EfficientSAM.git",
#     "torch<=2.11.0"
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from efficient_sam.build_efficient_sam import build_efficient_sam


def reference_inputs(
    dtype: torch.dtype,
    dynamic: bool = False,
    num_queries: int = 1,
    num_pts: int = 2,
) -> dict[str, torch.Tensor]:
    B = 2 if dynamic else 1
    H, W = 1024, 1024
    batched_points = torch.rand((B, num_queries, num_pts, 2)) * torch.tensor([H, W])
    return {
        "batched_images": torch.randn((B, 3, H, W)).to(dtype),
        "batched_points": batched_points.to(dtype),
        "batched_point_labels": torch.ones((B, num_queries, num_pts)).to(dtype),
    }


def dynamic_shapes() -> dict:
    """Dynamic shape specification for EfficientSAM.

    Static (default): batch is fixed at 1, images at 1024x1024.
    Dynamic (--dynamic): batch dim (1-64) can vary; spatial dims stay fixed.
    """
    batch = torch.export.Dim("batch_size", min=1, max=64)
    return {
        "batched_images": {0: batch},
        "batched_points": {0: batch},
        "batched_point_labels": {0: batch},
    }


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "exports")


def _variant_name(
    model_name: str, dtype: torch.dtype, dynamic: bool, num_queries: int, num_pts: int
) -> str:
    safe_name = Path(model_name).name
    dtype_name = str(dtype).split(".")[-1]
    static_or_dynamic = "dynamic" if dynamic else "static"
    q_suffix = f"_q{num_queries}" if num_queries != 1 else ""
    p_suffix = f"_p{num_pts}" if num_pts != 1 else ""
    return f"{safe_name}_{dtype_name}_{static_or_dynamic}{q_suffix}{p_suffix}"


def _bundle_paths(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    dynamic: bool,
    num_queries: int,
    num_pts: int,
) -> tuple[Path, Path]:
    """Return (bundle_dir, model_path) where the .aimodel sits inside the bundle dir."""
    variant = _variant_name(model_name, dtype, dynamic, num_queries, num_pts)
    bundle_dir = Path(output_dir) / variant
    return bundle_dir, bundle_dir / f"{variant}.aimodel"


def _save_asset(
    coreai_program, bundle_dir: Path, model_path: Path, overwrite: bool
) -> None:
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{bundle_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    coreai_program.save_asset(model_path, _build_aimodel_metadata())


def _build_aimodel_metadata() -> AIModelAssetMetadata:
    # Source: https://github.com/yformer/EfficientSAM
    metadata = AIModelAssetMetadata()
    metadata.author = "Y. Xiong et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = "EfficientSAM is a lightweight, promptable image segmentation model that uses a masked autoencoder pretrained ViT-Tiny encoder to reduce compute while preserving accuracy. Source: https://github.com/yformer/EfficientSAM"
    metadata.creation_date = int(time.time())
    return metadata


def _write_metadata(bundle_dir: Path, variant: str) -> None:
    metadata = {
        "metadata_version": "0.2",
        "kind": "segmenter",
        "name": variant,
        "assets": {"main": f"{variant}.aimodel"},
    }
    metadata_path = bundle_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote metadata to {metadata_path}.")


def create_efficient_sam(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    dynamic: bool,
    num_queries: int,
    num_pts: int,
):
    if dynamic and dtype == torch.float16:
        raise ValueError(
            "float16 + dynamic is not supported for EfficientSAM. "
            "The Core AI runtime cannot handle dynamic reshape in the "
            "attention heads at float16. Use float32 --dynamic or "
            "float16 without --dynamic."
        )

    print("[INFO] Downloading weights and sourcing model...")
    checkpoint = (
        "https://huggingface.co/merve/EfficientSAM/resolve/main/efficient_sam_vitt.pt"
    )
    state_dict = torch.hub.load_state_dict_from_url(
        checkpoint, map_location="cpu", progress=True, weights_only=True
    )
    weights_dir = Path(__file__).resolve().parents[2] / ".build"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_filepath = weights_dir / "efficient_sam_weights.pt"
    torch.save(state_dict, weights_filepath)
    model = build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=weights_filepath,
    ).eval()
    model.to(dtype)
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(dtype, dynamic, num_queries, num_pts)
    ds = dynamic_shapes() if dynamic else None

    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            model, args=(), kwargs=example_inputs, dynamic_shapes=ds
        )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["batched_images", "batched_points", "batched_point_labels"],
        output_names=["pred_masks", "iou_scores"],
    )
    coreai_program = converter.to_coreai()
    print("[INFO] Model converted.")
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    bundle_dir, model_path = _bundle_paths(
        output_dir, model_name, dtype, dynamic, num_queries, num_pts
    )
    _save_asset(coreai_program, bundle_dir, model_path, overwrite)
    print(f"[INFO] Successfully created and saved Core AI model to {model_path}.")

    _write_metadata(
        bundle_dir, _variant_name(model_name, dtype, dynamic, num_queries, num_pts)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create and save a Core AI AIProgram for EfficientSAM."
    )
    parser.add_argument(
        "--model",
        choices=["efficient_sam_vitt"],
        default="efficient_sam_vitt",
        help="Model variant to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the bundle (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Torch dtype to use for the model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing bundle at the output path.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic input shapes.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=1,
        help=(
            "Number of prompt queries (Q dim of batched_points). Use 1 for interactive "
            "single-point use, or a perfect square (e.g. 64 = 8x8, 256 = 16x16) for "
            "segment-everything grid mode."
        ),
    )
    parser.add_argument(
        "--num-pts",
        type=int,
        default=2,
        help=(
            "Number of points per query (P dim of batched_points). Use 1 for a single "
            "point prompt, or 2 with labels [2, 3] for a box prompt (top-left + "
            "bottom-right). Higher values support combined point+box prompts."
        ),
    )
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_efficient_sam(
        output_dir,
        args.model,
        dtype,
        args.overwrite,
        args.dynamic,
        args.num_queries,
        args.num_pts,
    )


if __name__ == "__main__":
    main()
