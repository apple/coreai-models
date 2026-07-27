# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "timm",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
import argparse
import shutil
import time
from pathlib import Path

import timm
import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table


def reference_inputs(
    dynamic: bool = False, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    B = 2 if dynamic else 1
    return {"x": torch.randn(B, 3, 224, 224, dtype=dtype)}


def dynamic_shapes() -> dict:
    """Dynamic shape specification for PVT.

    Static (default): x is fixed at (1, 3, 224, 224).
    Dynamic (--dynamic): batch dim (1-64) can vary; spatial dims stay 224x224.
    Reference batch is bumped to 2 so torch.export doesn't specialize dim 0.
    """
    batch = torch.export.Dim("batch_size", min=1, max=64)
    return {"x": {0: batch}}


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "exports")


def _variant_name(model_name: str, dtype: torch.dtype, dynamic: bool) -> str:
    safe_name = Path(model_name).name
    dtype_name = str(dtype).split(".")[-1]
    static_or_dynamic = "dynamic" if dynamic else "static"
    return f"{safe_name}_{dtype_name}_{static_or_dynamic}"


def _asset_path(
    output_dir: str, model_name: str, dtype: torch.dtype, dynamic: bool
) -> Path:
    return Path(output_dir) / f"{_variant_name(model_name, dtype, dynamic)}.aimodel"


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
    # Source: https://huggingface.co/docs/transformers/model_doc/pvt
    metadata = AIModelAssetMetadata()
    metadata.author = "W. Wang et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = "PVT v2 (Pyramid Vision Transformer v2) uses a pyramid structure as an effective backbone for dense prediction tasks. Source: https://huggingface.co/docs/transformers/model_doc/pvt"
    metadata.creation_date = int(time.time())
    return metadata


def create_pvt(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    dynamic: bool,
):
    print("[INFO] Sourcing model...")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(dtype)
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = example_inputs = reference_inputs(dynamic, dtype)
    ds = dynamic_shapes() if dynamic else None

    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            model, args=(), kwargs=example_inputs, dynamic_shapes=ds
        )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["x"],
        output_names=["logits"],
    )
    coreai_program = converter.to_coreai()
    print("[INFO] Model converted.")
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    model_path = _asset_path(output_dir, model_name, dtype, dynamic)
    _save_asset(coreai_program, model_path, overwrite)
    print(f"[INFO] Successfully created and saved Core AI model to {model_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Create and save a Core AI AIProgram for PVT."
    )
    parser.add_argument(
        "--model",
        choices=["pvt_v2_b0"],
        default="pvt_v2_b0",
        help="Model variant to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the .aimodel asset (default: <repo-root>/exports/)",
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
        help="Overwrite an existing .aimodel asset at the output path.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic input shapes.",
    )
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_pvt(
        output_dir,
        args.model,
        dtype,
        args.overwrite,
        args.dynamic,
    )


if __name__ == "__main__":
    main()
