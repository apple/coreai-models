# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "transformers==4.57.3",
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

import torch
import transformers
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table


class YolosModule(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self._model = transformers.AutoModelForObjectDetection.from_pretrained(
            model_name
        )

    def forward(self, pixel_values):
        outputs = self._model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes, outputs.last_hidden_state


def reference_inputs(
    dtype: torch.dtype, model_name: str, dynamic: bool = False
) -> dict[str, torch.Tensor]:
    B = 2 if dynamic else 1
    processor = transformers.AutoImageProcessor.from_pretrained(model_name)
    size = processor.size["shortest_edge"]
    return {"pixel_values": torch.randn(B, 3, size, size).to(dtype)}


def dynamic_shapes() -> dict:
    """Dynamic shape specification for YOLO.

    Static (default): pixel_values is fixed at (1, 3, 800, 800).
    Dynamic (--dynamic): batch (1-64) can vary; spatial dims must be
    multiples of 16
    """
    batch = torch.export.Dim("batch_size", min=1, max=64)
    _height = torch.export.Dim("height", min=8, max=64)
    _width = torch.export.Dim("width", min=8, max=64)
    height = _height * 16
    width = _width * 16
    return {"pixel_values": {0: batch, 2: height, 3: width}}


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
    # Source: https://huggingface.co/hustvl/yolos-tiny
    metadata = AIModelAssetMetadata()
    metadata.author = "Y. Fang et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = "YOLOS (You Only Look at One Sequence) applies a plain Vision Transformer directly to image patches and predicts object queries as bounding boxes and class logits. Source: https://huggingface.co/hustvl/yolos-tiny"
    metadata.creation_date = int(time.time())
    return metadata


def create_yolos(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    dynamic: bool,
):
    print("[INFO] Sourcing model...")
    model = YolosModule(model_name)
    model.eval()
    model.to(dtype)
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(dtype, model_name, dynamic)
    ds = dynamic_shapes() if dynamic else None

    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            model, args=(), kwargs=example_inputs, dynamic_shapes=ds
        )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes", "last_hidden_state"],
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
        description="Create and save a Core AI AIProgram for YOLOS."
    )
    parser.add_argument(
        "--model",
        choices=["hustvl/yolos-tiny", "hustvl/yolos-base"],
        default="hustvl/yolos-base",
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
    create_yolos(
        output_dir,
        args.model,
        dtype,
        args.overwrite,
        args.dynamic,
    )


if __name__ == "__main__":
    main()
