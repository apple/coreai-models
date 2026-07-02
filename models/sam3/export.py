# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "tokenizers<0.23.0rc",
#     "torchvision",
#     "transformers>=5.5.4,<5.10.1",
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
import transformers
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table


def reference_inputs(model_name: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    processor = transformers.Sam3Processor.from_pretrained(model_name)
    text_inputs = processor.tokenizer(["dummy"], return_tensors="pt")
    return {
        "pixel_values": torch.randn(1, 3, 1008, 1008).to(dtype),
        "input_ids": text_inputs["input_ids"].to(torch.int32),
    }


class Sam3Module(torch.nn.Module):
    def __init__(self, model_id: str = "facebook/sam3"):
        super().__init__()
        self._model = transformers.Sam3Model.from_pretrained(model_id)

    def forward(self, pixel_values, input_ids):
        outputs = self._model(pixel_values=pixel_values, input_ids=input_ids)
        return (
            outputs.pred_masks,
            outputs.pred_boxes,
            outputs.pred_logits,
            outputs.presence_logits,
            outputs.semantic_seg,
        )


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "exports")


def _variant_name(model_name: str, dtype: torch.dtype) -> str:
    safe_name = Path(model_name).name
    dtype_name = str(dtype).split(".")[-1]
    return f"{safe_name}_{dtype_name}"


def _bundle_paths(
    output_dir: str, model_name: str, dtype: torch.dtype
) -> tuple[Path, Path]:
    """Return (bundle_dir, model_path) where the .aimodel sits inside the bundle dir."""
    variant = _variant_name(model_name, dtype)
    bundle_dir = Path(output_dir) / variant
    return bundle_dir, bundle_dir / f"{variant}.aimodel"


def _build_aimodel_metadata() -> AIModelAssetMetadata:
    # Source: https://github.com/facebookresearch/sam3
    metadata = AIModelAssetMetadata()
    metadata.author = "N. Carion et al."
    metadata.license = "SAM License"
    metadata.model_description = "SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. This variant is explicitly for image segmentation. Source: https://github.com/facebookresearch/sam3"
    metadata.creation_date = int(time.time())
    return metadata


def _save_asset(coreai_program, model_path: Path, overwrite: bool) -> None:
    bundle_dir = model_path.parent
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{bundle_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    coreai_program.save_asset(model_path, _build_aimodel_metadata())


def _write_tokenizer(dest: Path, model_id: str) -> None:
    print(f"[INFO] Saving tokenizer from {model_id} to {dest}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(dest))


def _write_bundle_metadata(bundle_dir: Path, variant: str) -> None:
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


def create_sam3(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
):
    print("[INFO] Sourcing model...")
    model = Sam3Module(model_id=model_name)
    model.eval()
    model.to(dtype)
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(model_name, dtype)

    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            model,
            args=(),
            kwargs=example_inputs,
        )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["pixel_values", "input_ids"],
        output_names=[
            "pred_masks",
            "pred_boxes",
            "pred_logits",
            "presence_logits",
            "semantic_seg",
        ],
    )
    coreai_program = converter.to_coreai()
    print("[INFO] Model converted.")
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    bundle_dir, model_path = _bundle_paths(output_dir, model_name, dtype)
    _save_asset(coreai_program, model_path, overwrite)
    print(f"[INFO] Successfully created and saved Core AI model to {model_path}.")

    _write_tokenizer(bundle_dir / "tokenizer", model_name)
    _write_bundle_metadata(bundle_dir, _variant_name(model_name, dtype))


def main():
    parser = argparse.ArgumentParser(
        description="Create and save a Core AI AIProgram for SAM3."
    )
    parser.add_argument(
        "--model",
        choices=["facebook/sam3"],
        default="facebook/sam3",
        help="Model variant to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the .aimodel bundle (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
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
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_sam3(output_dir, args.model, dtype, args.overwrite)


if __name__ == "__main__":
    main()
