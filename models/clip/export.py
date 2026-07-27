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


class ClipModule(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self._model = transformers.CLIPModel.from_pretrained(model_name)
        self.vision_model = self._model.vision_model
        self.text_model = self._model.text_model
        self.visual_projection = self._model.visual_projection
        self.text_projection = self._model.text_projection

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self._model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        return logits_per_image, logits_per_text, image_embeds, text_embeds


def reference_inputs(
    model_name: str, dtype: torch.dtype, dynamic: bool = False
) -> dict[str, torch.Tensor]:
    tokenizer = transformers.CLIPTokenizer.from_pretrained(model_name)
    text_inputs = tokenizer(
        ["a photo of a cat", "a photo of a dog", "a photo of a goat"],
        return_tensors="pt",
        padding=True,
    )
    B = 2 if dynamic else 1
    return {
        "pixel_values": torch.randn(B, 3, 224, 224).to(dtype),
        "input_ids": text_inputs["input_ids"].to(torch.int32),
        "attention_mask": text_inputs["attention_mask"].to(torch.int32),
    }


def dynamic_shapes() -> dict:
    """Dynamic shape specification for CLIP.

    Static (default): pixel_values is (1, 3, 224, 224), input_ids and
    attention_mask are (2, 77) — tokenizer always pads to 77.
    Dynamic (--dynamic): image batch (1-64) and text batch (1-64) can
    vary independently; spatial and sequence dims are fixed.
    """
    image_batch = torch.export.Dim("image_batch", min=1, max=64)
    text_batch = torch.export.Dim("text_batch", min=1, max=64)
    return {
        "pixel_values": {0: image_batch},
        "input_ids": {0: text_batch},
        "attention_mask": {0: text_batch},
    }


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
    # Source: https://huggingface.co/openai/clip-vit-base-patch32
    metadata = AIModelAssetMetadata()
    metadata.author = "A. Radford et al."
    metadata.license = "MIT"
    metadata.model_description = "CLIP (Contrastive Language-Image Pretraining) learns joint representations of images and text, enabling zero-shot image classification with natural language labels. Source: https://huggingface.co/openai/clip-vit-base-patch32"
    metadata.creation_date = int(time.time())
    return metadata


def create_clip(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    dynamic: bool,
):
    print("[INFO] Sourcing model...")
    model = ClipModule(model_name)
    model.eval()
    model.to(dtype)
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(model_name, dtype, dynamic)
    ds = dynamic_shapes() if dynamic else None

    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            model, args=(), kwargs=example_inputs, dynamic_shapes=ds
        )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["pixel_values", "input_ids", "attention_mask"],
        output_names=[
            "logits_per_image",
            "logits_per_text",
            "image_embeds",
            "text_embeds",
        ],
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
        description="Create and save a Core AI AIProgram for CLIP."
    )
    parser.add_argument(
        "--model",
        choices=["openai/clip-vit-base-patch32"],
        default="openai/clip-vit-base-patch32",
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
        help="Export with dynamic batch size.",
    )
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_clip(
        output_dir,
        args.model,
        dtype,
        args.overwrite,
        args.dynamic,
    )


if __name__ == "__main__":
    main()
