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

import numpy as np
import torch
import transformers
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table


class ClapModule(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self._model = transformers.ClapModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, input_features, is_longer):
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            is_longer=is_longer,
        )
        return (
            outputs.logits_per_audio,
            outputs.logits_per_text,
            outputs.text_embeds,
            outputs.audio_embeds,
        )


def reference_inputs(model_name: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    input_text = ["Sound of a dog", "Sound of vacuum cleaner"]
    # CLAP's feature extractor expects 48kHz mono audio as a numpy array.
    audio_sample = np.random.randn(48000).astype(np.float32)
    features = processor(
        text=input_text,
        audio=audio_sample,
        sampling_rate=48000,
        return_tensors="pt",
        padding=True,
    )
    return {
        "input_ids": features["input_ids"].to(torch.int32),
        "attention_mask": features["attention_mask"].to(torch.int32),
        "input_features": features["input_features"].to(dtype),
        "is_longer": features["is_longer"],
    }


def dynamic_shapes() -> dict:
    """Dynamic shape specification for CLAP.

    Static (default): input_ids and attention_mask are (2, N) where N is
    the padded token length; input_features and is_longer are fixed by the
    audio processor.
    Dynamic (--dynamic): text batch (1-64) can vary; audio inputs and
    sequence lengths stay fixed.
    """
    batch = torch.export.Dim("batch_size", min=1, max=64)
    return {
        "input_ids": {0: batch},
        "attention_mask": {0: batch},
        "input_features": {},
        "is_longer": {},
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
    # Source: https://huggingface.co/laion/clap-htsat-unfused
    metadata = AIModelAssetMetadata()
    metadata.author = "Y. Wu et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = "CLAP (Contrastive Language-Audio Pretraining) learns joint representations of audio and text, enabling zero-shot audio classification with natural language labels. Source: https://huggingface.co/laion/clap-htsat-unfused"
    metadata.creation_date = int(time.time())
    return metadata


def create_clap(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    dynamic: bool,
):
    print("[INFO] Sourcing model...")
    model = ClapModule(model_name)
    model.eval()
    model.to(dtype)
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(model_name, dtype)
    ds = dynamic_shapes() if dynamic else None

    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            model, args=(), kwargs=example_inputs, dynamic_shapes=ds
        )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "attention_mask", "input_features", "is_longer"],
        output_names=[
            "logits_per_audio",
            "logits_per_text",
            "text_embeds",
            "audio_embeds",
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
        description="Create and save a Core AI AIProgram for CLAP."
    )
    parser.add_argument(
        "--model",
        choices=["laion/clap-htsat-unfused"],
        default="laion/clap-htsat-unfused",
        help="Model variant to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the .aimodel asset (default: <repo-root>/exports/)",
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
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic batch size.",
    )
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_clap(
        output_dir,
        args.model,
        dtype,
        args.overwrite,
        args.dynamic,
    )


if __name__ == "__main__":
    main()
