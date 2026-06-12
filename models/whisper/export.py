# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
#     "coreai-torch==0.4.0",
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


class WhisperModule(torch.nn.Module):
    def __init__(self, model_name: str, dtype: torch.dtype):
        super().__init__()
        self._model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )

    def forward(self, input_features, decoder_input_ids):
        outputs = self._model(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        )
        return outputs.logits


def reference_inputs(model_name: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    # 5 seconds of 16kHz mono audio; the feature extractor pads/trims to 30s.
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32)
    feature = processor.feature_extractor(dummy_audio, sampling_rate=16000)
    return {
        "input_features": torch.tensor(feature["input_features"]).to(dtype),
        # Whisper's <|startoftranscript|> token.
        "decoder_input_ids": torch.tensor([[50258]], dtype=torch.int32),
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
    # Source: https://huggingface.co/openai/whisper-large-v3
    metadata = AIModelAssetMetadata()
    metadata.author = "A. Radford et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = "Whisper is an automatic speech recognition (ASR) encoder-decoder model from OpenAI, trained on a large multilingual and multitask supervised dataset. Source: https://huggingface.co/openai/whisper-large-v3"
    metadata.creation_date = int(time.time())
    return metadata


def create_whisper(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
):
    print("[INFO] Sourcing model...")
    model = WhisperModule(model_name, dtype)
    model.eval()
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(model_name, dtype)

    example_inputs["decoder_input_ids"] = torch.tensor(
        [[50258, 50259, 50360, 50364]], dtype=torch.int32
    )
    dynamic_shapes = {
        "input_features": {},
        "decoder_input_ids": {1: torch.export.Dim("dec_seq_len", min=1, max=448)},
    }
    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            model, args=(), kwargs=example_inputs, dynamic_shapes=dynamic_shapes
        )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_features", "decoder_input_ids"],
        output_names=["logits"],
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
        description="Create and save a Core AI AIProgram for Whisper."
    )
    parser.add_argument(
        "--model",
        choices=["openai/whisper-large-v3-turbo", "openai/whisper-large-v3"],
        default="openai/whisper-large-v3-turbo",
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
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_whisper(output_dir, args.model, dtype, args.overwrite)


if __name__ == "__main__":
    main()
