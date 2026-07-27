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

MAX_IOS_LENGTH = 4096


class T5Module(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_ids, decoder_input_ids):
        outputs = self._model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        return outputs["logits"], outputs["encoder_last_hidden_state"]


def reference_inputs(
    model_name: str, dtype: torch.dtype, dynamic: bool = False
) -> dict[str, torch.Tensor]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer("The weather is nice today", return_tensors="pt")["input_ids"]
    decoder_input_ids = tokenizer("Il fait beau aujourd'hui", return_tensors="pt")[
        "input_ids"
    ]
    inputs = {
        "input_ids": input_ids.to(torch.int32),
        "decoder_input_ids": decoder_input_ids.to(torch.int32),
    }
    if dynamic:
        for k, v in inputs.items():
            inputs[k] = v.expand(2, *[-1] * (v.ndim - 1)).contiguous()
    return inputs


def dynamic_shapes(dtype: torch.dtype) -> dict:
    """Dynamic shape specification for T5.

    Static (default): both sequence dims are fixed to the reference
    tokenization lengths (input_ids ~ 7 tokens, decoder_input_ids ~ 9),
    batch is 1.
    Dynamic (--dynamic): batch (1-64) and sequence dims can vary.
      f16 (ANE): sequence capped at 4096 tokens.
      f32 (GPU): sequence unconstrained.
    """
    batch = torch.export.Dim("batch_size", min=1, max=64)
    if dtype == torch.float32:
        return {
            "input_ids": {0: batch, 1: torch.export.Dim("input_len", min=1)},
            "decoder_input_ids": {
                0: batch,
                1: torch.export.Dim("output_len", min=1),
            },
        }
    return {
        "input_ids": {
            0: batch,
            1: torch.export.Dim("input_len", min=1, max=MAX_IOS_LENGTH),
        },
        "decoder_input_ids": {
            0: batch,
            1: torch.export.Dim("output_len", min=1, max=MAX_IOS_LENGTH),
        },
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
    # Source: https://huggingface.co/docs/transformers/model_doc/t5
    metadata = AIModelAssetMetadata()
    metadata.author = "C. Raffel et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = "T5 (Text-to-Text Transfer Transformer) is an encoder-decoder model pre-trained on a mixture of unsupervised and supervised tasks. Works well on many tasks via input prefixes (e.g. 'translate English to German: ...', 'summarize: ...'). Source: https://huggingface.co/docs/transformers/model_doc/t5"
    metadata.creation_date = int(time.time())
    return metadata


def create_t5(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    dynamic: bool,
):
    print("[INFO] Sourcing model...")
    model = T5Module(model_name)
    model.eval()
    model.to(dtype)
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(model_name, dtype, dynamic)
    ds = dynamic_shapes(dtype) if dynamic else None

    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            model, args=(), kwargs=example_inputs, dynamic_shapes=ds
        )
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "decoder_input_ids"],
        output_names=["logits", "encoder_last_hidden_state"],
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
        description="Create and save a Core AI AIProgram for T5."
    )
    parser.add_argument(
        "--model",
        choices=["google-t5/t5-small", "google-t5/t5-base", "google-t5/t5-large"],
        default="google-t5/t5-small",
        help="Model variant to convert (e.g. t5-small, t5-base, google/flan-t5-base).",
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
        help="Export with dynamic input shapes.",
    )
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_t5(
        output_dir,
        args.model,
        dtype,
        args.overwrite,
        args.dynamic,
    )


if __name__ == "__main__":
    main()
