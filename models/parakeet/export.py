# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
#     "coreai-torch==0.4.0",
#     "transformers[audio]>=5.9.0,<5.10.1",
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

import numpy as np
import torch
import transformers
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table


# Parakeet TDT exports as three separate graphs because the autoregressive
# transducer loop (encoder frame pointer + (token, duration) sampling) lives
# in ParakeetTDTGenerationMixin, not in `forward`, and torch.export cannot
# capture it. The Swift runtime drives the loop and calls each graph in turn.
#
# 1. encoder      : (B, T_audio, n_mels) -> (B, T_enc, decoder_hidden_size)
#                   Includes the FastConformer encoder + the encoder_projector
#                   linear, so the joint network's decoder/encoder addends are
#                   already in the same hidden_size.
# 2. decoder_step : (input_ids, h, c) -> (decoder_out, new_h, new_c)
#                   One LSTM step of the prediction network, stateless. The
#                   runtime owns the LSTM state and seeds it with zeros on a
#                   new utterance / resets after a non-blank emission per the
#                   TDT decoding rules.
# 3. joint        : (decoder_hidden, encoder_hidden) -> logits[vocab+durations]
#                   Single-frame fuse: activation(enc + dec) -> linear head.

ENCODER_GRAPH = "encoder"
DECODER_STEP_GRAPH = "decoder_step"
JOINT_GRAPH = "joint"


class ParakeetEncoderModule(torch.nn.Module):
    """FastConformer encoder + encoder_projector linear."""

    def __init__(self, model: "transformers.ParakeetForTDT"):
        super().__init__()
        self._encoder = model.encoder
        self._encoder_projector = model.encoder_projector

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        outputs = self._encoder(
            input_features=input_features,
            output_attention_mask=False,
        )
        return self._encoder_projector(outputs.last_hidden_state)


class ParakeetDecoderStepModule(torch.nn.Module):
    """Single autoregressive step of the LSTM prediction network.

    The HF `ParakeetTDTDecoder` mutates a `ParakeetTDTDecoderCache` object;
    we expose the LSTM state explicitly as input/output tensors so the graph
    is stateless and the Swift side owns the cache.
    """

    def __init__(self, model: "transformers.ParakeetForTDT"):
        super().__init__()
        self._embedding = model.decoder.embedding
        self._lstm = model.decoder.lstm
        self._projector = model.decoder.decoder_projector

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self._embedding(input_ids)
        lstm_output, (new_hidden, new_cell) = self._lstm(
            embeddings, (hidden_state, cell_state)
        )
        decoder_output = self._projector(lstm_output)
        return decoder_output, new_hidden, new_cell


class ParakeetJointModule(torch.nn.Module):
    """Joint network: activation(enc + dec) -> linear over vocab+durations."""

    def __init__(self, model: "transformers.ParakeetForTDT"):
        super().__init__()
        self._joint = model.joint

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._joint(
            decoder_hidden_states=decoder_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )


def _audio_features(
    model_name: str, dtype: torch.dtype, seconds: float
) -> torch.Tensor:
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    sample_rate = processor.feature_extractor.sampling_rate
    dummy_audio = np.random.randn(int(sample_rate * seconds)).astype(np.float32)
    features = processor.feature_extractor(dummy_audio, sampling_rate=sample_rate)
    return torch.tensor(features["input_features"]).to(dtype)


def _decoder_step_inputs(
    config: "transformers.ParakeetTDTConfig", dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    batch = 1
    return {
        "input_ids": torch.zeros((batch, 1), dtype=torch.int32),
        "hidden_state": torch.zeros(
            config.num_decoder_layers, batch, config.decoder_hidden_size, dtype=dtype
        ),
        "cell_state": torch.zeros(
            config.num_decoder_layers, batch, config.decoder_hidden_size, dtype=dtype
        ),
    }


def _joint_inputs(
    config: "transformers.ParakeetTDTConfig", dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    batch = 1
    hidden = config.decoder_hidden_size
    return {
        "decoder_hidden_states": torch.zeros(batch, 1, hidden, dtype=dtype),
        "encoder_hidden_states": torch.zeros(batch, 1, hidden, dtype=dtype),
    }


def _encoder_dynamic_shapes() -> dict:
    """Allow variable audio length when --dynamic is set; batch stays at 1.

    Feature-extractor output is (B, T_audio, n_mels), so the time axis is 1.
    n_mels (axis 2) is fixed by the checkpoint (128 for v3) and stays static.
    """
    return {"input_features": {1: torch.export.Dim.DYNAMIC}}


def _convert(
    module: torch.nn.Module,
    example_inputs: dict[str, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    dtype: torch.dtype,
    dynamic_shapes: dict | None = None,
):
    module.eval()
    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(
            module,
            args=(),
            kwargs=example_inputs,
            dynamic_shapes=dynamic_shapes,
        )
    exported = exported.run_decompositions(get_decomp_table())
    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=input_names,
        output_names=output_names,
    )
    program = converter.to_coreai()
    program.optimize()
    return program


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "exports")


def _variant_name(model_name: str, dtype: torch.dtype, dynamic: bool) -> str:
    safe_name = Path(model_name).name
    dtype_name = str(dtype).split(".")[-1]
    static_or_dynamic = "dynamic" if dynamic else "static"
    return f"{safe_name}_{dtype_name}_{static_or_dynamic}"


def _bundle_paths(
    output_dir: str, model_name: str, dtype: torch.dtype, dynamic: bool
) -> tuple[Path, dict[str, Path]]:
    variant = _variant_name(model_name, dtype, dynamic)
    bundle_dir = Path(output_dir) / variant
    assets = {
        ENCODER_GRAPH: bundle_dir / f"{variant}_{ENCODER_GRAPH}.aimodel",
        DECODER_STEP_GRAPH: bundle_dir / f"{variant}_{DECODER_STEP_GRAPH}.aimodel",
        JOINT_GRAPH: bundle_dir / f"{variant}_{JOINT_GRAPH}.aimodel",
    }
    return bundle_dir, assets


def _build_aimodel_metadata(graph: str) -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "K. C. Puvvada et al."
    metadata.license = "CC-BY-4.0"
    metadata.model_description = (
        f"Parakeet-TDT v3 ASR ({graph} subgraph). Parakeet is a FastConformer "
        f"encoder paired with a Token-and-Duration Transducer decoder that "
        f"predicts (token, duration) pairs for blank-skipping greedy decoding. "
        f"Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"
    )
    metadata.creation_date = int(time.time())
    return metadata


def _save_program(program, model_path: Path, graph: str) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(model_path, _build_aimodel_metadata(graph))
    print(f"[INFO] Saved {graph} graph to {model_path}.")


def _prepare_bundle_dir(bundle_dir: Path, overwrite: bool) -> None:
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{bundle_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)


def _write_processor(dest: Path, model_name: str) -> None:
    print(
        f"[INFO] Saving processor (feature extractor + tokenizer) from {model_name} to {dest}..."
    )
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(str(dest))


def _write_bundle_metadata(
    bundle_dir: Path,
    variant: str,
    config: "transformers.ParakeetTDTConfig",
    assets: dict[str, Path],
) -> None:
    metadata = {
        "metadata_version": "0.2",
        "kind": "speech_recognizer_tdt",
        "name": variant,
        "assets": {graph: path.name for graph, path in assets.items()},
        "config": {
            "vocab_size": config.vocab_size,
            "blank_token_id": config.blank_token_id,
            "decoder_hidden_size": config.decoder_hidden_size,
            "num_decoder_layers": config.num_decoder_layers,
            "max_symbols_per_step": config.max_symbols_per_step,
            "durations": list(config.durations),
            "encoder": {
                "num_mel_bins": config.encoder_config.num_mel_bins,
                "subsampling_factor": config.encoder_config.subsampling_factor,
            },
        },
    }
    metadata_path = bundle_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote bundle metadata to {metadata_path}.")


def create_parakeet(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    dynamic: bool,
    audio_seconds: float,
):
    print(f"[INFO] Sourcing {model_name}...")
    model = transformers.AutoModelForTDT.from_pretrained(model_name)
    model.eval()
    model.to(dtype)
    config = model.config
    print(
        f"[INFO] Loaded ParakeetForTDT — encoder hidden={config.encoder_config.hidden_size}, "
        f"decoder hidden={config.decoder_hidden_size}, vocab={config.vocab_size}, "
        f"durations={list(config.durations)}."
    )

    bundle_dir, assets = _bundle_paths(output_dir, model_name, dtype, dynamic)
    _prepare_bundle_dir(bundle_dir, overwrite)

    print(f"[INFO] Exporting {ENCODER_GRAPH} graph...")
    encoder_inputs = {
        "input_features": _audio_features(model_name, dtype, audio_seconds)
    }
    encoder_program = _convert(
        ParakeetEncoderModule(model),
        encoder_inputs,
        input_names=["input_features"],
        output_names=["encoder_hidden_states"],
        dtype=dtype,
        dynamic_shapes=_encoder_dynamic_shapes() if dynamic else None,
    )
    _save_program(encoder_program, assets[ENCODER_GRAPH], ENCODER_GRAPH)

    print(f"[INFO] Exporting {DECODER_STEP_GRAPH} graph...")
    decoder_program = _convert(
        ParakeetDecoderStepModule(model),
        _decoder_step_inputs(config, dtype),
        input_names=["input_ids", "hidden_state", "cell_state"],
        output_names=["decoder_output", "new_hidden_state", "new_cell_state"],
        dtype=dtype,
    )
    _save_program(decoder_program, assets[DECODER_STEP_GRAPH], DECODER_STEP_GRAPH)

    print(f"[INFO] Exporting {JOINT_GRAPH} graph...")
    joint_program = _convert(
        ParakeetJointModule(model),
        _joint_inputs(config, dtype),
        input_names=["decoder_hidden_states", "encoder_hidden_states"],
        output_names=["logits"],
        dtype=dtype,
    )
    _save_program(joint_program, assets[JOINT_GRAPH], JOINT_GRAPH)

    _write_processor(bundle_dir / "processor", model_name)
    _write_bundle_metadata(
        bundle_dir, _variant_name(model_name, dtype, dynamic), config, assets
    )
    print(f"[INFO] Successfully created Parakeet TDT bundle at {bundle_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export NVIDIA Parakeet TDT to Core AI. Produces a bundle directory "
            "containing three .aimodel assets (encoder, decoder_step, joint) "
            "plus the processor and bundle metadata."
        )
    )
    parser.add_argument(
        "--model",
        choices=["nvidia/parakeet-tdt-0.6b-v3"],
        default="nvidia/parakeet-tdt-0.6b-v3",
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
        help="Overwrite an existing bundle at the output path.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export the encoder with dynamic audio length (decoder/joint stay static).",
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=5.0,
        help=(
            "Length (seconds) of dummy audio used to shape the encoder's static "
            "trace. Ignored when --dynamic is set."
        ),
    )
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_parakeet(
        output_dir,
        args.model,
        dtype,
        args.overwrite,
        args.dynamic,
        args.audio_seconds,
    )


if __name__ == "__main__":
    main()
