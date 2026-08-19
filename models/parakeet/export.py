# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "transformers[audio]>=5.9.0,<5.10.1",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
import argparse
import dataclasses
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

    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # attention_mask is a (B, T_audio) bool mask marking real-audio frames.
        # It makes the encoder exclude padding from self-attention *and* the
        # subsampling / conformer conv modules (matching HF). Without it a
        # fixed-window static export attends over its zero padding and degrades.
        outputs = self._encoder(
            input_features=input_features,
            attention_mask=attention_mask,
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


def _audio_features_samples(
    processor: "transformers.ProcessorMixin", dtype: torch.dtype, num_samples: int
) -> torch.Tensor:
    sample_rate = processor.feature_extractor.sampling_rate
    dummy_audio = np.random.randn(num_samples).astype(np.float32)
    features = processor.feature_extractor(dummy_audio, sampling_rate=sample_rate)
    return features["input_features"].to(dtype).detach().clone()


def _audio_features(
    processor: "transformers.ProcessorMixin", dtype: torch.dtype, seconds: float
) -> torch.Tensor:
    sample_rate = processor.feature_extractor.sampling_rate
    return _audio_features_samples(processor, dtype, int(sample_rate * seconds))


def _encoder_frame_count(mel_frames: int, subsampling_factor: int) -> int:
    """Encoder frames emitted for `mel_frames`, applying the subsampling stack stage by stage.

    Each stride-2, kernel-3, pad-1 conv maps `T` to `floor((T - 1) / 2) + 1`, so the count follows
    from halving once per factor of two rather than from the `ceil(L/8)` closed form. Mirrors
    `encoderFrameCount` in StreamingWindow.swift and HF
    `ParakeetPreTrainedModel._get_subsampling_output_length`, so the export, the simulator and the
    runtime cannot disagree about the same quantity.

    `subsampling_factor` must be a power of two, which is all a stack of stride-2 convs can
    express — the loop halves, so a factor of 6 would silently behave as 4.
    """
    if mel_frames <= 0 or subsampling_factor <= 1:
        return max(0, mel_frames)
    if subsampling_factor & (subsampling_factor - 1) != 0:
        raise ValueError(
            f"subsampling_factor must be a power of two, got {subsampling_factor}"
        )
    length, factor = mel_frames, subsampling_factor
    while factor > 1:
        length = (length - 1) // 2 + 1
        factor //= 2
    return length


@dataclasses.dataclass(frozen=True)
class StreamingWindowArgs:
    """The three knobs that size a streaming window, in encoder frames.

    `None` rather than a `streaming=False` flag is what makes "not streaming" unable to carry
    window values nothing reads.
    """

    left_context_frames: int = 126
    chunk_frames: int = 12
    right_context_frames: int = 12


def _streaming_geometry(
    processor: "transformers.ProcessorMixin",
    config: "transformers.ParakeetTDTConfig",
    window: StreamingWindowArgs,
) -> dict:
    """Window geometry for a streaming encoder export, in exact integers.

    Everything hangs off one rule: make the PCM window a whole number of encoder
    frames. The feature extractor emits `1 + N/hop` frames (torch.stft
    center=True), and each subsampling conv maps `T` to `floor((T - 1) / 2) + 1` — see
    `_encoder_frame_count`, which is the definition; `ceil(L/8)` is only its consequence for
    a factor of 8. So a window of `W * hop * subsampling` samples gives `8W + 1` mel frames and
    `W + 1` encoder frames, of which `W` are fully backed by real audio and the last covers the
    zero-padded remainder. The `8W + 1` identity is asserted below rather than assumed.

    Deriving the sample count from `seconds` instead would be lossy at exactly the
    lengths we care about: `16000 * 6.4 == 102400.00000000001`.
    """
    extractor = processor.feature_extractor
    sample_rate = extractor.sampling_rate
    hop_length = extractor.hop_length
    subsampling = config.encoder_config.subsampling_factor

    left, chunk, right = (
        window.left_context_frames,
        window.chunk_frames,
        window.right_context_frames,
    )
    usable = left + chunk + right
    samples_per_encoder_frame = hop_length * subsampling
    window_samples = usable * samples_per_encoder_frame
    window_mel_frames = usable * subsampling + 1

    # The whole window arithmetic rests on a frame-aligned window: the mel frames backed by real
    # audio must subsample to exactly `usable`. Check it rather than trusting the closed form.
    valid_mel_frames = window_mel_frames - 1
    recovered = _encoder_frame_count(valid_mel_frames, subsampling)
    if recovered != usable:
        raise ValueError(
            f"window of {usable} encoder frames gives {valid_mel_frames} valid mel frames, "
            f"which subsample to {recovered}, not {usable}"
        )

    return {
        "left_context_encoder_frames": left,
        "chunk_encoder_frames": chunk,
        "right_context_encoder_frames": right,
        "usable_encoder_frames": usable,
        "window_encoder_frames": _encoder_frame_count(window_mel_frames, subsampling),
        "window_mel_frames": window_mel_frames,
        "window_sample_count": window_samples,
        "seconds_per_encoder_frame": samples_per_encoder_frame / sample_rate,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "subsampling_factor": subsampling,
    }


# Keys the Swift runtime reads (StreamingConfig.StreamingBlock). Everything else
# `_streaming_geometry` computes is for this script's own use — naming the bundle, sizing the
# dummy input, the forward-pass assertions, the log line — and is deliberately not published:
# a derived value in the file is one more thing that can contradict the traced graph.
_RECORDED_GEOMETRY_KEYS = (
    "left_context_encoder_frames",
    "chunk_encoder_frames",
    "right_context_encoder_frames",
    "window_mel_frames",
    "sample_rate",
    "hop_length",
    "subsampling_factor",
)


def _recorded_geometry(geometry: dict) -> dict:
    """The subset of the geometry a bundle records, in the order above."""
    return {key: geometry[key] for key in _RECORDED_GEOMETRY_KEYS}


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
    The attention_mask shares that same time axis; DYNAMIC lets the exporter
    unify the two dims (they must be equal).
    """
    return {
        "input_features": {1: torch.export.Dim.DYNAMIC},
        "attention_mask": {1: torch.export.Dim.DYNAMIC},
    }


def _convert(
    module: torch.nn.Module,
    example_inputs: dict[str, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    dtype: torch.dtype,
    include_debug_info: bool,
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
    mode = (
        TorchConverter.Mode.DEBUG if include_debug_info else TorchConverter.Mode.RELEASE
    )
    converter = TorchConverter(mode=mode).add_exported_program(
        exported_program=exported,
        input_names=input_names,
        output_names=output_names,
    )
    program = converter.to_coreai()
    program.optimize()
    return program


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "exports")


def _variant_name(
    model_name: str,
    dtype: torch.dtype,
    dynamic: bool,
    streaming: dict | None = None,
) -> str:
    safe_name = Path(model_name).name
    dtype_name = str(dtype).split(".")[-1]
    if streaming is not None:
        # The usable frame count is what distinguishes one streaming window from
        # another, so it belongs in the name.
        kind = f"streaming{streaming['usable_encoder_frames']}"
    else:
        kind = "dynamic" if dynamic else "static"
    return f"{safe_name}_{dtype_name}_{kind}"


def _bundle_paths(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    dynamic: bool,
    streaming: dict | None = None,
) -> tuple[Path, dict[str, Path]]:
    variant = _variant_name(model_name, dtype, dynamic, streaming)
    bundle_dir = Path(output_dir) / variant
    assets = {
        ENCODER_GRAPH: bundle_dir / f"{variant}_{ENCODER_GRAPH}.aimodel",
        DECODER_STEP_GRAPH: bundle_dir / f"{variant}_{DECODER_STEP_GRAPH}.aimodel",
        JOINT_GRAPH: bundle_dir / f"{variant}_{JOINT_GRAPH}.aimodel",
    }
    return bundle_dir, assets


def _build_aimodel_metadata(graph: str) -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "M. Sekoyan et al."
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


def _write_processor(
    dest: Path, processor: "transformers.ProcessorMixin", model_name: str
) -> None:
    print(
        f"[INFO] Saving processor (feature extractor + tokenizer) from {model_name} to {dest}..."
    )
    processor.save_pretrained(str(dest))


def _write_bundle_metadata(
    bundle_dir: Path,
    variant: str,
    config: "transformers.ParakeetTDTConfig",
    assets: dict[str, Path],
    streaming: dict | None = None,
) -> None:
    metadata = {
        "metadata_version": "0.2",
        "kind": "speech_recognizer",
        "name": variant,
        "assets": {graph: path.name for graph, path in assets.items()},
        "config": {
            "architecture": "parakeet_tdt",
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
    if streaming is not None:
        # A sibling of `config`, not a member of it, so ParakeetTDTConfig.decode on
        # the Swift side is untouched and existing bundles keep decoding. Note
        # metadata_version stays "0.2": ModelBundle hard-rejects anything else.
        metadata["streaming"] = _recorded_geometry(streaming)
    metadata_path = bundle_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote bundle metadata to {metadata_path}.")


def _encoder_inputs(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "input_features": features,
        # All-valid mask for the trace; the Swift runtime supplies the real
        # per-frame mask (1 for real audio, 0 for the static window's padding).
        "attention_mask": torch.ones(features.shape[:2], dtype=torch.bool),
    }


def _streaming_encoder_inputs(
    processor: "transformers.ProcessorMixin",
    model: "transformers.ParakeetForTDT",
    dtype: torch.dtype,
    geometry: dict,
) -> dict[str, torch.Tensor]:
    """Trace inputs for a streaming window, checked against the geometry that sized them.

    Catches an arithmetic error here in Python rather than six files later in Swift: a
    one-frame slice error is 80 ms of audio and would drop or duplicate words at every chunk
    boundary. Costs one forward pass.
    """
    features = _audio_features_samples(
        processor, dtype, geometry["window_sample_count"]
    )
    if features.shape[1] != geometry["window_mel_frames"]:
        raise ValueError(
            f"streaming geometry mismatch: {geometry['window_sample_count']} samples "
            f"produced {features.shape[1]} mel frames, expected "
            f"{geometry['window_mel_frames']}"
        )
    inputs = _encoder_inputs(features)
    with torch.no_grad():
        probe = ParakeetEncoderModule(model)(**inputs)
    if probe.shape[1] != geometry["window_encoder_frames"]:
        raise ValueError(
            f"streaming geometry mismatch: traced encoder emits {probe.shape[1]} "
            f"frames, expected {geometry['window_encoder_frames']}"
        )
    print(
        f"[INFO] Verified encoder emits {probe.shape[1]} frames "
        f"({geometry['usable_encoder_frames']} usable + 1 padding boundary)."
    )
    return inputs


def _log_streaming_window(geometry: dict) -> None:
    latency = (
        geometry["chunk_encoder_frames"] + geometry["right_context_encoder_frames"]
    ) * geometry["seconds_per_encoder_frame"]
    print(
        f"[INFO] Streaming window: left {geometry['left_context_encoder_frames']} / "
        f"chunk {geometry['chunk_encoder_frames']} / "
        f"right {geometry['right_context_encoder_frames']} encoder frames "
        f"({geometry['usable_encoder_frames']} usable) = "
        f"{geometry['window_sample_count']} samples "
        f"({geometry['window_sample_count'] / geometry['sample_rate']:.2f} s), "
        f"{geometry['window_mel_frames']} mel frames. Theoretical latency {latency:.2f} s."
    )


def create_parakeet(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    dynamic: bool,
    audio_seconds: float,
    include_debug_info: bool,
    window: StreamingWindowArgs | None = None,
):
    print(f"[INFO] Sourcing {model_name}...")
    model = transformers.AutoModelForTDT.from_pretrained(
        model_name, dtype=dtype, use_safetensors=True
    )
    model.eval()
    config = model.config
    print(
        f"[INFO] Loaded ParakeetForTDT — encoder hidden={config.encoder_config.hidden_size}, "
        f"decoder hidden={config.decoder_hidden_size}, vocab={config.vocab_size}, "
        f"durations={list(config.durations)}."
    )
    # One load, threaded through: it sizes the window, shapes the dummy input, and ships in
    # the bundle.
    processor = transformers.AutoProcessor.from_pretrained(model_name)

    geometry = None
    if window is not None:
        geometry = _streaming_geometry(processor, config, window)
        _log_streaming_window(geometry)

    bundle_dir, assets = _bundle_paths(output_dir, model_name, dtype, dynamic, geometry)
    _prepare_bundle_dir(bundle_dir, overwrite)

    print(f"[INFO] Exporting {ENCODER_GRAPH} graph...")
    if geometry is not None:
        encoder_inputs = _streaming_encoder_inputs(processor, model, dtype, geometry)
    else:
        encoder_inputs = _encoder_inputs(
            _audio_features(processor, dtype, audio_seconds)
        )

    encoder_program = _convert(
        ParakeetEncoderModule(model),
        encoder_inputs,
        input_names=["input_features", "attention_mask"],
        output_names=["encoder_hidden_states"],
        dtype=dtype,
        include_debug_info=include_debug_info,
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
        include_debug_info=include_debug_info,
    )
    _save_program(decoder_program, assets[DECODER_STEP_GRAPH], DECODER_STEP_GRAPH)

    print(f"[INFO] Exporting {JOINT_GRAPH} graph...")
    joint_program = _convert(
        ParakeetJointModule(model),
        _joint_inputs(config, dtype),
        input_names=["decoder_hidden_states", "encoder_hidden_states"],
        output_names=["logits"],
        dtype=dtype,
        include_debug_info=include_debug_info,
    )
    _save_program(joint_program, assets[JOINT_GRAPH], JOINT_GRAPH)

    _write_processor(bundle_dir / "processor", processor, model_name)
    _write_bundle_metadata(
        bundle_dir,
        _variant_name(model_name, dtype, dynamic, geometry),
        config,
        assets,
        geometry,
    )
    print(f"[INFO] Successfully created Parakeet TDT bundle at {bundle_dir}.")


_WINDOW_FLAGS = (
    ("--chunk-frames", "chunk_frames"),
    ("--right-context-frames", "right_context_frames"),
    ("--left-context-frames", "left_context_frames"),
)
_AUDIO_SECONDS_FLAG = ("--audio-seconds", "audio_seconds")


def _warn_ignored_shape_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Warn about window flags the chosen shape mode never reads.

    Each mode sizes the encoder trace a different way, and a flag belonging to
    another one is otherwise dropped in silence — the wrong window only shows up
    a full export later, in the bundle name.
    """
    if args.streaming:
        candidates = [_AUDIO_SECONDS_FLAG]
        reason = "--streaming sizes the window from the frame counts"
    elif args.dynamic:
        candidates = [_AUDIO_SECONDS_FLAG, *_WINDOW_FLAGS]
        reason = "--dynamic leaves the encoder's time axis symbolic"
    else:
        candidates = list(_WINDOW_FLAGS)
        reason = "the frame counts only apply with --streaming"

    ignored = [
        flag
        for flag, dest in candidates
        if getattr(args, dest) != parser.get_default(dest)
    ]
    if ignored:
        print(f"[WARN] Ignoring {', '.join(ignored)} — {reason}.")


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
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--dynamic",
        action="store_true",
        help="Export the encoder with dynamic audio length (decoder/joint stay static).",
    )
    shape_group.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Export a fixed streaming window sized from --left/--chunk/"
            "--right-context-frames, and record the geometry in metadata.json. "
            "Ignores --audio-seconds."
        ),
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=12,
        help=(
            "Encoder frames consumed per streaming hop (1 frame = 80 ms). Sets the "
            "emission cadence. Every hop re-encodes the whole window, so halving this "
            "roughly doubles the encoder work per second of audio. Ignored unless "
            "--streaming is set."
        ),
    )
    parser.add_argument(
        "--right-context-frames",
        type=int,
        default=12,
        help=(
            "Encoder frames of lookahead. Theoretical latency is "
            "(chunk + right) x 80 ms. Must be >= max(durations). Ignored unless "
            "--streaming is set."
        ),
    )
    parser.add_argument(
        "--left-context-frames",
        type=int,
        default=126,
        help=(
            "Encoder frames of past context. Improves quality at no latency cost. "
            "Ignored unless --streaming is set."
        ),
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=5.0,
        help=(
            "Length (seconds) of dummy audio used to shape the encoder's static "
            "trace. Ignored when --dynamic or --streaming is set."
        ),
    )
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help="Embed debug information in the exported .aimodel for debugging a conversion. "
        "Default: off, which embeds minimum debug information and makes the exported asset smaller.",
    )
    args = parser.parse_args()
    _warn_ignored_shape_args(parser, args)

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()
    create_parakeet(
        output_dir=output_dir,
        model_name=args.model,
        dtype=dtype,
        overwrite=args.overwrite,
        dynamic=args.dynamic,
        audio_seconds=args.audio_seconds,
        include_debug_info=args.include_debug_info,
        window=StreamingWindowArgs(
            left_context_frames=args.left_context_frames,
            chunk_frames=args.chunk_frames,
            right_context_frames=args.right_context_frames,
        )
        if args.streaming
        else None,
    )


if __name__ == "__main__":
    main()
