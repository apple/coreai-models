# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import huggingface_hub
import torch
import transformers
import yaml
from tokenizers import AddedToken
from transformers.convert_slow_tokenizer import ParakeetConverter

# NeMo checkpoint conversion. A `.nemo` file is a tar archive of `model_config.yaml`,
# `model_weights.ckpt` (a plain torch state dict) and a SentencePiece tokenizer, so repos that ship
# only that format convert to `ParakeetForTDT` in memory without depending on NeMo. Ported from
# transformers' `models/parakeet/convert_nemo_to_hf.py`; transformers strips conversion scripts from
# release wheels.

_NEMO_TO_HF_WEIGHT_MAPPING = {
    r"encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
    r"decoder\.prediction\.embed\.": r"decoder.embedding.",
    r"decoder\.prediction\.dec_rnn\.lstm\.": r"decoder.lstm.",
    r"joint\.enc\.": r"encoder_projector.",
    r"joint\.pred\.": r"decoder.decoder_projector.",
    r"joint\.joint_net\.2\.": r"joint.head.",
}

_NEMO_ENCODER_CONFIG_MAPPING = {
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
    "n_layers": "num_hidden_layers",
    "feat_in": "num_mel_bins",
    "conv_kernel_size": "conv_kernel_size",
    "subsampling_factor": "subsampling_factor",
    "subsampling_conv_channels": "subsampling_conv_channels",
    "pos_emb_max_len": "max_position_embeddings",
    "dropout": "dropout",
    "dropout_emb": "dropout_positions",
    "dropout_att": "attention_dropout",
    "xscaling": "scale_input",
    "use_bias": "attention_bias",
}

# Training-only or implementation knobs, or ones the HF architecture fixes (`ff_expansion_factor`
# is folded into `intermediate_size` below). Anything not listed here or above fails loudly.
_NEMO_ENCODER_KEYS_TO_IGNORE = {
    "_target_",
    "att_context_size",
    "att_context_style",
    "causal_downsampling",
    "conv_context_size",
    "conv_norm_type",
    "dropout_pre_encoder",
    "feat_out",
    "ff_expansion_factor",
    "reduction",
    "reduction_factor",
    "reduction_position",
    "self_attention_model",
    "stochastic_depth_drop_prob",
    "stochastic_depth_mode",
    "stochastic_depth_start_layer",
    "subsampling",
    "untie_biases",
}

_NEMO_PREPROCESSOR_MAPPING = {
    "sample_rate": "sampling_rate",
    "window_size": "win_length",
    "window_stride": "hop_length",
    "n_fft": "n_fft",
    "features": "feature_size",
    "pad_value": "padding_value",
}

# ParakeetFeatureExtractor hardcodes these, so a checkpoint that differs cannot be served by it.
_NEMO_PREPROCESSOR_FIXED = {
    "window": "hann",
    "log": True,
    "normalize": "per_feature",
    "frame_splicing": 1,
}

# Training-time augmentation and batch padding; no effect on inference features.
_NEMO_PREPROCESSOR_KEYS_TO_IGNORE = {"_target_", "dither", "pad_to"}


def read_nemo_checkpoint(model_name: str) -> tuple[dict, dict[str, torch.Tensor], bytes]:
    """Fetch `<repo>.nemo` from the Hub; return its config, state dict and tokenizer.model bytes."""
    nemo_path = huggingface_hub.hf_hub_download(model_name, f"{Path(model_name).name}.nemo")
    print(f"[INFO] Reading NeMo checkpoint {nemo_path}...")
    # Read members by name rather than extracting: the archive is untrusted input.
    with tarfile.open(nemo_path, "r") as archive:
        members = {PurePosixPath(m.name).name: m for m in archive.getmembers() if m.isfile()}
        (tokenizer_name,) = [name for name in members if name.endswith("tokenizer.model")]
        nemo_config = yaml.safe_load(archive.extractfile(members["model_config.yaml"]))
        tokenizer_model = archive.extractfile(members[tokenizer_name]).read()
        state_dict = torch.load(
            archive.extractfile(members["model_weights.ckpt"]),
            map_location="cpu",
            weights_only=True,
        )
    return nemo_config, state_dict, tokenizer_model


def _nemo_encoder_config(nemo_config: dict) -> "transformers.ParakeetEncoderConfig":
    encoder = nemo_config["encoder"]
    kwargs = {"intermediate_size": encoder["d_model"] * encoder["ff_expansion_factor"]}
    for key, value in encoder.items():
        if key in _NEMO_ENCODER_KEYS_TO_IGNORE:
            continue
        if key not in _NEMO_ENCODER_CONFIG_MAPPING:
            raise ValueError(f"Unhandled NeMo encoder option {key}={value!r}")
        kwargs[_NEMO_ENCODER_CONFIG_MAPPING[key]] = value
        if key == "use_bias":
            kwargs["convolution_bias"] = value
    return transformers.ParakeetEncoderConfig(**kwargs)


def nemo_tdt_config(nemo_config: dict) -> "transformers.ParakeetTDTConfig":
    # NeMo appends blank after the SentencePiece vocab; the joint head covers both.
    blank_token_id = len(nemo_config["labels"])
    prednet = nemo_config["decoder"]["prednet"]
    return transformers.ParakeetTDTConfig(
        vocab_size=blank_token_id + 1,
        decoder_hidden_size=prednet["pred_hidden"],
        num_decoder_layers=prednet["pred_rnn_layers"],
        durations=list(nemo_config["model_defaults"]["tdt_durations"]),
        hidden_act=nemo_config["joint"]["jointnet"]["activation"],
        max_symbols_per_step=nemo_config["decoding"]["greedy"]["max_symbols"],
        encoder_config=_nemo_encoder_config(nemo_config).to_dict(),
        # The vocab has no `<pad>`, and only the training loss reads it, so pad reuses blank.
        pad_token_id=blank_token_id,
        blank_token_id=blank_token_id,
    )


def nemo_feature_extractor(nemo_config: dict) -> "transformers.ParakeetFeatureExtractor":
    preprocessor = nemo_config["preprocessor"]
    kwargs = {}
    for key, value in preprocessor.items():
        if key in _NEMO_PREPROCESSOR_KEYS_TO_IGNORE:
            continue
        if key in _NEMO_PREPROCESSOR_FIXED:
            if value != _NEMO_PREPROCESSOR_FIXED[key]:
                raise ValueError(
                    f"NeMo preprocessor {key}={value!r}; ParakeetFeatureExtractor only "
                    f"supports {_NEMO_PREPROCESSOR_FIXED[key]!r}"
                )
            continue
        if key not in _NEMO_PREPROCESSOR_MAPPING:
            raise ValueError(f"Unhandled NeMo preprocessor option {key}={value!r}")
        if key in ("window_size", "window_stride"):
            # NeMo gives these in seconds; HF wants samples.
            value = round(value * preprocessor["sample_rate"])
        kwargs[_NEMO_PREPROCESSOR_MAPPING[key]] = value
    return transformers.ParakeetFeatureExtractor(**kwargs)


def nemo_tokenizer(tokenizer_model: bytes, blank_token_id: int) -> "transformers.ParakeetTokenizer":
    with tempfile.TemporaryDirectory() as tmp:
        spm_path = Path(tmp) / "tokenizer.model"
        spm_path.write_bytes(tokenizer_model)
        tokenizer = transformers.ParakeetTokenizer(
            tokenizer_object=ParakeetConverter(str(spm_path)).converted(),
            clean_up_tokenization_spaces=False,
        )

    def special(token: str) -> AddedToken:
        return AddedToken(token, normalized=False, special=True)

    # `<blank>` must land on the id right after the SentencePiece vocab, where the joint head puts
    # it, so it doubles as pad rather than appending a `<pad>` first.
    tokenizer.add_tokens([special("<blank>")])
    tokenizer.add_special_tokens({"pad_token": special("<blank>"), "unk_token": special("<unk>")})
    if tokenizer.convert_tokens_to_ids("<blank>") != blank_token_id:
        raise ValueError(
            f"tokenizer puts <blank> at {tokenizer.convert_tokens_to_ids('<blank>')}, "
            f"but the model's blank id is {blank_token_id}"
        )
    return tokenizer


def nemo_to_hf_state_dict(nemo_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = {}
    for key, value in nemo_state_dict.items():
        # The featurizer's window and mel filterbank are recomputed by the feature extractor.
        if key.startswith("preprocessor.featurizer."):
            continue
        for pattern, replacement in _NEMO_TO_HF_WEIGHT_MAPPING.items():
            key = re.sub(pattern, replacement, key)
        state_dict[key] = value
    return state_dict


def load_from_nemo(
    model_name: str, dtype: torch.dtype
) -> tuple["transformers.ParakeetForTDT", "transformers.ProcessorMixin"]:
    nemo_config, nemo_state_dict, tokenizer_model = read_nemo_checkpoint(model_name)
    config = nemo_tdt_config(nemo_config)

    # Built on CPU, not the meta device: the positional encoding's `inv_freq` is a non-persistent
    # buffer the checkpoint never supplies.
    model = transformers.ParakeetForTDT(config)
    model.load_state_dict(nemo_to_hf_state_dict(nemo_state_dict))
    model.to(dtype)

    processor = transformers.ParakeetProcessor(
        feature_extractor=nemo_feature_extractor(nemo_config),
        tokenizer=nemo_tokenizer(tokenizer_model, config.blank_token_id),
    )
    return model, processor
