# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Parakeet NeMo checkpoint conversion (``models/parakeet/nemo_checkpoint.py``).

The module is not part of the installed package, so it is loaded by path. Everything runs on a
v2-shaped config shrunk to toy sizes and a synthetic ``.nemo`` archive, so nothing is downloaded.
The tokenizer conversion is not covered: it needs ``protobuf``, which the test env does not ship.
"""

import copy
import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
yaml = pytest.importorskip("yaml")


def _load():
    name = "nemo_checkpoint"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[4] / "models/parakeet/nemo_checkpoint.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nemo_checkpoint = _load()

# The keys of nvidia/parakeet-tdt-0.6b-v2's `model_config.yaml` that the converter reads, with its
# sizes shrunk (1 encoder layer, 1 LSTM layer) and everything else as shipped.
_LABELS = ["<unk>", "▁t", "▁th", "▁a", "in", "e"]
_V2_CONFIG = {
    "labels": _LABELS,
    "model_defaults": {"tdt_durations": [0, 1, 2, 3, 4]},
    "decoding": {"durations": [0, 1, 2, 3, 4], "greedy": {"max_symbols": 10}},
    "decoder": {"prednet": {"pred_hidden": 8, "pred_rnn_layers": 1, "t_max": None, "dropout": 0.2}},
    "joint": {"jointnet": {"joint_hidden": 8, "activation": "relu", "dropout": 0.2}},
    "encoder": {
        "_target_": "nemo.collections.asr.modules.ConformerEncoder",
        "feat_in": 8,
        "feat_out": -1,
        "n_layers": 1,
        "d_model": 16,
        "use_bias": False,
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "subsampling_conv_channels": 4,
        "causal_downsampling": False,
        "reduction": None,
        "reduction_position": None,
        "reduction_factor": 1,
        "ff_expansion_factor": 4,
        "self_attention_model": "rel_pos",
        "n_heads": 2,
        "att_context_size": [-1, -1],
        "att_context_style": "regular",
        "xscaling": False,
        "untie_biases": True,
        "pos_emb_max_len": 5000,
        "conv_kernel_size": 9,
        "conv_norm_type": "batch_norm",
        "conv_context_size": None,
        "dropout": 0.1,
        "dropout_pre_encoder": 0.1,
        "dropout_emb": 0.0,
        "dropout_att": 0.1,
        "stochastic_depth_drop_prob": 0.0,
        "stochastic_depth_mode": "linear",
        "stochastic_depth_start_layer": 1,
    },
    "preprocessor": {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        "sample_rate": 16000,
        "normalize": "per_feature",
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "features": 128,
        "n_fft": 512,
        "log": True,
        "frame_splicing": 1,
        "dither": 1e-05,
        "pad_to": 0,
        "pad_value": 0.0,
    },
}

# Every weight name v2's `model_weights.ckpt` has for encoder layer 0 and LSTM layer 0, plus the
# ones outside the layer stacks.
_V2_LAYER_FREE_KEYS = [
    "preprocessor.featurizer.window",
    "preprocessor.featurizer.fb",
    "encoder.pre_encode.out.weight",
    "encoder.pre_encode.out.bias",
    *[f"encoder.pre_encode.conv.{i}.{p}" for i in (0, 2, 3, 5, 6) for p in ("weight", "bias")],
    "decoder.prediction.embed.weight",
    *[
        f"decoder.prediction.dec_rnn.lstm.{p}_l0"
        for p in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    ],
    *[f"joint.{m}.{p}" for m in ("pred", "enc", "joint_net.2") for p in ("weight", "bias")],
]
_V2_LAYER_0_KEYS = [
    f"encoder.layers.0.{name}"
    for name in [
        *[
            f"norm_{m}.{p}"
            for m in ("feed_forward1", "conv", "self_att", "feed_forward2", "out")
            for p in ("weight", "bias")
        ],
        *[f"feed_forward{i}.linear{j}.weight" for i in (1, 2) for j in (1, 2)],
        "conv.pointwise_conv1.weight",
        "conv.depthwise_conv.weight",
        "conv.pointwise_conv2.weight",
        *[
            f"conv.batch_norm.{p}"
            for p in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
        ],
        "self_attn.pos_bias_u",
        "self_attn.pos_bias_v",
        *[f"self_attn.linear_{p}.weight" for p in ("q", "k", "v", "out", "pos")],
    ]
]


def _config(**overrides) -> dict:
    config = copy.deepcopy(_V2_CONFIG)
    for section, values in overrides.items():
        config[section].update(values)
    return config


def test_nemo_weight_names_cover_every_hf_weight():
    nemo_keys = _V2_LAYER_FREE_KEYS + _V2_LAYER_0_KEYS
    assert len(nemo_keys) == 54
    model = nemo_checkpoint.transformers.ParakeetForTDT(nemo_checkpoint.nemo_tdt_config(_config()))
    remapped = nemo_checkpoint.nemo_to_hf_state_dict(dict.fromkeys(nemo_keys))
    assert sorted(remapped) == sorted(model.state_dict())


def test_tdt_config_from_v2_config():
    config = nemo_checkpoint.nemo_tdt_config(_config())
    # NeMo's blank is the id after the SentencePiece vocab, and it doubles as pad.
    assert config.blank_token_id == len(_LABELS)
    assert config.pad_token_id == config.blank_token_id
    assert config.vocab_size == len(_LABELS) + 1
    assert list(config.durations) == [0, 1, 2, 3, 4]
    assert (config.decoder_hidden_size, config.num_decoder_layers) == (8, 1)
    assert (config.hidden_act, config.max_symbols_per_step) == ("relu", 10)

    encoder = config.encoder_config
    assert (encoder.hidden_size, encoder.num_attention_heads) == (16, 2)
    assert encoder.num_hidden_layers == 1
    assert encoder.intermediate_size == 16 * 4
    assert encoder.attention_bias is False and encoder.convolution_bias is False
    assert (encoder.num_mel_bins, encoder.subsampling_conv_channels) == (8, 4)
    assert encoder.scale_input is False


def test_feature_extractor_from_v2_config(monkeypatch):
    # ParakeetFeatureExtractor needs librosa, which the test env lacks; check the kwargs it gets.
    monkeypatch.setattr(nemo_checkpoint.transformers, "ParakeetFeatureExtractor", dict)
    features = nemo_checkpoint.nemo_feature_extractor(_config())
    # NeMo gives the window and hop in seconds; HF wants samples at 16 kHz.
    assert features == {
        "sampling_rate": 16000,
        "win_length": 400,
        "hop_length": 160,
        "feature_size": 128,
        "n_fft": 512,
        "padding_value": 0.0,
    }


def test_unhandled_encoder_option_fails():
    with pytest.raises(ValueError, match="Unhandled NeMo encoder option new_knob"):
        nemo_checkpoint.nemo_tdt_config(_config(encoder={"new_knob": 1}))


def test_unsupported_preprocessor_value_fails():
    with pytest.raises(ValueError, match="window='hamming'"):
        nemo_checkpoint.nemo_feature_extractor(_config(preprocessor={"window": "hamming"}))


def _write_nemo(path: Path, members: dict[str, bytes]) -> None:
    """Write a `.nemo` laid out like v2's: a `.` dir, `./`-prefixed members, hashed tokenizers."""
    with tarfile.open(path, "w") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, data in members.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _nemo_members(**extra: bytes) -> dict[str, bytes]:
    weights = io.BytesIO()
    torch.save({"joint.enc.weight": torch.arange(6.0).reshape(2, 3)}, weights)
    return {
        "4cf7_vocab.txt": b"vocab",
        "705f_tokenizer.model": b"spm-model",
        "a471_tokenizer.vocab": b"spm-vocab",
        "model_config.yaml": yaml.safe_dump(_V2_CONFIG).encode(),
        "model_weights.ckpt": weights.getvalue(),
        **extra,
    }


def _patch_download(monkeypatch, path: Path) -> list:
    calls = []

    def fake_download(repo_id, filename):
        calls.append((repo_id, filename))
        return str(path)

    monkeypatch.setattr(nemo_checkpoint.huggingface_hub, "hf_hub_download", fake_download)
    return calls


def test_read_nemo_checkpoint(tmp_path, monkeypatch):
    nemo_path = tmp_path / "parakeet-tdt-0.6b-v2.nemo"
    _write_nemo(nemo_path, _nemo_members())
    calls = _patch_download(monkeypatch, nemo_path)

    config, state_dict, tokenizer_model = nemo_checkpoint.read_nemo_checkpoint(
        "nvidia/parakeet-tdt-0.6b-v2"
    )

    assert calls == [("nvidia/parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo")]
    assert config == _V2_CONFIG
    assert torch.equal(state_dict["joint.enc.weight"], torch.arange(6.0).reshape(2, 3))
    assert tokenizer_model == b"spm-model"


def test_read_nemo_checkpoint_rejects_two_tokenizers(tmp_path, monkeypatch):
    nemo_path = tmp_path / "parakeet-tdt-0.6b-v2.nemo"
    _write_nemo(nemo_path, _nemo_members(**{"9999_tokenizer.model": b"other"}))
    _patch_download(monkeypatch, nemo_path)

    with pytest.raises(ValueError):
        nemo_checkpoint.read_nemo_checkpoint("nvidia/parakeet-tdt-0.6b-v2")
