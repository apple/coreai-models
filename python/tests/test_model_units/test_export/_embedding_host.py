# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""A reference host for ``kind: "embedding"`` bundles, and tiny test snapshots.

``Bundle`` reads only ``metadata.json``, ``tokenizer/`` and the assets of a bundle directory and
follows the ``coreai_models.embedding.contract`` rules: tokenization (R6), image preprocessing
with PIL (R7), the audio front end in numpy (R8) and function calls through the Core AI runtime.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REVISION = "0123456789abcdef0123456789abcdef01234567"


def runtime_options(compute_unit: str):
    """Core AI runtime options for ``"cpu"`` or ``"gpu"``; skips when the runtime cannot run."""
    import coreai.runtime as rt

    if not rt.SpecializationOptions.is_supported():
        pytest.skip("the Core AI runtime cannot run programs here")
    if compute_unit == "cpu":
        return rt.SpecializationOptions.cpu_only()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())


async def _run(path: Path, function: str, inputs: dict[str, torch.Tensor], options) -> dict:
    import coreai.runtime as rt
    from coreai.authoring import AIModelAsset

    async with AIModelAsset.load(path).executable(options) as model:
        outputs = await model.load_function(function)(
            {key: rt.NDArray(data=value.contiguous()) for key, value in inputs.items()}
        )
        return {key: torch.from_numpy(value.numpy().copy()) for key, value in outputs.items()}


def run_function(path: Path, function: str, inputs: dict[str, torch.Tensor], options) -> dict:
    """Outputs of ``function`` of the asset at ``path`` on ``inputs``."""
    return asyncio.run(_run(path, function, inputs, options))


def asset_signatures(path: Path) -> dict[str, tuple[list, list]]:
    """Function name -> (inputs, outputs) of an asset, each ``[(name, type string)]``."""
    from coreai.authoring import AIModelAsset

    summary = AIModelAsset.load(path).summary(include_statistics=False)
    return {
        name: (summary.function_inputs(name), summary.function_outputs(name))
        for name in summary.function_names
    }


def runtime_type(spec: str, length: int | None) -> str:
    """The runtime's type string for a contract ``"dtype[shape]"`` (batch axes are ``?``)."""
    dtype, dims = spec[:-1].split("[")
    sizes = {"B": "?", "N": "?", "L": str(length)}
    shape = " × ".join(sizes.get(d, d) for d in dims.split(","))
    names = {"float32": "Float32", "int32": "Int32", "bool": "Bool"}
    return f"NDArray ({names[dtype]}, {shape})"


class Bundle:
    """A bundle directory, used through its metadata alone."""

    def __init__(self, path: Path, options) -> None:
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        self.block = self.metadata["embedding"]
        self.options = options

    def function(self, component: str) -> dict[str, Any]:
        (entry,) = self.block["components"][component]["functions"]
        return entry

    def run(self, component: str, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run ``component`` on ``n`` rows, split into batches of at most ``batch.max``."""
        spec = self.block["components"][component]
        file = self.path / self.metadata["assets"][spec["asset"]]
        entry = self.function(component)
        n = len(next(iter(inputs.values())))
        step = entry["batch"]["max"]
        parts = [
            run_function(
                file,
                entry["function"],
                {k: v[i : i + step] for k, v in inputs.items()},
                self.options,
            )
            for i in range(0, n, step)
        ]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}

    def tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """R6: the embedded tokenizer adds bos/eos; truncate keeping eos; pad right."""
        from tokenizers import Tokenizer

        text = self.block["text"]
        tokenizer = Tokenizer.from_file(str(self.path / "tokenizer" / "tokenizer.json"))
        assert tokenizer.padding is None and tokenizer.truncation is None
        max_tokens = text["truncation"]["max_tokens"]
        length = self.function("text_encoder")["length"]
        ids = torch.full((len(texts), length), text["pad_token_id"], dtype=torch.int32)
        mask = torch.zeros((len(texts), length), dtype=torch.int32)
        for row, encoding in enumerate(tokenizer.encode_batch(texts)):
            tokens = encoding.ids
            assert tokens[0] == text["bos_token_id"] and tokens[-1] == text["eos_token_id"]
            if len(tokens) > max_tokens:
                tokens = tokens[: max_tokens - 1] + [text["eos_token_id"]]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, : len(tokens)] = 1
        return ids, mask

    def embed_text(self, texts: list[str]) -> torch.Tensor:
        ids, mask = self.tokenize(texts)
        return self.run("text_encoder", {"input_ids": ids, "attention_mask": mask})["embedding"]

    def preprocess_images(self, images: list) -> torch.Tensor:
        """R7 with PIL: RGB, bicubic shortest-edge resize, center crop, u8/255, mean/std."""
        from PIL import Image

        image = self.block["image"]
        assert image["image_strategy"] == "center_crop"
        size = image["image_size"]
        rows = []
        for img in images:
            img = img.convert("RGB")
            w, h = img.size
            scale = size / min(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
            left, top = (img.width - size) // 2, (img.height - size) // 2
            img = img.crop((left, top, left + size, top + size))
            x = np.asarray(img, dtype=np.float32) / 255.0
            x = (x - np.array(image["image_mean"])) / np.array(image["image_std"])
            rows.append(torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1))
        return torch.stack(rows)

    def embed_images(self, images: list) -> torch.Tensor:
        pixels = self.preprocess_images(images)
        return self.run("image_encoder", {"pixel_values": pixels})["embedding"]

    def audio_features(self, waveforms: list[np.ndarray]) -> torch.Tensor:
        """R8 in numpy: repeatpad or head, then the dB Slaney mel spectrogram."""
        a = self.block["audio"]
        clip = (a["n_frames"] - 1) * a["hop_length"]
        filters = slaney_mel_filters(
            a["sample_rate"], a["n_fft"], a["n_mel_bins"], a["f_min"], a["f_max"]
        )
        n = np.arange(a["win_length"])
        window = 0.5 - 0.5 * np.cos(2 * np.pi * n / a["win_length"])
        rows = []
        for wave in waveforms:
            wave = np.asarray(wave, dtype=np.float64)
            if len(wave) < clip:
                wave = np.tile(wave, clip // len(wave))
                wave = np.pad(wave, (0, clip - len(wave)))
            wave = np.pad(wave[:clip], a["n_fft"] // 2, mode="reflect")
            starts = np.arange(a["n_frames"]) * a["hop_length"]
            frames = np.stack([wave[s : s + a["n_fft"]] for s in starts]) * window
            power = np.abs(np.fft.rfft(frames, n=a["n_fft"])) ** 2
            mel = power @ filters
            rows.append(10.0 * np.log10(np.maximum(mel, 1e-10)))
        return torch.from_numpy(np.stack(rows)[:, None].astype(np.float32))

    def embed_audio(self, waveforms: list[np.ndarray]) -> torch.Tensor:
        features = self.audio_features(waveforms)
        return self.run("audio_encoder", {"input_features": features})["embedding"]


def slaney_mel_filters(sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """``[n_fft // 2 + 1, n_mels]`` Slaney-scale triangles with Slaney area normalization."""
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0

    def to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        log = min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep
        return np.where(f >= min_log_hz, log, f / f_sp)

    def to_hz(m):
        log = min_log_hz * np.exp(logstep * (m - min_log_mel))
        return np.where(m >= min_log_mel, log, f_sp * m)

    hz = to_hz(np.linspace(to_mel(f_min), to_mel(f_max), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    lower = (freqs[:, None] - hz[None, :-2]) / (hz[1:-1] - hz[:-2])
    upper = (hz[None, 2:] - freqs[:, None]) / (hz[2:] - hz[1:-1])
    return np.maximum(0, np.minimum(lower, upper)) * (2.0 / (hz[2:] - hz[:-2]))


def save_tiny_tokenizer(snapshot: Path, words: list[str], bos: int, eos: int, pad: int) -> None:
    """A word-level tokenizer with ``<s> A </s>`` post-processing: bos/eos/pad take the given ids
    (pad may equal eos), ``words`` the lowest free ids in order.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    free = (i for i in range(len(words) + 3) if i not in (bos, eos, pad))
    vocab = dict(zip(words, free, strict=False))
    vocab.update({"<s>": bos, "</s>": eos})
    if pad != eos:
        vocab["<pad>"] = pad
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=words[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.RobertaProcessing(sep=("</s>", eos), cls=("<s>", bos))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>" if pad == eos else "<pad>",
        unk_token=words[0],
    ).save_pretrained(snapshot)


def snapshot_dir(root: Path) -> Path:
    """An empty HF-cache-style ``.../snapshots/<sha>`` directory."""
    path = root / "hub" / "snapshots" / REVISION
    path.mkdir(parents=True)
    return path
