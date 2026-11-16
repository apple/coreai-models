# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Building blocks for embedding exporters.

- ``EmbeddingExporter``: what a model provides to ``coreai.embedding.export``.
- ``resolve_snapshot``, ``save_tokenizer``: get the checkpoint and write ``tokenizer/``.
- ``read_prompts``, ``fetch_weights``: read the sentence-transformers prompts and download the
  ``*.safetensors`` weights of a snapshot.
- ``FunctionSpec``, ``bucket_plan``: static ``len{L}_b{B}`` text functions.
- ``export_program``, ``save_program``: convert tower modules and save them as an ``.aimodel``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from coreai_models._constants import DEFAULT_INCLUDE_DEBUG_INFO
from coreai_models.embedding.contract import CheckpointFacts, EmbeddingFamily, function_entry

if TYPE_CHECKING:
    from coreai.authoring import AIProgram


TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


SNAPSHOT_PATTERNS = [
    "*.json",
    "pytorch_model.bin",
]


def resolve_snapshot(hf_model_id: str, model_dir: str | None = None) -> tuple[Path, str]:
    """Return ``(snapshot_dir, revision)``: ``model_dir``, or the downloaded snapshot.

    ``revision`` is the name of a Hugging Face cache snapshot directory (``.../snapshots/<sha>``),
    or ``""`` for a directory outside the cache. Raises ``FileNotFoundError`` if ``model_dir`` is
    not a directory.
    """
    if model_dir is not None:
        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"model dir {path} is not a directory")
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(hf_model_id, allow_patterns=SNAPSHOT_PATTERNS)).resolve()
    sha = path.name if path.parent.name == "snapshots" else ""
    return path, sha


def read_prompts(snapshot: Path) -> tuple[dict[str, str], str | None]:
    """Read ``prompts`` / ``default_prompt_name`` from ``config_sentence_transformers.json``."""
    cfg_path = snapshot / "config_sentence_transformers.json"
    if not cfg_path.exists():
        return {}, None
    cfg = json.loads(cfg_path.read_text())
    return dict(cfg.get("prompts") or {}), cfg.get("default_prompt_name")


def fetch_weights(family: EmbeddingFamily, snapshot: Path) -> None:
    """Download the ``*.safetensors`` files of the snapshot's revision into its HF cache if it
    has none.
    """
    if any(snapshot.glob("*.safetensors")):
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        family.hf_model_id,
        revision=snapshot.name,
        cache_dir=snapshot.parents[2],
        allow_patterns=["*.safetensors"],
    )


def save_tokenizer(snapshot: Path, dest: Path, facts: CheckpointFacts) -> None:
    """Write the snapshot's tokenizer to ``dest`` with padding and truncation off.

    BPE merges are written as pairs, the form Swift ``CLIPTokenizer`` reads. Raises
    ``ValueError`` unless the tokenizer's ``bos_token_id`` and ``pad_token_id`` equal the facts'
    and it encodes a text as bos (none when ``facts.bos_token_id`` is None), the text, then
    ``facts.eos_token_id``. Its ``eos_token`` attribute is not compared: Qwen3-Embedding's is
    ``<|im_end|>`` while it appends ``<|endoftext|>`` (R6).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    bos, eos = facts.bos_token_id, facts.eos_token_id
    text = tokenizer("a", add_special_tokens=False)["input_ids"]
    got = (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer("a")["input_ids"])
    want = (bos, facts.pad_token_id, ([] if bos is None else [bos]) + text + [eos])
    if got != want:
        raise ValueError(f"tokenizer (bos, pad, ids of 'a') {got} disagrees with the facts {want}")
    tokenizer.save_pretrained(dest)
    path = dest / "tokenizer.json"
    spec = json.loads(path.read_text())
    spec["padding"] = spec["truncation"] = None
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")


@dataclass(frozen=True)
class FunctionSpec:
    """One static function; text inputs are ``int32[batch, length]``."""

    name: str
    batch: int
    length: int

    def as_dict(self) -> dict[str, Any]:
        return function_entry(self.name, self.batch, self.length)


def function_name(length: int, batch: int, prefix: str = "") -> str:
    return f"{prefix}len{length}_b{batch}"


def bucket_plan(
    length_buckets: list[int] | tuple[int, ...],
    batch_sizes: list[int] | tuple[int, ...],
    max_batch_tokens: int,
) -> list[FunctionSpec]:
    """Cross length buckets with batch sizes, dropping pairs above ``max_batch_tokens``.

    Batch 1 is always kept for every length. Functions are named ``len{L}_b{B}``.

    Raises:
        ValueError: if ``length_buckets`` is empty or holds a length below 2, or a batch size is
            below 1.
    """
    lengths = sorted(set(int(x) for x in length_buckets))
    batches = sorted(set(int(x) for x in batch_sizes) | {1})
    if not lengths or lengths[0] < 2:
        raise ValueError(f"length_buckets must be >= 2 (BOS + EOS), got {length_buckets}")
    if batches[0] < 1:
        raise ValueError(f"batch_sizes must be >= 1, got {batch_sizes}")
    return [
        FunctionSpec(function_name(length, batch), batch, length)
        for length in lengths
        for batch in batches
        if batch == 1 or batch * length <= max_batch_tokens
    ]


_SPEC = re.compile(r"^(\w+)\[([\w,]+)\]$")


def parse_spec(spec: str) -> tuple[str, list[str]]:
    """Split ``"dtype[d0,d1,...]"`` into the dtype and its dimensions."""
    match = _SPEC.match(spec)
    if match is None:
        raise ValueError(f"bad tensor spec {spec!r}")
    return match.group(1), match.group(2).split(",")


def trace_inputs(
    inputs: dict[str, str], entry: dict[str, Any]
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[int, Any], ...]]:
    """Sample arguments and dynamic shapes for tracing one function of a component.

    ``B`` and ``N`` follow ``entry["batch"]``; a ``{min, max}`` batch makes them dynamic. ``L``
    is ``entry["length"]``. Values: ones for ``attention_mask``, random normal for floats, zeros
    otherwise.
    """
    batch = entry["batch"]
    sizes: dict[str, int] = {"L": entry.get("length", 0)}
    dims: dict[str, Any] = {}
    for axis, sample in (("B", 2), ("N", 3)):
        if isinstance(batch, int):
            sizes[axis] = batch
        else:
            sizes[axis] = min(max(sample, batch["min"]), batch["max"])
            dims[axis] = torch.export.Dim(f"batch_{axis}", min=batch["min"], max=batch["max"])
    generator = torch.Generator().manual_seed(0)
    args, shapes = [], []
    for name, spec in inputs.items():
        dtype, axes = parse_spec(spec)
        shape = [sizes[a] if a in sizes else int(a) for a in axes]
        if dtype.startswith("float"):
            value = torch.randn(shape, generator=generator, dtype=getattr(torch, dtype))
        elif name == "attention_mask":
            value = torch.ones(shape, dtype=getattr(torch, dtype))
        else:
            value = torch.zeros(shape, dtype=getattr(torch, dtype))
        args.append(value)
        shapes.append({i: dims[a] for i, a in enumerate(axes) if a in dims})
    return tuple(args), tuple(shapes)


def export_program(
    functions: list[tuple[nn.Module, dict[str, Any]]],
    include_debug_info: bool = DEFAULT_INCLUDE_DEBUG_INFO,
) -> AIProgram:
    """Convert ``(module, component spec)`` pairs into one program whose functions share weights.

    Each spec's ``inputs``, ``outputs`` and single ``functions`` entry give its function's
    signature.
    """
    from coreai_models.converter import add_function, make_converter

    converter = make_converter(include_debug_info)
    for module, spec in functions:
        (entry,) = spec["functions"]
        args, dynamic_shapes = trace_inputs(spec["inputs"], entry)
        add_function(
            converter,
            entry["function"],
            module,
            args,
            list(spec["inputs"]),
            list(spec["outputs"]),
            dynamic_shapes=dynamic_shapes,
        )
    return converter.to_coreai()


def save_program(program: AIProgram, out: Path, family: EmbeddingFamily, component: str) -> None:
    """Save ``program`` as ``out/<component>.aimodel`` with the model's card metadata."""
    from coreai_models.export import metadata as export_metadata

    program.save_asset(
        out / f"{component}.aimodel",
        export_metadata.build_aimodel_metadata(family.hf_model_id, component=component),
    )


@dataclass(frozen=True)
class EmbeddingExporter:
    """A model ``coreai.embedding.export`` can export.

    - ``read_facts(snapshot)``: returns the checkpoint's ``CheckpointFacts``. Raises
      ``ValueError`` if the checkpoint uses a setting the contract cannot express. Runs before
      conversion.
    - ``export_programs(snapshot, out, facts, *, platform, text_only, fused, compute_precision,
      include_debug_info, revision)``: writes the assets into ``out`` and returns
      ``(assets, embedding block)``.
    - ``fused``: whether the model supports ``--fused``.
    """

    family: EmbeddingFamily
    read_facts: Callable[[Path], CheckpointFacts]
    export_programs: Callable[..., tuple[dict[str, str], dict[str, Any]]]
    fused: bool = False
