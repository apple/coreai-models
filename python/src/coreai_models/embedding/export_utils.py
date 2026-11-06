# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Building blocks for embedding exporters.

- ``EmbeddingExporter``: what a model provides to ``coreai.embedding.export``.
- ``resolve_snapshot``, ``save_tokenizer``: get the checkpoint and write ``tokenizer/``.
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
from coreai_models.embedding.contract import CheckpointFacts, EmbeddingFamily

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


def save_tokenizer(snapshot: Path, dest: Path, facts: CheckpointFacts) -> None:
    """Write the snapshot's tokenizer to ``dest`` with padding and truncation off.

    BPE merges are written as pairs, the form Swift ``CLIPTokenizer`` reads. Raises
    ``ValueError`` if the special tokens disagree with ``facts`` or the tokenizer does not add
    bos and eos.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    ids = tokenizer("a")["input_ids"]
    got = (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, ids[0], ids[-1])
    bos, eos = facts.bos_token_id, facts.eos_token_id
    want = (bos, eos, facts.pad_token_id, bos, eos)
    if got != want:
        raise ValueError(
            f"tokenizer (bos, eos, pad, first, last) {got} disagrees with the facts {want}"
        )
    tokenizer.save_pretrained(dest)
    path = dest / "tokenizer.json"
    spec = json.loads(path.read_text())
    spec["padding"] = spec["truncation"] = None
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")


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
