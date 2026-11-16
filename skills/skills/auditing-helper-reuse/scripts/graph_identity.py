# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Prove that a refactor left PyTorch modules identical: exported graphs, weights and outputs.

    # 1. snapshot the committed code (from a clean `git archive` copy) and the working tree
    git archive HEAD python/src | tar -x -C /tmp/before-src
    python graph_identity.py snapshot --spec spec.py --src /tmp/before-src/python/src --out /tmp/id/before
    python graph_identity.py snapshot --spec spec.py --src python/src --out /tmp/id/after
    # 2. compare: exit 0 only when everything is identical
    python graph_identity.py compare /tmp/id/before /tmp/id/after

The spec is a small Python file that defines ``cases()`` and, optionally, ``extras()``:

    def cases():
        from coreai_models.models.macos.my_model import MyEncoder, MyConfig
        cfg = MyConfig(hidden_size=64, num_hidden_layers=2)          # tiny, seeded
        return [{
            "name": "macos_text",
            "build": lambda: MyEncoder(cfg),                           # weights from the seed
            "inputs": lambda: (torch.randint(0, 100, (1, 16)), torch.ones(1, 16, dtype=torch.int32)),
        }]

    def extras():                                                      # any string-comparable facts
        import subprocess, sys
        help_text = subprocess.run([sys.executable, "-m", "coreai_models.my_model.export", "--help"],
                                   capture_output=True, text=True).stdout
        return {"cli_help": help_text}

Each case records: the ``torch.export`` op sequence (raw, after ``run_decompositions()``, and after
Core AI's decomposition table when ``coreai_torch`` is installed; a case may pass its own table as
``"decompositions": lambda: table``), every state_dict entry's name, shape, dtype and value hash,
and the outputs (compared with ``torch.equal``). Each case's module class must come from ``--src``;
an installed copy shadowing the tree is an error.

To prove conversion too, hash the converted program in ``extras()`` with ``mlir_hash`` (the spec can
``from graph_identity import mlir_hash``): it strips locations and renumbers resource names, which
are random per process.
Modules are built under ``torch.manual_seed(0)`` and run in eval mode with no grad. ``PYTHONPATH``
is set to ``--src`` so subprocesses started by ``extras()`` import the same tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path


def _load_spec(path: Path):
    spec = importlib.util.spec_from_file_location("identity_spec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ops(graph_module) -> list[str]:
    """Each call as ``target(args, kwargs)`` with graph nodes replaced by their position, so constant
    arguments (dims, eps, ``approximate=``, slice bounds) are compared too, and renames are not."""
    import torch.fx

    position = {node: i for i, node in enumerate(graph_module.graph.nodes)}

    def norm(value):
        return torch.fx.node.map_arg(value, lambda n: f"%{position[n]}")

    return [
        f"{node.target}{norm(node.args)}{norm(dict(node.kwargs)) if node.kwargs else ''}"
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ]


def normalize_mlir(text: str) -> str:
    """MLIR text without locations, with ``resource_N`` names renumbered by first appearance."""
    text = re.sub(r"^#loc\d* = .*\n", "", text, flags=re.M)
    text = re.sub(r"\s*loc\(.*?\)\s*$", "", text, flags=re.M)
    names: dict[str, str] = {}
    return re.sub(
        r"resource_\d+",
        lambda m: names.setdefault(m.group(0), f"resource_{len(names)}"),
        text,
    )


def mlir_hash(text: str) -> str:
    normalized = normalize_mlir(text)
    return (
        f"{hashlib.sha256(normalized.encode()).hexdigest()} ({len(normalized)} chars)"
    )


def _coreai_table():
    try:
        import coreai_torch
    except ImportError:
        return None
    return coreai_torch.get_decomp_table()


def _flatten(value, prefix="out"):
    import torch

    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        return {
            k2: v2
            for k, v in value.items()
            for k2, v2 in _flatten(v, f"{prefix}.{k}").items()
        }
    if isinstance(value, (list, tuple)):
        return {
            k2: v2
            for i, v in enumerate(value)
            for k2, v2 in _flatten(v, f"{prefix}.{i}").items()
        }
    return {}


def snapshot(spec_path: Path, src: Path, out: Path) -> int:
    src = src.resolve()
    sys.path.insert(0, str(src))
    sys.path.insert(
        1, str(Path(__file__).resolve().parent)
    )  # lets the spec import mlir_hash
    # Subprocesses started by extras() (e.g. a CLI --help) import from the same tree.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(src), os.environ.get("PYTHONPATH")])
    )
    import torch

    spec = _load_spec(spec_path)
    out.mkdir(parents=True, exist_ok=True)
    record: dict = {"cases": {}, "extras": {}}
    tensors: dict = {}
    for case in spec.cases():
        name = case["name"]
        torch.manual_seed(0)
        module = case["build"]().eval()
        owner = sys.modules.get(type(module).__module__)
        origin = Path(getattr(owner, "__file__", None) or spec_path).resolve()
        if (
            type(module).__module__.startswith("coreai_models")
            and src not in origin.parents
        ):
            raise SystemExit(
                f"{name}: {type(module).__name__} comes from {origin}, not --src {src}"
            )
        torch.manual_seed(1)
        inputs = case["inputs"]()
        with torch.no_grad():
            outputs = module(*inputs)
            exported = torch.export.export(
                module, inputs, **case.get("export_kwargs", {})
            )
        state = {
            k: [
                list(v.shape),
                str(v.dtype),
                hashlib.sha256(
                    v.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            ]
            for k, v in module.state_dict().items()
        }
        record["cases"][name] = {
            "ops": _ops(exported.graph_module),
            "ops_decomposed": _ops(exported.run_decompositions().graph_module),
            "state_dict": state,
        }
        table = (
            case["decompositions"]() if "decompositions" in case else _coreai_table()
        )
        if table is not None:
            record["cases"][name]["ops_coreai"] = _ops(
                exported.run_decompositions(table).graph_module
            )
        for key, tensor in _flatten(outputs).items():
            tensors[f"{name}:{key}"] = tensor.detach().cpu()
    if hasattr(spec, "extras"):
        record["extras"] = {k: str(v) for k, v in spec.extras().items()}
    (out / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    torch.save(tensors, out / "outputs.pt")
    print(
        f"snapshot: {len(record['cases'])} cases, {len(tensors)} output tensors -> {out}"
    )
    return 0


def compare(a: Path, b: Path) -> int:
    import torch

    ra, rb = (json.loads((p / "record.json").read_text()) for p in (a, b))
    ta, tb = (torch.load(p / "outputs.pt") for p in (a, b))
    diffs = 0
    for name in sorted(set(ra["cases"]) | set(rb["cases"])):
        ca, cb = ra["cases"].get(name), rb["cases"].get(name)
        if ca is None or cb is None:
            print(f"DIFF {name}: case present in only one snapshot")
            diffs += 1
            continue
        for field in ("ops", "ops_decomposed", "ops_coreai"):
            if ca.get(field) != cb.get(field):
                if ca.get(field) is None or cb.get(field) is None:
                    print(f"DIFF {name}.{field}: recorded in only one snapshot")
                    diffs += 1
                    continue
                first = next(
                    (i for i, (x, y) in enumerate(zip(ca[field], cb[field])) if x != y),
                    min(len(ca[field]), len(cb[field])),
                )
                print(
                    f"DIFF {name}.{field}: lengths {len(ca[field])} vs {len(cb[field])}, first difference at op {first}: "
                    f"{ca[field][first : first + 3]} vs {cb[field][first : first + 3]}"
                )
                diffs += 1
        for key in sorted(set(ca["state_dict"]) | set(cb["state_dict"])):
            if ca["state_dict"].get(key) != cb["state_dict"].get(key):
                print(
                    f"DIFF {name}.state_dict[{key}]: {ca['state_dict'].get(key)} vs {cb['state_dict'].get(key)}"
                )
                diffs += 1
    for key in sorted(set(ta) | set(tb)):
        if key not in ta or key not in tb or not torch.equal(ta[key], tb[key]):
            print(f"DIFF output {key}")
            diffs += 1
    for key in sorted(set(ra["extras"]) | set(rb["extras"])):
        if ra["extras"].get(key) != rb["extras"].get(key):
            print(f"DIFF extra {key}")
            diffs += 1
    print(
        f"compare: {len(ra['cases'])} cases, {len(ta)} outputs, {len(ra['extras'])} extras, {diffs} differences"
    )
    return 1 if diffs else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="record graphs, weights, outputs and extras")
    snap.add_argument("--spec", type=Path, required=True)
    snap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="python source root to import from (e.g. python/src)",
    )
    snap.add_argument("--out", type=Path, required=True)
    cmp = sub.add_parser("compare", help="exit 1 if two snapshots differ")
    cmp.add_argument("before", type=Path)
    cmp.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        return snapshot(args.spec, args.src, args.out)
    return compare(args.before, args.after)


if __name__ == "__main__":
    sys.exit(main())
