# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export the Gemma 4 MTP/EAGLE drafter's layer stack as a static-shape Core AI program.

The program is one draft step minus the two lookups that stay on the CPU: the token-embedding
gather before it and the masked embedder's candidate scan after it.

    inputs_embeds = cat(embed(token), target_hidden)                 # (1, 1, 2 * backbone_hidden)
    x = pre_projection(inputs_embeds)
    for layer in layers:                                             # attends to the target's K/V
        x = layer(x)
    x = norm(x)
    -> hidden = post_projection(x), centroid_logits = centroids(x), x

The drafter owns no K/V: each layer attends to the target's last sliding or full storing layer,
masked where empty. The sliding layers see the most recent ``window`` positions, as the target's do;
the full-attention layer sees the whole context, so the program has one ``draft_step_<n>`` function
per context bucket ``n`` of the target export's ladder (``export.context_ladder``), up to
``--max-context-length``. Keys and values arrive as
``(1, kv_heads, head_dim, n)``, the layout the target's cache already uses. Rotary tables are inputs
because the angle for a large position is not representable in fp16.

Beside the program it writes ``draft_scan.bin``, the candidate-scan table a runtime reads on the
CPU, so the program and the scan always come from the same checkpoint. With ``--bundle`` it
copies the four files into a target bundle exported with ``--emit-hidden-states`` and declares them
as the bundle's ``drafter`` component in ``metadata.json``.

    cd models/gemma4
    uv run export_drafter.py --model google/gemma-4-E2B-it --bundle <target bundle>

The assistant is the target's own (``<model>-assistant`` on the Hugging Face hub), downloaded on
first use; ``--drafter`` takes another repo id or a local directory instead.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from coreai_torch import TorchConverter, get_decomp_table
from export import IOS_MAX_CONTEXT_LENGTH, context_ladder
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from safetensors import safe_open
from transformers import AutoConfig

from coreai_models._constants import DRAFT_STEP_FUNCTION_NAME
from coreai_models.export.metadata import build_aimodel_metadata

INPUT_NAMES = [
    "inputs_embeds",
    "cos_sliding",
    "sin_sliding",
    "cos_full",
    "sin_full",
    "k_sliding",
    "v_sliding",
    "mask_sliding",
    "k_full",
    "v_full",
    "mask_full",
]
OUTPUT_NAMES = ["hidden", "centroid_logits", "x"]
SCAN_NAME = "draft_scan.bin"
# The bundle's `drafter` component: the program under `assets`, its sidecars under
# `auxiliary_assets`.
DRAFTER_ASSET_KEY = "drafter"
# `speculative.drafter_kind` says how the `drafter` component proposes tokens, since other drafters
# declare the same asset key with different semantics. This head runs off the target's final hidden
# state and shared K/V.
SPECULATIVE_KEY = "speculative"
DRAFTER_KIND_KEY = "drafter_kind"
DRAFTER_KIND = "mtp"
DRAFTER_AUXILIARY_KEYS = {
    "drafter_config": f"{DRAFT_STEP_FUNCTION_NAME}.json",
    "drafter_scan": SCAN_NAME,
    "drafter_scan_index": "draft_scan.json",
}
LAYER_TENSORS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "layer_scalar",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
TOP_LEVEL_TENSORS = (
    "model.embed_tokens.weight",
    "masked_embedding.token_ordering",
    "masked_embedding.centroids.weight",
    "pre_projection.weight",
    "post_projection.weight",
    "model.norm.weight",
)
# Sidecar format version. A consumer reads a missing version as 1 and rejects one newer than it
# understands, so bump it when a field is removed or changes meaning; adding a field needs no bump.
SIDECAR_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


class RMSNorm(nn.Module):
    """RMS normalization in fp16, with the input scaled down before squaring.

    ``rsqrt(mean((x / s)^2) + eps / s^2) / s`` equals ``rsqrt(mean(x^2) + eps)`` and moves fp16
    overflow of the square from ``|x| > 256`` to ``|x| > 256 * s``. Computing the square and mean
    in fp32 instead moves the program off the Neural Engine.
    """

    SCALE = 16.0

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        s = self.SCALE
        variance = (x / s).pow(2).mean(-1, keepdim=True)
        return x * (torch.rsqrt(variance + self.eps / (s * s)) / s) * self.weight


def linear(w: torch.Tensor) -> nn.Linear:
    out_features, in_features = w.shape
    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight = nn.Parameter(w.clone(), requires_grad=False)
    return layer


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class DecoderLayer(nn.Module):
    def __init__(self, t: dict, i: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        p = f"model.layers.{i}."
        self.head_dim = t[p + "self_attn.q_norm.weight"].shape[0]
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.q_proj = linear(t[p + "self_attn.q_proj.weight"])
        self.o_proj = linear(t[p + "self_attn.o_proj.weight"])
        self.q_norm = RMSNorm(t[p + "self_attn.q_norm.weight"])
        self.gate_proj = linear(t[p + "mlp.gate_proj.weight"])
        self.up_proj = linear(t[p + "mlp.up_proj.weight"])
        self.down_proj = linear(t[p + "mlp.down_proj.weight"])
        self.input_layernorm = RMSNorm(t[p + "input_layernorm.weight"])
        self.post_attention_layernorm = RMSNorm(t[p + "post_attention_layernorm.weight"])
        self.pre_feedforward_layernorm = RMSNorm(t[p + "pre_feedforward_layernorm.weight"])
        self.post_feedforward_layernorm = RMSNorm(t[p + "post_feedforward_layernorm.weight"])
        self.register_buffer("layer_scalar", t[p + "layer_scalar"].clone())

    def forward(self, x, cos, sin, k, v, mask):
        b, s, _ = x.shape
        q = self.q_proj(self.input_layernorm(x)).view(b, s, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # (b, heads, s, head_dim)
        q = q * cos + rotate_half(q) * sin
        # Grouped-query attention with each group's query heads stacked on the query axis, so K and
        # V are read once per KV head rather than copied to every query head. Gemma's attention
        # scale is 1.0 (q_norm absorbs it).
        groups = self.num_heads // self.num_kv_heads
        q = q.reshape(b, self.num_kv_heads, groups * s, self.head_dim)
        scores = torch.matmul(q, k) + mask  # (b, kv_heads, groups*s, w)
        # v stays in the cache's (head_dim, w) layout: transposing a (512, 512) full-attention block
        # makes the ANE compiler reject the program, and the whole step then falls back to the GPU.
        attn = torch.matmul(
            v, torch.softmax(scores, dim=-1).transpose(-1, -2)
        )  # (b, kv_heads, head_dim, groups*s)
        attn = attn.transpose(-1, -2).reshape(b, self.num_heads, s, self.head_dim)
        attn = attn.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        x = x + self.post_attention_layernorm(self.o_proj(attn))
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(
            nn.functional.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h)
        )
        x = x + self.post_feedforward_layernorm(h)
        return x * self.layer_scalar


class DraftStep(nn.Module):
    """One draft step of the Gemma 4 assistant, without the embedding gather and candidate scan."""

    def __init__(self, t: dict, num_kv_heads: int):
        super().__init__()
        num_layers = 1 + max(int(k.split(".")[2]) for k in t if k.startswith("model.layers."))
        head_dim = t["model.layers.0.self_attn.q_norm.weight"].shape[0]
        num_heads = t["model.layers.0.self_attn.q_proj.weight"].shape[0] // head_dim
        self.pre_projection = linear(t["pre_projection.weight"])
        self.layers = nn.ModuleList(
            DecoderLayer(t, i, num_heads, num_kv_heads) for i in range(num_layers)
        )
        self.norm = RMSNorm(t["model.norm.weight"])
        self.post_projection = linear(t["post_projection.weight"])
        self.centroids = linear(t["masked_embedding.centroids.weight"])

    def forward(
        self,
        inputs_embeds,
        cos_sliding,
        sin_sliding,
        cos_full,
        sin_full,
        k_sliding,
        v_sliding,
        mask_sliding,
        k_full,
        v_full,
        mask_full,
    ):
        x = self.pre_projection(inputs_embeds)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            # Gemma 4 assistant: the last layer is full attention, the rest sliding.
            if i == last:
                x = layer(x, cos_full, sin_full, k_full, v_full, mask_full)
            else:
                x = layer(x, cos_sliding, sin_sliding, k_sliding, v_sliding, mask_sliding)
        x = self.norm(x)
        return self.post_projection(x), self.centroids(x), x


HUB_REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def default_drafter(model_id: str) -> str:
    """The assistant trained for a Gemma 4 target: ``<model id>-assistant`` on the hub."""
    return f"{model_id}-assistant"


def resolve_drafter(drafter: str | None, model_id: str) -> Path:
    """A directory holding the assistant checkpoint.

    ``drafter`` is a local directory (used as is) or a hub repo id; omitted, it is the target's own
    assistant. Only the safetensors and ``config.json`` are fetched, not a model.
    """
    if drafter is not None and Path(drafter).expanduser().is_dir():
        return Path(drafter).expanduser()
    repo = drafter or default_drafter(model_id)
    if not HUB_REPO_ID.match(repo):
        raise SystemExit(f"--drafter {drafter!r} is neither a directory nor a hub repo id")
    try:
        return Path(snapshot_download(repo, allow_patterns=["*.safetensors", "config.json"]))
    except GatedRepoError as e:
        raise SystemExit(
            f"{repo} is gated: accept its license on the hub and log in (`hf auth login`)"
        ) from e
    except RepositoryNotFoundError as e:
        hint = "" if drafter else f"; there is no default assistant for {model_id}, pass --drafter"
        raise SystemExit(f"hub repo {repo} was not found{hint}") from e


def load_drafter(ckpt: Path) -> dict[str, torch.Tensor]:
    t = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():  # noqa: SIM118 -- safe_open is not iterable
                t[k] = sf.get_tensor(k)
    if not t:
        raise SystemExit(f"no *.safetensors in {ckpt}")
    return t


def output_table(t: dict[str, torch.Tensor]) -> torch.Tensor:
    """The matrix the masked embedder scores candidates against.

    A fine-tuned assistant can ship an untied head; drafting against the embedding instead proposes
    from the wrong matrix.
    """
    return t.get("lm_head.weight", t["model.embed_tokens.weight"])


def check_checkpoint(
    t: dict[str, torch.Tensor], text: dict | None = None, backbone_hidden_size: int | None = None
) -> None:
    """Refuse a checkpoint the runtime cannot serve.

    ``text`` is the assistant's ``text_config``, when its ``config.json`` is at hand. The runtime
    treats the last layer as full attention and every other as sliding, and runs the weights in
    float16. ``backbone_hidden_size`` is the target's hidden size: the assistant reads the target's
    final hidden state, so its input projection must be twice that wide (token embedding plus
    hidden state).
    """
    num_layers = 1 + max(
        (int(k.split(".")[2]) for k in t if k.startswith("model.layers.")), default=-1
    )
    names = list(TOP_LEVEL_TENSORS) + [
        f"model.layers.{i}.{n}" for i in range(num_layers) for n in LAYER_TENSORS
    ]
    missing = [n for n in names if n not in t]
    if missing:
        raise SystemExit(f"checkpoint is missing tensor(s): {', '.join(missing)}")
    width = t["pre_projection.weight"].shape[1]
    if backbone_hidden_size is not None and width != 2 * backbone_hidden_size:
        raise SystemExit(
            f"the assistant takes hidden states {width // 2} wide, but the target's hidden size is "
            f"{backbone_hidden_size}: it was trained for a different target"
        )
    layer_types = (text or {}).get("layer_types")
    expected = ["sliding_attention"] * (num_layers - 1) + ["full_attention"]
    if layer_types is not None and list(layer_types) != expected:
        raise SystemExit(
            f"layer_types {list(layer_types)} is not supported: the runtime implements only "
            f"{expected} (sliding layers, then one full-attention layer last)"
        )
    for name, x in t.items():
        if not x.is_floating_point():
            continue
        if not torch.isfinite(x).all():
            raise SystemExit(f"tensor '{name}' holds non-finite values")
        if not torch.isfinite(x.to(torch.float16)).all():
            raise SystemExit(
                f"tensor '{name}' overflows float16 (largest magnitude {x.abs().max():.3g})"
            )


def example_inputs(
    module: DraftStep, num_kv_heads: int, window: int, full_window: int
) -> tuple[torch.Tensor, ...]:
    """Zero inputs that fix the shapes one ``draft_step_<full_window>`` function is traced at.

    The order is ``INPUT_NAMES``. ``hd`` and ``full_hd`` are the head dims of the sliding layers and
    of the full-attention layer, ``kv`` is the target's KV head count, ``window`` the sliding window
    and ``full_window`` the context bucket the function serves.

    - ``inputs_embeds`` (1, 1, 2 * backbone_hidden): the token's embedding concatenated with the
      target's final hidden state.
    - ``cos_sliding``, ``sin_sliding`` (1, 1, 1, hd) and ``cos_full``, ``sin_full``
      (1, 1, 1, full_hd): rotary tables at the draft position, written by the runtime because the
      angle for a large position is not representable in fp16.
    - ``k_sliding``, ``v_sliding`` (1, kv, hd, window): the target's last sliding layer's keys and
      values for the most recent ``window`` positions, in the target cache's layout.
    - ``mask_sliding`` (1, 1, 1, window): additive mask, 0 where a position is visible.
    - ``k_full``, ``v_full`` (1, kv, full_hd, full_window) and ``mask_full`` (1, 1, 1, full_window):
      the same for the target's last full-attention layer, over the whole context bucket.
    """
    backbone = module.pre_projection.in_features // 2
    hd, full_hd = module.layers[0].head_dim, module.layers[-1].head_dim

    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.float16)

    return (
        zeros(1, 1, 2 * backbone),
        zeros(1, 1, 1, hd),
        zeros(1, 1, 1, hd),
        zeros(1, 1, 1, full_hd),
        zeros(1, 1, 1, full_hd),
        zeros(1, num_kv_heads, hd, window),
        zeros(1, num_kv_heads, hd, window),
        zeros(1, 1, 1, window),
        zeros(1, num_kv_heads, full_hd, full_window),
        zeros(1, num_kv_heads, full_hd, full_window),
        zeros(1, 1, 1, full_window),
    )


def full_windows(max_context: int) -> list[int]:
    """The full-attention windows to export: the target export's context buckets.

    The runtime picks the smallest bucket that covers the context, so using the target's ladder
    keeps the draft step's bucket the same as the target's.
    """
    return context_ladder(max_context)


def export(model_id: str, drafter: Path, window: int, max_context: int, out_dir: Path) -> Path:
    text = AutoConfig.from_pretrained(model_id).text_config
    if not 0 < window <= text.sliding_window:
        raise SystemExit(
            f"--window {window} must be in 1..{text.sliding_window}, the target's sliding window"
        )
    if max_context < window:
        raise SystemExit(
            f"--max-context-length {max_context} is below --window {window}: the full-attention "
            f"layer would see less context than the sliding layers"
        )
    if max_context > text.max_position_embeddings:
        raise SystemExit(
            f"--max-context-length {max_context} exceeds the target's "
            f"max_position_embeddings {text.max_position_embeddings}"
        )
    t = load_drafter(drafter)
    config_path = drafter / "config.json"
    text_config = (
        json.loads(config_path.read_text()).get("text_config") if config_path.exists() else None
    )
    check_checkpoint(t, text_config, backbone_hidden_size=text.hidden_size)
    module = DraftStep(t, text.num_key_value_heads).half().eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scan_table(t, len(module.layers), out_dir / SCAN_NAME)
    del t  # the module cloned its weights; free the 134 MB table before converting
    # One function per context bucket: a single function at `max_context` costs more per step at
    # short contexts.
    converter = TorchConverter()
    sizes = full_windows(max_context)
    for full_window in sizes:
        with torch.no_grad():
            program = torch.export.export(
                module, example_inputs(module, text.num_key_value_heads, window, full_window)
            )
        converter.add_exported_program(
            program.run_decompositions(get_decomp_table()),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            entrypoint_name=f"{DRAFT_STEP_FUNCTION_NAME}_{full_window}",
        )
    coreai_program = converter.to_coreai()

    aimodel = out_dir / f"{DRAFT_STEP_FUNCTION_NAME}.aimodel"
    coreai_program.save_asset(aimodel, build_aimodel_metadata(model_id))
    # The runtime writes the rotary tables, so it needs the target's RoPE parameters. Gemma 4's
    # full-attention layers rotate only `partial_rotary_factor` of the frequency pairs.
    rope = text.rope_parameters
    (out_dir / f"{DRAFT_STEP_FUNCTION_NAME}.json").write_text(
        json.dumps(
            {
                "version": SIDECAR_VERSION,
                "full_windows": sizes,
                "sliding_theta": rope["sliding_attention"]["rope_theta"],
                "full_theta": rope["full_attention"]["rope_theta"],
                "full_partial_rotary_factor": rope["full_attention"].get(
                    "partial_rotary_factor", 1.0
                ),
            },
            indent=2,
        )
    )
    print(f"Exported {aimodel} (window {window}, full windows {sizes})")
    return aimodel


def write_scan_table(t: dict[str, torch.Tensor], num_layers: int, path: Path) -> None:
    """The masked embedder's candidate-scan inputs: a flat blob and a JSON index of its sections.

    Only what stays on the CPU: the output table permuted so each centroid's tokens are contiguous,
    and the permutation to map a slot back to a token. The layer weights and centroids are already
    in the program.
    """
    order = t["masked_embedding.token_ordering"].to(torch.int64)
    head = output_table(t)
    vocab = head.shape[0]
    centroids = t["masked_embedding.centroids.weight"].shape[0]
    if centroids <= 0 or vocab % centroids != 0:
        raise SystemExit(
            f"vocabulary {vocab} is not divisible by the {centroids} centroids: the tokens past "
            f"{vocab // max(centroids, 1) * max(centroids, 1)} would never be scored"
        )
    tensors = [
        ("embed_perm_f16", head[order].to(torch.float16)),
        ("token_ordering_i32", order.to(torch.int32)),
    ]
    index, offset = {}, 0
    with open(path, "wb") as f:
        for name, x in tensors:
            data = x.contiguous().numpy()
            data.tofile(f)
            index[name] = {
                "offset": offset,
                "bytes": data.nbytes,
                "shape": list(x.shape),
                "dtype": str(x.dtype).removeprefix("torch."),
            }
            offset += data.nbytes
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": SIDECAR_VERSION,
                "vocab": vocab,
                "centroids": centroids,
                "per_centroid": vocab // centroids,
                "layers": num_layers,
                "tensors": index,
            },
            indent=2,
        )
    )


def add_to_bundle(out_dir: Path, bundle: Path) -> None:
    """Copy the draft head into ``bundle`` and declare it as the ``drafter`` component."""
    metadata_path = bundle / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"{bundle} has no metadata.json")
    metadata = json.loads(metadata_path.read_text())
    overrides = metadata.get("language", {}).get("overrides", {})
    if not overrides.get("emits_hidden_states"):
        raise SystemExit(
            f"{bundle} was not exported with --emit-hidden-states: its decode graphs do not output "
            "the hidden state the draft head seeds from"
        )
    speculative = metadata.get(SPECULATIVE_KEY, {})
    if DRAFTER_ASSET_KEY in metadata.get("assets", {}) and (
        speculative.get(DRAFTER_KIND_KEY) != DRAFTER_KIND
    ):
        raise SystemExit(
            f"{bundle} already declares a drafter ({DRAFTER_KIND_KEY} "
            f"{speculative.get(DRAFTER_KIND_KEY)!r}); adding this head would replace it"
        )
    aimodel_filename = f"{DRAFT_STEP_FUNCTION_NAME}.aimodel"
    if (bundle / aimodel_filename).exists():
        print(f"Warning: replacing the existing {aimodel_filename} in {bundle}")
        shutil.rmtree(bundle / aimodel_filename)
    shutil.copytree(out_dir / aimodel_filename, bundle / aimodel_filename)
    for name in DRAFTER_AUXILIARY_KEYS.values():
        shutil.copyfile(out_dir / name, bundle / name)
    metadata.setdefault("assets", {})[DRAFTER_ASSET_KEY] = aimodel_filename
    metadata.setdefault("auxiliary_assets", {}).update(DRAFTER_AUXILIARY_KEYS)
    metadata.setdefault(SPECULATIVE_KEY, {})[DRAFTER_KIND_KEY] = DRAFTER_KIND
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Added the draft head to {bundle}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Target HuggingFace model ID")
    parser.add_argument(
        "--drafter",
        default=None,
        help="The assistant: a hub repo id or a local directory holding its safetensors "
        "(default: the target's own, '<model>-assistant')",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=512,
        help="Most recent K/V positions the sliding layers attend to (default: 512)",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=IOS_MAX_CONTEXT_LENGTH,
        help="Longest context the full-attention layer attends to in full; the "
        "target export's flag of the same name, with the same default "
        f"({IOS_MAX_CONTEXT_LENGTH})",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "exports" / "drafter")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Target bundle (exported by export.py with --emit-hidden-states) to add the draft "
        "head to, as its 'drafter' component",
    )
    args = parser.parse_args()
    drafter = resolve_drafter(args.drafter, args.model)
    export(args.model, drafter, args.window, args.max_context_length, args.output_dir)
    if args.bundle is not None:
        add_to_bundle(args.output_dir, args.bundle)


if __name__ == "__main__":
    main()
