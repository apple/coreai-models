# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""coreai_models.model_registry — catalogue of supported models with export presets.

CLI:
    uv run coreai.model.registry --list-families --type llm
    uv run coreai.model.registry --list-models --type llm --platform iOS
    uv run coreai.model.registry --model-info qwen3-0.6b --platform iOS --json
    uv run coreai.model.registry --model-info qwen3-0.6b --platform iOS --as-export-args
    uv run coreai.model.registry --list-models --type utility
    uv run coreai.model.registry --model-info clip-vit-b32 --type utility --as-export-args

`--type` selects the preset table (`llm`, `diffusion`, or `utility`).
`--platform` (`macOS` / `iOS`) filters LLM variants and utility model platforms.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from coreai_models.export._constants import IOS_DEFAULT_MAX_CONTEXT_LENGTH

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPreset:
    """Export preset for one (model, variant) combination."""

    short_name: str
    hf_id: str
    family: str
    type: str  # "llm" | "diffusion"
    variant: str | None = None  # "macOS" | "iOS" for llm; None for diffusion
    compression: str = "none"
    compute_precision: str | None = None  # None = use export tool default
    max_context_length: int | None = None  # llm-only; None for diffusion
    experimental: bool = False
    notes: str | None = None
    # Optional YAML config path, resolved relative to the repo root in the
    # source tree (e.g. `models/qwen3/qwen3_0_6b_mixed_4bit_8bit.yaml`).
    compression_config: str | None = None


@dataclass(frozen=True)
class UtilityModel:
    """Lightweight registry entry for standalone (non-LLM/diffusion) models."""

    short_name: str
    hf_id: str
    model_type: str
    task: str
    export_script: str
    platforms: tuple[str, ...] = ("iOS", "macOS")
    notes: str | None = None


# ---------------------------------------------------------------------------
# LLM presets
# ---------------------------------------------------------------------------

LLM_PRESETS: list[ModelPreset] = [
    # --- macOS (compute_precision varies, compression = 4bit unless noted) ---
    ModelPreset(
        "qwen2.5-1.5b-instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5",
        "llm",
        "macOS",
        "4bit",
        "float16",
        32768,
    ),
    ModelPreset("qwen3-0.6b", "Qwen/Qwen3-0.6B", "qwen3", "llm", "macOS", "4bit", "float16", 8192),
    ModelPreset("qwen3-4b", "Qwen/Qwen3-4B", "qwen3", "llm", "macOS", "4bit", "float16", 40960),
    ModelPreset("qwen3-8b", "Qwen/Qwen3-8B", "qwen3", "llm", "macOS", "4bit", "float16", 40960),
    ModelPreset(
        "qwen3-coder-30b-a3b-instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "qwen3",
        "llm",
        "macOS",
        "4bit",
        "float16",
        262144,
    ),
    ModelPreset(
        "gemma3-4b-it", "google/gemma-3-4b-it", "gemma3", "llm", "macOS", "4bit", "bfloat16", 131072
    ),
    ModelPreset(
        "gemma3-12b-it",
        "google/gemma-3-12b-it",
        "gemma3",
        "llm",
        "macOS",
        "4bit",
        "bfloat16",
        131072,
    ),
    ModelPreset(
        "mistral-7b-instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral",
        "llm",
        "macOS",
        "4bit",
        "float16",
        8192,
    ),
    ModelPreset(
        "mixtral-8x7b-instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mixtral",
        "llm",
        "macOS",
        "4bit",
        "float16",
        32768,
    ),
    ModelPreset(
        "gpt-oss-20b", "openai/gpt-oss-20b", "gpt-oss", "llm", "macOS", "none", "bfloat16", 32768
    ),
    # --- iOS (compression = palettized) ---
    ModelPreset(
        "qwen3-0.6b",
        "Qwen/Qwen3-0.6B",
        "qwen3",
        "llm",
        "iOS",
        "none",
        "float16",
        IOS_DEFAULT_MAX_CONTEXT_LENGTH,
        compression_config="models/qwen3/qwen3_0_6b_mixed_4bit_8bit.yaml",
    ),
    ModelPreset(
        "qwen2.5-1.5b-instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5",
        "llm",
        "iOS",
        "4bit_weight_palettized_group8",
        "float16",
        IOS_DEFAULT_MAX_CONTEXT_LENGTH,
    ),
    ModelPreset(
        "qwen3-4b",
        "Qwen/Qwen3-4B",
        "qwen3",
        "llm",
        "iOS",
        "none",
        "float16",
        IOS_DEFAULT_MAX_CONTEXT_LENGTH,
        compression_config="models/qwen3/qwen3_4b_mixed_4bit_8bit.yaml",
    ),
]

# ---------------------------------------------------------------------------
# Diffusion presets
# ---------------------------------------------------------------------------

DIFFUSION_PRESETS: list[ModelPreset] = [
    ModelPreset(
        "sd-1.5",
        "runwayml/stable-diffusion-v1-5",
        "stable-diffusion",
        "diffusion",
        None,
        "none",
        "float16",
        None,
    ),
    ModelPreset(
        "sd-2.1",
        "sd2-community/stable-diffusion-2-1",
        "stable-diffusion",
        "diffusion",
        None,
        "none",
        "float16",
        None,
    ),
    ModelPreset(
        "sd-3.5-medium",
        "stabilityai/stable-diffusion-3.5-medium",
        "stable-diffusion-3",
        "diffusion",
        None,
        "none",
        "float16",
        None,
    ),
    ModelPreset(
        "flux2-klein-4b",
        "black-forest-labs/FLUX.2-klein-4B",
        "flux2",
        "diffusion",
        None,
        "4bit",
        "float16",
        None,
        notes="4bit recommended; use --compression none for full precision",
    ),
]

# ---------------------------------------------------------------------------
# Utility model presets (standalone export scripts)
# ---------------------------------------------------------------------------

UTILITY_PRESETS: list[UtilityModel] = [
    # --- Embedding ---
    UtilityModel(
        "clip-vit-b32",
        "openai/clip-vit-base-patch32",
        "clip",
        "embedding",
        "models/clip/export.py",
    ),
    UtilityModel(
        "clap-htsat",
        "laion/clap-htsat-unfused",
        "clap",
        "embedding",
        "models/clap/export.py",
    ),
    # --- ASR ---
    UtilityModel(
        "whisper-large-v3-turbo",
        "openai/whisper-large-v3-turbo",
        "whisper",
        "asr",
        "models/whisper/export.py",
    ),
    UtilityModel(
        "whisper-large-v3",
        "openai/whisper-large-v3",
        "whisper",
        "asr",
        "models/whisper/export.py",
    ),
    UtilityModel(
        "wav2vec2-base",
        "wav2vec2_asr_base_960h",
        "wav2vec2",
        "asr",
        "models/wav2vec2/export.py",
    ),
    # --- Detection ---
    UtilityModel(
        "yolos-base",
        "hustvl/yolos-base",
        "yolo",
        "detection",
        "models/yolo/export.py",
    ),
    UtilityModel(
        "yolos-tiny",
        "hustvl/yolos-tiny",
        "yolo",
        "detection",
        "models/yolo/export.py",
    ),
    # --- Segmentation ---
    UtilityModel(
        "efficient-sam-vitt",
        "efficient_sam_vitt",
        "efficient-sam",
        "segmentation",
        "models/efficient-sam/export.py",
    ),
    UtilityModel(
        "sam3",
        "facebook/sam3",
        "sam3",
        "segmentation",
        "models/sam3/export.py",
    ),
    # --- Depth ---
    UtilityModel(
        "depth-anything-3-small",
        "depth-anything/da3-small",
        "depth-anything",
        "depth",
        "models/depth-anything/export.py",
        platforms=("macOS",),
    ),
    # --- Super-resolution ---
    UtilityModel(
        "edsr-x2",
        "edsr_r16f64_x2",
        "edsr",
        "super-resolution",
        "models/edsr/export.py",
    ),
    # --- Encoding ---
    UtilityModel(
        "roberta-base",
        "roberta-base",
        "roberta",
        "encoding",
        "models/roberta/export.py",
    ),
    UtilityModel(
        "t5-small",
        "google-t5/t5-small",
        "t5",
        "encoding",
        "models/t5/export.py",
    ),
    UtilityModel(
        "t5-base",
        "google-t5/t5-base",
        "t5",
        "encoding",
        "models/t5/export.py",
    ),
    UtilityModel(
        "t5-large",
        "google-t5/t5-large",
        "t5",
        "encoding",
        "models/t5/export.py",
    ),
    # --- Classification ---
    UtilityModel(
        "pvt-v2-b0",
        "pvt_v2_b0",
        "pvt",
        "classification",
        "models/pvt/export.py",
    ),
]

# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

KNOWN_TYPES = ("llm", "diffusion", "utility")
KNOWN_VARIANTS = ("macOS", "iOS")


def all_presets() -> list[ModelPreset]:
    return LLM_PRESETS + DIFFUSION_PRESETS


def all_utility_models() -> list[UtilityModel]:
    return UTILITY_PRESETS


def presets_for_type(model_type: str) -> list[ModelPreset]:
    if model_type == "llm":
        return LLM_PRESETS
    if model_type == "diffusion":
        return DIFFUSION_PRESETS
    if model_type == "utility":
        raise ValueError(
            "Use all_utility_models() for utility type — UtilityModel is a different schema."
        )
    raise ValueError(f"Unknown --type {model_type!r}. Known: {', '.join(KNOWN_TYPES)}")


def filter_presets(
    presets: Iterable[ModelPreset],
    *,
    family: str | None = None,
    variant: str | None = None,
    include_experimental: bool = False,
) -> list[ModelPreset]:
    out: list[ModelPreset] = []
    for p in presets:
        if not include_experimental and p.experimental:
            continue
        if family is not None and p.family != family:
            continue
        if variant is not None and p.variant != variant:
            continue
        out.append(p)
    return out


def lookup_preset(
    short_name: str,
    *,
    model_type: str,
    variant: str | None = None,
) -> ModelPreset:
    """Look up a single preset. If `variant` is None and the model has multiple
    variants, raises — caller must disambiguate."""
    candidates = [
        p
        for p in presets_for_type(model_type)
        if p.short_name == short_name and (variant is None or p.variant == variant)
    ]
    if not candidates:
        msg = f"No preset for short_name={short_name!r} type={model_type!r}"
        if variant is not None:
            msg += f" variant={variant!r}"
        raise KeyError(msg)
    if len(candidates) > 1:
        variants = sorted({p.variant for p in candidates if p.variant is not None})
        raise KeyError(
            f"Multiple variants for {short_name!r} (type={model_type!r}): "
            f"{variants}. Pass --platform to disambiguate."
        )
    return candidates[0]


def try_lookup_preset(
    short_name: str,
    *,
    model_type: str | None = None,
    variant: str | None = None,
) -> ModelPreset | None:
    """Try to resolve a short-name. Returns None if not found.

    If model_type is None, searches all preset tables.
    """
    presets = presets_for_type(model_type) if model_type else all_presets()
    candidates = [
        p
        for p in presets
        if p.short_name == short_name and (variant is None or p.variant == variant)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple matches — prefer the variant that matches or macOS default
    if variant is None:
        macos = [p for p in candidates if p.variant == "macOS"]
        if len(macos) == 1:
            return macos[0]
    return None


def try_lookup_preset_by_hf_id(
    hf_id: str,
    *,
    model_type: str | None = None,
    variant: str | None = None,
) -> ModelPreset | None:
    """Try to resolve a HuggingFace ID to a preset. Returns None if not found.

    Case-insensitive match on hf_id. When multiple variants exist and no
    variant is specified, prefers 'macOS'.
    """
    presets = presets_for_type(model_type) if model_type else all_presets()
    hf_id_lower = hf_id.lower()
    candidates = [
        p
        for p in presets
        if p.hf_id.lower() == hf_id_lower and (variant is None or p.variant == variant)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if variant is None:
        macos = [p for p in candidates if p.variant == "macOS"]
        if len(macos) == 1:
            return macos[0]
    return None


def families(model_type: str, *, include_experimental: bool = False) -> list[str]:
    if model_type == "utility":
        seen: list[str] = []
        for u in UTILITY_PRESETS:
            if u.model_type not in seen:
                seen.append(u.model_type)
        return seen
    seen = []
    for p in presets_for_type(model_type):
        if not include_experimental and p.experimental:
            continue
        if p.family not in seen:
            seen.append(p.family)
    return seen


def utility_tasks() -> list[str]:
    """Return the distinct task categories across all utility models."""
    seen: list[str] = []
    for u in UTILITY_PRESETS:
        if u.task not in seen:
            seen.append(u.task)
    return seen


def lookup_utility_model(short_name: str) -> UtilityModel | None:
    """Look up a utility model by short_name. Returns None if not found."""
    for u in UTILITY_PRESETS:
        if u.short_name == short_name:
            return u
    return None


def filter_utility_models(
    *,
    model_type: str | None = None,
    task: str | None = None,
    platform: str | None = None,
) -> list[UtilityModel]:
    out: list[UtilityModel] = []
    for u in UTILITY_PRESETS:
        if model_type is not None and u.model_type != model_type:
            continue
        if task is not None and u.task != task:
            continue
        if platform is not None and platform not in u.platforms:
            continue
        out.append(u)
    return out


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _preset_to_export_args(preset: ModelPreset) -> list[str]:
    """Build the argv that `coreai.{type}.export` should receive for this preset."""
    args: list[str] = [preset.hf_id]
    if preset.compression_config:
        args += [
            "--compression-config",
            preset.compression_config,
        ]
    elif preset.compression and preset.compression != "none":
        args += ["--compression", preset.compression]
    elif preset.compression == "none":
        args += ["--compression", "none"]
    if preset.compute_precision is not None:
        args += ["--compute-precision", preset.compute_precision]
    if preset.max_context_length is not None:
        args += ["--max-context-length", str(preset.max_context_length)]
    # variant value matches coreai.llm.export's --platform choices, so pass through directly.
    # Skip the flag for the macOS default to match the batch scripts.
    if preset.type == "llm" and preset.variant and preset.variant != "macOS":
        args += ["--platform", preset.variant]
    return args


def _preset_to_output_name(preset: ModelPreset) -> str:
    """Return the artifact basename the export tool would write for this preset.

    LLM: `<sanitized_hf_tail>[_<compression>]_<dynamic|static>` — matches the
    filename produced by `coreai_models.export.pipeline._generate_output_name`.
    macOS variant maps to `_dynamic`, iOS variant maps to `_static`. When the
    preset points at a YAML config, the YAML stem (which encodes model identity)
    replaces the `<hf_tail>_<compression>` segment to avoid duplication.

    Diffusion: the HF id's tail (e.g. `stable-diffusion-v1-5`); the export
    tool writes a directory of components there.
    """
    tail = preset.hf_id.split("/")[-1]
    if preset.type == "diffusion":
        return tail
    if preset.variant == "macOS":
        variant_suffix = "_dynamic"
    elif preset.variant == "iOS":
        variant_suffix = "_static"
    else:
        raise ValueError(
            f"Unsupported variant {preset.variant!r} for LLM preset {preset.short_name!r}; "
            "expected 'macOS' or 'iOS'"
        )
    base = re.sub(r"[^a-z0-9]+", "_", tail.lower()).strip("_")
    if preset.compression_config:
        stem = Path(preset.compression_config).stem
        # Mirror pipeline._generate_output_name: skip the hf-tail prefix only
        # when the YAML stem already starts with it.
        suffix = stem if stem == base or stem.startswith(f"{base}_") else f"{base}_{stem}"
        return f"{suffix}{variant_suffix}"
    suffix = (
        f"{base}_{preset.compression}"
        if preset.compression and preset.compression != "none"
        else base
    )
    return f"{suffix}{variant_suffix}"


def _compression_display(p: ModelPreset) -> str:
    """COMPRESSION column value: YAML filename when the preset points at a config
    file, otherwise the named compression preset."""
    if p.compression_config:
        return Path(p.compression_config).name
    return p.compression


def _format_text_preset_row(p: ModelPreset, *, show_type: bool = False) -> str:
    """Single line for --list-models --text output."""
    type_col = f"{p.type:11s} " if show_type else ""
    variant_col = (p.variant or "-").ljust(10)
    ctx_col = str(p.max_context_length or "-").rjust(8)
    compression_col = _compression_display(p)
    return f"{type_col}{p.short_name:32s} {variant_col} {compression_col:38s} {ctx_col}  {p.hf_id}"


def _format_text_header(*, show_type: bool = False) -> str:
    type_col = "TYPE        " if show_type else ""
    return f"{type_col}{'SHORT_NAME':32s} {'PLATFORM':10s} {'COMPRESSION':38s} {'CTX':>8s}  HF_ID"


def _format_diffusion_row(p: ModelPreset) -> str:
    return f"{p.short_name:32s} {p.compression:18s} {p.hf_id}"


def _format_diffusion_header() -> str:
    return f"{'SHORT_NAME':32s} {'COMPRESSION':18s} HF_ID"


def _utility_to_export_args(model: UtilityModel) -> list[str]:
    """Build the shell command to export a utility model."""
    return ["uv", "run", model.export_script, "--model", model.hf_id]


def _format_utility_row(u: UtilityModel) -> str:
    platforms = ", ".join(u.platforms)
    return f"{u.short_name:24s} {u.task:16s} {platforms:12s} {u.hf_id:32s} {u.export_script}"


def _format_utility_header() -> str:
    return f"{'SHORT_NAME':24s} {'TASK':16s} {'PLATFORMS':12s} {'HF_ID':32s} EXPORT_SCRIPT"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.model.registry",
        description="Catalogue of supported models and their export presets.",
    )
    parser.add_argument(
        "--type",
        choices=KNOWN_TYPES,
        help="Model type. Required for --list-families, --model-info, --list-variants. "
        "Optional for --list-models (lists all types if omitted).",
    )
    parser.add_argument(
        "--platform",
        choices=KNOWN_VARIANTS,
        help="Platform filter (macOS / iOS). Filters LLM variants and utility model platforms; "
        "diffusion entries are excluded when --platform is set.",
    )
    parser.add_argument(
        "--family",
        help="Family filter (e.g. qwen3). For --type utility, filters by model_type.",
    )
    parser.add_argument(
        "--task",
        help="Task filter for utility models (e.g. embedding, asr, detection).",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Include experimental models (excluded from listings by default).",
    )

    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--list-families",
        action="store_true",
        help="List family names for the given --type.",
    )
    actions.add_argument(
        "--list-models",
        action="store_true",
        help="List all models (optionally filtered by --type / --family / --platform).",
    )
    actions.add_argument(
        "--list-variants",
        metavar="SHORT_NAME",
        help="List the variants supported for the given model.",
    )
    actions.add_argument(
        "--model-info",
        metavar="SHORT_NAME",
        help="Show preset(s) for a single model. With --platform, returns one row.",
    )

    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument(
        "--text",
        action="store_const",
        dest="format",
        const="text",
        help="Human-readable output (default).",
    )
    fmt.add_argument(
        "--json",
        action="store_const",
        dest="format",
        const="json",
        help="Machine-readable JSON output.",
    )
    fmt.add_argument(
        "--tsv",
        action="store_const",
        dest="format",
        const="tsv",
        help="Tab-separated output (one row per preset).",
    )
    fmt.add_argument(
        "--as-export-args",
        action="store_const",
        dest="format",
        const="export-args",
        help="Print the argv to forward to coreai.{type}.export. "
        "Requires --model-info plus --platform for LLM.",
    )
    fmt.add_argument(
        "--as-output-name",
        action="store_const",
        dest="format",
        const="output-name",
        help="Print the basename the export tool would write for "
        "this preset (LLM: bundle directory name; "
        "diffusion: model directory name). "
        "Requires --model-info plus --platform for LLM.",
    )
    parser.set_defaults(format="text")

    return parser


def _require_type(args: argparse.Namespace, action: str) -> str:
    if not args.type:
        sys.stderr.write(f"Error: {action} requires --type {{{','.join(KNOWN_TYPES)}}}\n")
        sys.exit(2)
    return args.type


def _action_list_families(args: argparse.Namespace) -> None:
    model_type = _require_type(args, "--list-families")
    fams = families(model_type, include_experimental=args.experimental)
    if args.format == "json":
        print(json.dumps(fams, indent=2))
    elif args.format == "tsv":
        print("\n".join(fams))
    else:
        if not fams:
            print(f"No families registered for type={model_type}.")
            return
        print(f"{model_type} families ({len(fams)}):")
        for f in fams:
            print(f"  {f}")


def _action_list_utility_models(args: argparse.Namespace) -> None:
    models = filter_utility_models(model_type=args.family, task=args.task, platform=args.platform)
    if args.format == "json":
        print(json.dumps([asdict(u) for u in models], indent=2))
    elif args.format == "tsv":
        for u in models:
            print(
                "\t".join(
                    [
                        u.short_name,
                        u.model_type,
                        u.task,
                        ",".join(u.platforms),
                        u.hf_id,
                        u.export_script,
                    ]
                )
            )
    else:
        if not models:
            print("No utility models match the given filters.")
            return
        print(_format_utility_header())
        for u in models:
            print(_format_utility_row(u))


def _action_list_models(args: argparse.Namespace) -> None:
    if args.type == "utility":
        _action_list_utility_models(args)
        return

    presets = presets_for_type(args.type) if args.type else all_presets()

    presets = filter_presets(
        presets,
        family=args.family,
        variant=args.platform,
        include_experimental=args.experimental,
    )

    if args.format == "json":
        if args.type:
            print(json.dumps([asdict(p) for p in presets], indent=2))
        else:
            util = filter_utility_models(
                model_type=args.family, task=args.task, platform=args.platform
            )
            combined = [asdict(p) for p in presets] + [asdict(u) for u in util]
            print(json.dumps(combined, indent=2))
    elif args.format == "tsv":
        for p in presets:
            cols = [
                p.type,
                p.short_name,
                p.variant or "",
                _compression_display(p),
                str(p.max_context_length or ""),
                p.hf_id,
            ]
            print("\t".join(cols))
        if not args.type:
            for u in filter_utility_models(
                model_type=args.family, task=args.task, platform=args.platform
            ):
                print(
                    "\t".join(
                        [
                            "utility",
                            u.short_name,
                            ",".join(u.platforms),
                            "",
                            "",
                            u.hf_id,
                        ]
                    )
                )
    else:
        if not args.type:
            _print_all_tables(presets, args)
        else:
            if not presets:
                print("No models match the given filters.")
                return
            print(_format_text_header(show_type=False))
            for p in presets:
                print(_format_text_preset_row(p, show_type=False))


def _print_all_tables(presets: list[ModelPreset], args: argparse.Namespace) -> None:
    llm = [p for p in presets if p.type == "llm"]
    diffusion = [p for p in presets if p.type == "diffusion"]
    util = filter_utility_models(model_type=args.family, task=args.task, platform=args.platform)

    if llm:
        print("=== LLM ===")
        print(_format_text_header(show_type=False))
        for p in llm:
            print(_format_text_preset_row(p, show_type=False))

    if diffusion:
        if llm:
            print()
        print("=== Diffusion ===")
        print(_format_diffusion_header())
        for p in diffusion:
            print(_format_diffusion_row(p))

    if util:
        if llm or diffusion:
            print()
        print("=== Image, Text, Audio, and More ===")
        print(_format_utility_header())
        for u in util:
            print(_format_utility_row(u))


def _action_list_variants(args: argparse.Namespace) -> None:
    model_type = _require_type(args, "--list-variants")
    if model_type == "utility":
        model = lookup_utility_model(args.list_variants)
        if not model:
            sys.stderr.write(f"Error: no utility model {args.list_variants!r}\n")
            sys.exit(1)
        platforms = list(model.platforms)
        if args.format == "text":
            print(f"{args.list_variants}: {', '.join(platforms)}")
        elif args.format == "json":
            print(json.dumps(platforms))
        else:
            print("\n".join(platforms))
        return
    matches = [
        p
        for p in presets_for_type(model_type)
        if p.short_name == args.list_variants and (args.experimental or not p.experimental)
    ]
    if not matches:
        sys.stderr.write(f"Error: no model {args.list_variants!r} for type={model_type}\n")
        sys.exit(1)
    variants = sorted({p.variant for p in matches if p.variant is not None})
    if not variants:
        # Diffusion, no variants
        if args.format == "text":
            print(f"{args.list_variants}: no variants (type={model_type})")
        elif args.format == "json":
            print(json.dumps([]))
        else:
            print("")
        return
    if args.format == "json":
        print(json.dumps(variants))
    elif args.format == "tsv":
        print("\n".join(variants))
    else:
        print(f"{args.list_variants}: {', '.join(variants)}")


def _action_utility_model_info(args: argparse.Namespace) -> None:
    model = lookup_utility_model(args.model_info)
    if not model:
        sys.stderr.write(f"Error: no utility model {args.model_info!r}\n")
        sys.exit(1)
    if args.platform is not None and args.platform not in model.platforms:
        sys.stderr.write(f"Error: model {args.model_info!r} has no --platform {args.platform!r}\n")
        sys.exit(1)
    if args.format == "export-args":
        print(" ".join(_utility_to_export_args(model)))
    elif args.format == "json":
        print(json.dumps(asdict(model), indent=2))
    elif args.format == "tsv":
        cols = [
            model.short_name,
            model.model_type,
            model.task,
            ",".join(model.platforms),
            model.hf_id,
            model.export_script,
        ]
        print("\t".join(cols))
    else:
        print(_format_utility_header())
        print(_format_utility_row(model))


def _action_model_info(args: argparse.Namespace) -> None:
    model_type = _require_type(args, "--model-info")

    if model_type == "utility":
        _action_utility_model_info(args)
        return

    matches = [
        p
        for p in presets_for_type(model_type)
        if p.short_name == args.model_info and (args.experimental or not p.experimental)
    ]
    if not matches:
        sys.stderr.write(f"Error: no model {args.model_info!r} for type={model_type}\n")
        sys.exit(1)
    if args.platform is not None:
        matches = [p for p in matches if p.variant == args.platform]
        if not matches:
            sys.stderr.write(
                f"Error: model {args.model_info!r} has no --platform {args.platform!r}\n"
            )
            sys.exit(1)

    if args.format == "export-args":
        if len(matches) > 1:
            sys.stderr.write(
                "Error: --as-export-args needs a single preset; pass --platform to disambiguate.\n"
            )
            sys.exit(2)
        print(" ".join(_preset_to_export_args(matches[0])))
        return

    if args.format == "output-name":
        if len(matches) > 1:
            sys.stderr.write(
                "Error: --as-output-name needs a single preset; pass --platform to disambiguate.\n"
            )
            sys.exit(2)
        print(_preset_to_output_name(matches[0]))
        return

    if args.format == "json":
        if len(matches) == 1:
            print(json.dumps(asdict(matches[0]), indent=2))
        else:
            print(json.dumps([asdict(p) for p in matches], indent=2))
    elif args.format == "tsv":
        for p in matches:
            cols = [
                p.type,
                p.short_name,
                p.variant or "",
                _compression_display(p),
                str(p.max_context_length or ""),
                p.hf_id,
            ]
            print("\t".join(cols))
    else:
        print(_format_text_header(show_type=False))
        for p in matches:
            print(_format_text_preset_row(p, show_type=False))


def _action_summary() -> None:
    """No-args default — short summary, suggest the next commands."""
    print("coreai.model.registry — model catalogue\n")
    for t in ("llm", "diffusion"):
        presets = filter_presets(presets_for_type(t))
        unique_models = len({p.short_name for p in presets})
        fams = families(t)
        print(f"  {t}: {unique_models} models ({len(presets)} presets) across {len(fams)} families")
    util_models = all_utility_models()
    util_tasks = utility_tasks()
    print(f"  utility: {len(util_models)} models across {len(util_tasks)} tasks")
    print("\nTry:")
    print("  coreai.model.registry --list-models --type llm")
    print("  coreai.model.registry --list-models --type utility")
    print("  coreai.model.registry --list-families --type llm")
    print("  coreai.model.registry --model-info qwen3-0.6b --platform iOS")
    print("  coreai.model.registry --model-info clip-vit-b32 --type utility --as-export-args")
    print("  coreai.model.registry --help")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_families:
        _action_list_families(args)
    elif args.list_models:
        _action_list_models(args)
    elif args.list_variants:
        _action_list_variants(args)
    elif args.model_info:
        _action_model_info(args)
    else:
        _action_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
