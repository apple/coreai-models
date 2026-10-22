# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Load and validate exported model bundles.

A bundle is an export directory: ``metadata.json`` plus the assets it lists.

    bundle = ModelBundle.load("exports/qwen3_0_6b_4bit_dynamic")
    for issue in bundle.validate():
        print(issue)  # error [missing_asset] assets.main → model.aimodel is missing
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coreai_models._constants import METADATA_VERSION, METADATA_VERSION_BATCHED

KNOWN_METADATA_VERSIONS = frozenset({METADATA_VERSION, METADATA_VERSION_BATCHED})

# Kinds the Swift runtime supports (see BundleKind).
KNOWN_KINDS = frozenset(
    {
        "llm",
        "vlm",
        "diffusion",
        "video-diffusion",
        "segmenter",
        "video_segmenter",
        "speech_recognizer",
    }
)

MODEL_ASSET_EXTENSIONS = (".aimodel", ".aimodelc")
_LANGUAGE_FIELDS = ("tokenizer", "vocab_size", "max_context_length")


class BundleError(ValueError):
    """Raised when a bundle can't be loaded."""


@dataclass(frozen=True)
class BundleIssue:
    """A problem found by ``ModelBundle.validate``."""

    severity: str  # "error" or "warning"
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity} [{self.code}] {self.message}"


class BundleValidationError(BundleError):
    """Raised by ``ModelBundle.check`` when the bundle has errors."""

    def __init__(self, path: Path, issues: list[BundleIssue]) -> None:
        self.issues = issues
        errors = "\n  ".join(str(i) for i in issues if i.severity == "error")
        super().__init__(f"{path} failed validation:\n  {errors}")


@dataclass(frozen=True)
class ModelBundle:
    """A loaded model bundle."""

    path: Path
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        """Load the bundle at *path*."""
        path = Path(path)
        if not path.is_dir():
            raise BundleError(f"{path}: no such directory (resolved from {Path.cwd()})")
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            raise BundleError(
                f"{path} has no metadata.json; pass the export directory that contains it"
            )
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as error:
            raise BundleError(
                f"{metadata_path} is not valid JSON "
                f"(line {error.lineno}, column {error.colno}: {error.msg})"
            ) from error
        if not isinstance(metadata, dict):
            raise BundleError(f"{metadata_path} should hold a JSON object")
        return cls(path=path, metadata=metadata)

    @property
    def metadata_version(self) -> str | None:
        return self.metadata.get("metadata_version")

    @property
    def kind(self) -> str | None:
        return self.metadata.get("kind")

    @property
    def name(self) -> str | None:
        return self.metadata.get("name")

    @property
    def assets(self) -> dict[str, str]:
        """Model assets, by role."""
        return dict(self.metadata.get("assets") or {})

    @property
    def auxiliary_assets(self) -> dict[str, str]:
        """Other files the bundle lists, by role."""
        return dict(self.metadata.get("auxiliary_assets") or {})

    @property
    def user_data(self) -> dict[str, str]:
        return dict(self.metadata.get("user_data") or {})

    def asset_path(self, role: str) -> Path:
        """Path to the asset for *role*."""
        if role not in self.assets:
            raise BundleError(f"{self.path} has no asset with role {role!r}")
        return self.path / self.assets[role]

    def validate(self) -> list[BundleIssue]:
        """Return the errors and warnings found in this bundle."""
        return self._validate_common() + self._validate_assets() + self._validate_kind()

    def check(self) -> list[BundleIssue]:
        """Raise ``BundleValidationError`` if the bundle has errors; return any warnings."""
        issues = self.validate()
        if any(i.severity == "error" for i in issues):
            raise BundleValidationError(self.path, issues)
        return issues

    def _validate_common(self) -> list[BundleIssue]:
        issues = []
        version = self.metadata_version
        if not version:
            issues.append(
                BundleIssue(
                    "error",
                    "missing_metadata_version",
                    "metadata.json has no metadata_version (a legacy 0.1 bundle); re-export it",
                )
            )
        elif version not in KNOWN_METADATA_VERSIONS:
            known = ", ".join(sorted(KNOWN_METADATA_VERSIONS))
            issues.append(
                BundleIssue(
                    "warning",
                    "unknown_metadata_version",
                    f"unrecognized metadata_version {version!r} (expected one of {known})",
                )
            )
        for key in ("name", "kind"):
            if not self.metadata.get(key):
                issues.append(
                    BundleIssue("error", f"missing_{key}", f"metadata.json has no {key!r}")
                )
        if self.kind and self.kind not in KNOWN_KINDS:
            issues.append(
                BundleIssue(
                    "warning",
                    "unknown_kind",
                    f"{self.kind!r} isn't a known kind",
                )
            )
        return issues

    def _validate_assets(self) -> list[BundleIssue]:
        issues = []
        if not self.assets:
            issues.append(BundleIssue("error", "no_assets", "metadata.json lists no assets"))
        for role, file in sorted(self.assets.items()):
            path = self.path / file
            if not path.exists():
                issues.append(
                    BundleIssue("error", "missing_asset", f"assets.{role} → {file} is missing")
                )
            elif not (path.is_dir() and path.suffix in MODEL_ASSET_EXTENSIONS):
                issues.append(
                    BundleIssue(
                        "error",
                        "not_a_model_asset",
                        f"assets.{role} → {file} should be an .aimodel or .aimodelc folder; "
                        "list other files under auxiliary_assets",
                    )
                )
        for role, file in sorted(self.auxiliary_assets.items()):
            if not (self.path / file).exists():
                issues.append(
                    BundleIssue(
                        "error",
                        "missing_auxiliary_asset",
                        f"auxiliary_assets.{role} → {file} is missing",
                    )
                )
        listed = set(self.assets.values())
        for path in sorted(self.path.iterdir()):
            if path.suffix in MODEL_ASSET_EXTENSIONS and path.name not in listed:
                issues.append(
                    BundleIssue(
                        "warning",
                        "orphaned_asset",
                        f"{path.name} isn't listed in assets",
                    )
                )
        return issues

    def _validate_kind(self) -> list[BundleIssue]:
        issues = []
        kind = self.kind
        if kind in ("llm", "vlm"):
            language = self.metadata.get("language")
            if not isinstance(language, dict):
                issues.append(
                    BundleIssue(
                        "error", "missing_language_block", f"kind {kind!r} needs a language block"
                    )
                )
            else:
                for field in _LANGUAGE_FIELDS:
                    if not language.get(field):
                        issues.append(
                            BundleIssue(
                                "warning", "missing_language_field", f"language.{field} is missing"
                            )
                        )
                if language.get("embedded_tokenizer") and not (self.path / "tokenizer").is_dir():
                    issues.append(
                        BundleIssue(
                            "error",
                            "missing_tokenizer",
                            "language.embedded_tokenizer is true but there is no tokenizer/ folder",
                        )
                    )
        if kind == "vlm":
            if "vision" not in self.assets:
                issues.append(
                    BundleIssue(
                        "error", "missing_vision_asset", "kind 'vlm' needs an assets.vision entry"
                    )
                )
            if not isinstance(self.metadata.get("vision"), dict):
                issues.append(
                    BundleIssue("error", "missing_vision_block", "kind 'vlm' needs a vision block")
                )
        if kind in ("diffusion", "video-diffusion"):
            diffusion = self.metadata.get("diffusion")
            if not isinstance(diffusion, dict):
                issues.append(
                    BundleIssue(
                        "error",
                        "missing_diffusion_block",
                        f"kind {kind!r} needs a diffusion block",
                    )
                )
            elif not diffusion.get("type"):
                issues.append(
                    BundleIssue("warning", "missing_diffusion_type", "diffusion.type is missing")
                )
        return issues
