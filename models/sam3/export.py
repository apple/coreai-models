# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-models",
#     "transformers>=5.5.4,<5.10.1",
#     "tokenizers<0.23.0rc",
#     "huggingface-hub>=1.5.0,<2.0",
#     "torchvision",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# # Force these past the workspace's `coreai-models` pins:
# #   - transformers: workspace is <5.0, SAM3 needs Sam3Model from >=5.5.4
# #   - huggingface-hub: workspace is <1.0, transformers 5.x requires >=1.5.0,<2.0
# override-dependencies = [
#     "transformers>=5.5.4,<5.10.1",
#     "huggingface-hub>=1.5.0,<2.0",
# ]
#
# [tool.uv.sources]
# # Resolve `coreai-models` against the workspace checkout instead of PyPI so
# # this script picks up local edits to `coreai_models.segmentation.*` and
# # `coreai_models.models.ios.sam3.*`.
# coreai-models = { path = "../../python", editable = true }
# ///

"""SAM3 export entry point — runs as a uv inline-script.

The export pipeline lives in ``coreai_models.segmentation.pipeline``; this
script is a thin wrapper whose only job is to give uv the right
``transformers`` resolution (SAM3 needs >= 5.5.4, the workspace pins < 5.0).
The PEP 723 metadata at the top of this file is what makes that possible —
``uv run models/sam3/export.py`` resolves an isolated env per-script.

Usage:
    uv run models/sam3/export.py [--image-size N] [--n-bits N] [--output-dir PATH] ...
"""

import sys


def main() -> None:
    # Lazy import so the inline-script header parses cleanly even if the
    # workspace package isn't on sys.path yet (uv handles that before main()).
    from coreai_models.segmentation.export import main as segmentation_main

    # The shared CLI takes a `model` positional. This script only handles SAM3,
    # so inject "sam3" and forward any user-supplied flags untouched.
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "sam3")
    segmentation_main()


if __name__ == "__main__":
    main()
