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
# #   - transformers: workspace is <5.0, SAM3 needs Sam3VideoModel from >=5.5.4
# #   - huggingface-hub: workspace is <1.0, transformers 5.x requires >=1.5.0,<2.0
# override-dependencies = [
#     "transformers>=5.5.4,<5.10.1",
#     "huggingface-hub>=1.5.0,<2.0",
# ]
#
# [tool.uv.sources]
# # Resolve `coreai-models` against the workspace checkout instead of PyPI so
# # this script picks up local edits to `coreai_models.segmentation.*`.
# coreai-models = { path = "../../python", editable = true }
# ///

"""SAM3 video segmentation export entry point — runs as a uv inline-script.

Thin wrapper over ``coreai_models.segmentation.export``, whose only job is to
give uv the right ``transformers`` resolution (SAM3 needs >= 5.5.4, the
workspace pins < 5.0). Passes ``--video`` so the shared CLI selects the video
pipeline.

Usage:
    uv run models/sam3_video/export.py [--dtype float16] [--output-dir PATH] ...
"""

import sys


def main() -> None:
    # Lazy import so the inline-script header parses cleanly even if the
    # workspace package isn't on sys.path yet (uv handles that before main()).
    from coreai_models.segmentation.export import main as segmentation_main

    if "--video" not in sys.argv:
        sys.argv.insert(1, "--video")
    segmentation_main()


if __name__ == "__main__":
    main()
