# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""SAM3 Lite (`facebook/sam3`) for iOS export.

The submodules mirror the HuggingFace `Sam3Model` structure, but every
component is in BC1S layout with `nn.Linear` replaced by `nn.Conv2d(1x1)`,
GELU/LayerNorm/RoPE re-implemented in fp16-safe primitives, and
window-attention partitioning kept at rank 4.
"""

from coreai_models.models.ios.sam3.sam3_reauthored import Sam3Lite

__all__ = ["Sam3Lite"]
