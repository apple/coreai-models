# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from coreai_models.primitives.ios.quantization import dequantize_per_tensor, quantize_per_tensor

__all__ = ["quantize_per_tensor", "dequantize_per_tensor"]
