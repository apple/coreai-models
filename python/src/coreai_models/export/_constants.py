# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constants for the export pipeline."""

# KV cache names used by the Swift runner. keyCache/valueCache always name the
# growable, full-attention-equivalent cache (the only cache for single-cache
# models; the full-attention cache for dual-cache models like Gemma4).
KEY_CACHE_NAME = "keyCache"
VALUE_CACHE_NAME = "valueCache"

# Secondary KV cache names for models with dual caches (e.g. Gemma4's bounded
# sliding-window cache, which is distinct from its full-attention cache above).
KEY_CACHE_SLIDING_NAME = "slidingKeyCache"
VALUE_CACHE_SLIDING_NAME = "slidingValueCache"

# Trace-time KV cache sequence length. Used only for export/quantization tracing
# to bound peak memory; at inference the actual cache size is determined
# dynamically.
TRACE_KV_CACHE_SEQ_LEN = 2048

# Trace-time `input_ids` length and `position_ids` offset for export/quantization
QUANT_TRACE_QUERY_LEN = 16
QUANT_TRACE_OFFSET = 8

# Default max context length for iOS exports. Users can raise it via
# --max-context-length (up to the model's max_position_embeddings).
IOS_DEFAULT_MAX_CONTEXT_LENGTH = 4096
