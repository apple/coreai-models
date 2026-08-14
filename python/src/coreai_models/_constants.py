# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph and runner contract constants.

A leaf module: imported by both ``models/`` and ``export/``, imports nothing from
either.
"""

# Graph name for a single-graph (macOS) export. iOS uses its own entrypoint names.
MAIN_GRAPH_NAME = "main"

# KV cache names used by the Swift runner
KEY_CACHE_NAME = "keyCache"
VALUE_CACHE_NAME = "valueCache"

# Trace-time KV cache sequence length, to bound peak trace memory. At inference the
# cache size is dynamic.
TRACE_KV_CACHE_SEQ_LEN = 2048

# Trace-time `input_ids` length and `position_ids` offset for export/quantization
QUANT_TRACE_QUERY_LEN = 16
QUANT_TRACE_OFFSET = 8

# Default max context length for iOS exports. Users can raise it via
# --max-context-length (up to the model's max_position_embeddings).
IOS_DEFAULT_MAX_CONTEXT_LENGTH = 4096


# ---------------------------------------------------------------------------
# iOS graph I/O names (must match what the Swift runner expects)
# ---------------------------------------------------------------------------

LOAD_EMBEDDINGS_FUNCTION_NAME = "load_embeddings"
GATHER_EMBEDDINGS_FUNCTION_NAME = "gather_embeddings"
EXTEND_FUNCTION_NAME = "extend"
PROMPT_OPT_FUNCTION_NAME = "prompt_opt"

EMBEDDING_TABLE_INPUT_NAME = "embedding_table"
LOAD_EMBEDDINGS_OUTPUT_NAME = "embedding_table"
TOKEN_IDS_INPUT_NAME = "in_new_token_ids"
GATHERED_EMBEDDINGS_OUTPUT_NAME = "gathered_embeddings"

TRANSFORMER_INPUT_NAME = "transformer_input"
POSITION_IDS_INPUT_NAME = "position_ids"
IN_STEP_INPUT_NAME = "in_step"
CAUSAL_MASK_INPUT_NAME = "causal_mask"
KEY_CACHE_INPUT_NAME = "key_cache"
VALUE_CACHE_INPUT_NAME = "value_cache"
KEY_CACHE_OUTPUT_NAME = "new_k_cache"
VALUE_CACHE_OUTPUT_NAME = "new_v_cache"
OUTPUT_LOGITS_NAME = "out_logits"

# Whether exports embed debug information into .aimodel. Off by default, which puts
# the converter in RELEASE mode and embeds minimum debug information. The
# --include-debug-info flag turns it on (DEBUG mode).
DEFAULT_INCLUDE_DEBUG_INFO = False
