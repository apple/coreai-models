/*!
 * \file xgrammar_c_bridge.h
 * \brief C bridge for xgrammar C++ library
 *
 * Copyright 2026 Apple Inc.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause
 *
 * Apple-authored C bridge for the xgrammar C++ library.
 * This file is NOT part of the upstream xgrammar repository.
 * Upstream: https://github.com/mlc-ai/xgrammar
 *
 * This provides a simple C API that can be easily called from Swift,
 * avoiding complex C++ interop syntax issues.
 */

#ifndef XGRAMMAR_C_BRIDGE_H_
#define XGRAMMAR_C_BRIDGE_H_

#include <stdbool.h>
#include <stdint.h>
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles for C++ objects
typedef struct XGrammarTokenizerInfo XGrammarTokenizerInfo;
typedef struct XGrammarCompiler XGrammarCompiler;
typedef struct XGrammarCompiledGrammar XGrammarCompiledGrammar;
typedef struct XGrammarMatcher XGrammarMatcher;

// Vocabulary type enum
typedef enum {
    XGRAMMAR_VOCAB_RAW = 0,
    XGRAMMAR_VOCAB_BYTE_FALLBACK = 1,
    XGRAMMAR_VOCAB_BYTE_LEVEL = 2
} XGrammarVocabType;

// Create tokenizer info
XGrammarTokenizerInfo* xgrammar_tokenizer_info_create(
    const char** encoded_vocab,
    int vocab_size,
    XGrammarVocabType vocab_type,
    bool add_prefix_space
);

// Get vocabulary size from tokenizer info
int xgrammar_tokenizer_info_get_vocab_size(const XGrammarTokenizerInfo* info);

// Free tokenizer info
void xgrammar_tokenizer_info_free(XGrammarTokenizerInfo* info);

// Create grammar compiler
XGrammarCompiler* xgrammar_compiler_create(
    const XGrammarTokenizerInfo* tokenizer_info,
    int max_threads,
    bool cache_enabled
);

// Compile JSON schema
XGrammarCompiledGrammar* xgrammar_compile_json_schema(
    XGrammarCompiler* compiler,
    const char* schema,
    bool any_whitespace,
    bool strict_mode
);

// Get memory size of compiled grammar
size_t xgrammar_compiled_grammar_memory_size(const XGrammarCompiledGrammar* grammar);

// Free compiled grammar
void xgrammar_compiled_grammar_free(XGrammarCompiledGrammar* grammar);

// Free grammar compiler
void xgrammar_compiler_free(XGrammarCompiler* compiler);

// Create grammar matcher
XGrammarMatcher* xgrammar_matcher_create(
    const XGrammarCompiledGrammar* compiled_grammar,
    int max_rollback_tokens
);

// Fill next token bitmask
bool xgrammar_matcher_fill_next_token_bitmask(
    XGrammarMatcher* matcher,
    DLTensor* bitmask
);

// Accept a token
bool xgrammar_matcher_accept_token(
    XGrammarMatcher* matcher,
    int32_t token_id
);

// Check if terminated
bool xgrammar_matcher_is_terminated(const XGrammarMatcher* matcher);

// Reset matcher
void xgrammar_matcher_reset(XGrammarMatcher* matcher);

// Free grammar matcher
void xgrammar_matcher_free(XGrammarMatcher* matcher);

#ifdef __cplusplus
}
#endif

#endif  // XGRAMMAR_C_BRIDGE_H_
