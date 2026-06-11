/*!
 * \file xgrammar_c_bridge.cpp
 * \brief C bridge implementation for xgrammar C++ library
 *
 * Copyright 2026 Apple Inc.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause
 *
 * Apple-authored C bridge for the xgrammar C++ library.
 * This file is NOT part of the upstream xgrammar repository.
 * Upstream: https://github.com/mlc-ai/xgrammar
 */

#include "xgrammar_c_bridge.h"
#include <xgrammar/tokenizer_info.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <vector>
#include <string>

using namespace xgrammar;

// Opaque struct implementations
struct XGrammarTokenizerInfo {
    TokenizerInfo cpp_obj;
    XGrammarTokenizerInfo(TokenizerInfo&& obj) : cpp_obj(std::move(obj)) {}
};

struct XGrammarCompiler {
    GrammarCompiler cpp_obj;
    XGrammarCompiler(GrammarCompiler&& obj) : cpp_obj(std::move(obj)) {}
};

struct XGrammarCompiledGrammar {
    CompiledGrammar cpp_obj;
    XGrammarCompiledGrammar(CompiledGrammar&& obj) : cpp_obj(std::move(obj)) {}
};

struct XGrammarMatcher {
    GrammarMatcher cpp_obj;
    XGrammarMatcher(GrammarMatcher&& obj) : cpp_obj(std::move(obj)) {}
};

// Implementation of C bridge functions

XGrammarTokenizerInfo* xgrammar_tokenizer_info_create(
    const char** encoded_vocab,
    int vocab_size,
    XGrammarVocabType vocab_type,
    bool add_prefix_space
) {
    try {
        std::vector<std::string> cpp_vocab;
        cpp_vocab.reserve(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            cpp_vocab.push_back(std::string(encoded_vocab[i]));
        }

        VocabType cpp_vocab_type;
        switch (vocab_type) {
            case XGRAMMAR_VOCAB_RAW:
                cpp_vocab_type = VocabType::RAW;
                break;
            case XGRAMMAR_VOCAB_BYTE_FALLBACK:
                cpp_vocab_type = VocabType::BYTE_FALLBACK;
                break;
            case XGRAMMAR_VOCAB_BYTE_LEVEL:
                cpp_vocab_type = VocabType::BYTE_LEVEL;
                break;
            default:
                cpp_vocab_type = VocabType::RAW;
        }

        TokenizerInfo tok_info(
            cpp_vocab,
            cpp_vocab_type,
            std::nullopt,  // vocab_size
            std::nullopt,  // stop_token_ids
            add_prefix_space
        );

        return new XGrammarTokenizerInfo(std::move(tok_info));
    } catch (...) {
        return nullptr;
    }
}

int xgrammar_tokenizer_info_get_vocab_size(const XGrammarTokenizerInfo* info) {
    try {
        if (!info) return 0;
        return info->cpp_obj.GetVocabSize();
    } catch (...) {
        return 0;
    }
}

void xgrammar_tokenizer_info_free(XGrammarTokenizerInfo* info) {
    delete info;
}

XGrammarCompiler* xgrammar_compiler_create(
    const XGrammarTokenizerInfo* tokenizer_info,
    int max_threads,
    bool cache_enabled
) {
    try {
        if (!tokenizer_info) return nullptr;

        GrammarCompiler compiler(
            tokenizer_info->cpp_obj,
            max_threads,
            cache_enabled,
            -1  // unlimited memory
        );

        return new XGrammarCompiler(std::move(compiler));
    } catch (...) {
        return nullptr;
    }
}

XGrammarCompiledGrammar* xgrammar_compile_json_schema(
    XGrammarCompiler* compiler,
    const char* schema,
    bool any_whitespace,
    bool strict_mode
) {
    try {
        if (!compiler || !schema) return nullptr;

        CompiledGrammar compiled = compiler->cpp_obj.CompileJSONSchema(
            std::string(schema),
            any_whitespace,
            std::nullopt,  // indent
            std::nullopt,  // separators
            strict_mode,
            std::nullopt   // max_whitespace_cnt
        );

        return new XGrammarCompiledGrammar(std::move(compiled));
    } catch (...) {
        return nullptr;
    }
}

size_t xgrammar_compiled_grammar_memory_size(const XGrammarCompiledGrammar* grammar) {
    try {
        if (!grammar) return 0;
        return grammar->cpp_obj.MemorySizeBytes();
    } catch (...) {
        return 0;
    }
}

void xgrammar_compiled_grammar_free(XGrammarCompiledGrammar* grammar) {
    delete grammar;
}

void xgrammar_compiler_free(XGrammarCompiler* compiler) {
    delete compiler;
}

XGrammarMatcher* xgrammar_matcher_create(
    const XGrammarCompiledGrammar* compiled_grammar,
    int max_rollback_tokens
) {
    try {
        if (!compiled_grammar) return nullptr;

        GrammarMatcher matcher(
            compiled_grammar->cpp_obj,
            std::nullopt,  // override_stop_tokens
            false,  // terminate_without_stop_token
            max_rollback_tokens
        );

        return new XGrammarMatcher(std::move(matcher));
    } catch (...) {
        return nullptr;
    }
}

bool xgrammar_matcher_fill_next_token_bitmask(
    XGrammarMatcher* matcher,
    DLTensor* bitmask
) {
    try {
        if (!matcher || !bitmask) return false;
        return matcher->cpp_obj.FillNextTokenBitmask(bitmask, 0, false);
    } catch (...) {
        return false;
    }
}

bool xgrammar_matcher_accept_token(
    XGrammarMatcher* matcher,
    int32_t token_id
) {
    try {
        if (!matcher) return false;
        return matcher->cpp_obj.AcceptToken(token_id, false);
    } catch (...) {
        return false;
    }
}

bool xgrammar_matcher_is_terminated(const XGrammarMatcher* matcher) {
    try {
        if (!matcher) return true;
        return matcher->cpp_obj.IsTerminated();
    } catch (...) {
        return true;
    }
}

void xgrammar_matcher_reset(XGrammarMatcher* matcher) {
    try {
        if (matcher) {
            matcher->cpp_obj.Reset();
        }
    } catch (...) {}
}

void xgrammar_matcher_free(XGrammarMatcher* matcher) {
    delete matcher;
}
