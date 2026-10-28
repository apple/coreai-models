// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// The decoder state count `CoreAISequentialEngine` accepts, for token- and embeddings-input
/// decoders alike (issue #276).
@Suite("CoreAISequentialEngine state-name contract")
struct SequentialStateContractTests {
    @Test("accepts a full-attention two-state KV decoder")
    func acceptsTwoStates() throws {
        try CoreAISequentialEngine.validateStateNames(["k_cache", "v_cache"])
    }

    @Test("accepts a hybrid four-state decoder (conv + recurrent states)")
    func acceptsFourStates() throws {
        try CoreAISequentialEngine.validateStateNames(["k_cache", "v_cache", "conv_states", "recurrent_states"])
    }

    @Test("rejects fewer than two states")
    func rejectsOneState() {
        #expect(throws: InferenceRuntimeError.self) {
            try CoreAISequentialEngine.validateStateNames(["k_cache"])
        }
    }

    @Test("rejects more than four states")
    func rejectsFiveStates() {
        #expect(throws: InferenceRuntimeError.self) {
            try CoreAISequentialEngine.validateStateNames(["s0", "s1", "s2", "s3", "s4"])
        }
    }
}
