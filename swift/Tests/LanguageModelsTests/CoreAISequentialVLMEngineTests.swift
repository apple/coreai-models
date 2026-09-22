// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("CoreAISequentialVLMEngine state-name contract")
struct CoreAISequentialVLMEngineTests {
    @Test("accepts a full-attention two-state KV decoder")
    func acceptsTwoStates() throws {
        try CoreAISequentialVLMEngine.validateLLMStateNames(["k_cache", "v_cache"])
    }

    @Test("accepts a hybrid four-state decoder (conv + recurrent states)")
    func acceptsFourStates() throws {
        // Hybrid decoders carry conv/recurrent states beyond the two KV-cache states; the
        // engine previously rejected anything but exactly two states.
        try CoreAISequentialVLMEngine.validateLLMStateNames(
            ["k_cache", "v_cache", "conv_states", "recurrent_states"])
    }

    @Test("rejects fewer than two states")
    func rejectsOneState() {
        #expect(throws: InferenceRuntimeError.self) {
            try CoreAISequentialVLMEngine.validateLLMStateNames(["k_cache"])
        }
    }

    @Test("rejects more than four states")
    func rejectsFiveStates() {
        #expect(throws: InferenceRuntimeError.self) {
            try CoreAISequentialVLMEngine.validateLLMStateNames(
                ["s0", "s1", "s2", "s3", "s4"])
        }
    }
}
