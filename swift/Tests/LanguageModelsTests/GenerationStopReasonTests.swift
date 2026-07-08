// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("Generation StopReason")
struct GenerationStopReasonTests {
    @Test("iterates all tokens and sets .maxTokens on normal completion")
    func normalCompletion() async throws {
        let engine = MockEngine(tokens: [1, 2, 3], maxContextLength: 100)
        let stream = try await engine.generate(
            with: [0],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 3)
        )

        var collected: [Int32] = []
        for try await output in stream {
            collected.append(output.tokenId)
        }

        #expect(collected == [1, 2, 3])
        #expect(stream.stopReason == .maxTokens)
    }

    @Test("stopReason is .eos when set by decoder")
    func eosSetByDecoder() async throws {
        let eosToken: Int32 = 99
        let engine = MockEngine(tokens: [10, 20, eosToken, 40], maxContextLength: 100)
        let stream = try await engine.generate(
            with: [0],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 10)
        )

        for try await output in stream {
            if output.tokenId == eosToken {
                stream.setStopReason(.eos)
                break
            }
        }

        #expect(stream.stopReason == .eos)
    }
}
