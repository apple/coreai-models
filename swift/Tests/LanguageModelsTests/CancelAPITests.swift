// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("InferenceEngine cancel API")
struct CancelAPITests {
    @Test("isBusy is false when idle")
    func idleNotBusy() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        #expect(!engine.isBusy)
    }

    @Test("cancel() is safe when idle")
    func cancelWhenIdle() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        try await engine.cancel()
        #expect(!engine.isBusy)
    }

    @Test("engine is busy during generation")
    func busyDuringGeneration() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        let stream = try await engine.generate(
            with: [1],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 100)
        )
        #expect(engine.isBusy)

        // Consume to release
        for try await _ in stream {}
        #expect(!engine.isBusy)
    }

    @Test("cancel() stops generation and marks .cancelled")
    func cancelStopsGeneration() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        let stream = try await engine.generate(
            with: [1],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 100)
        )
        #expect(engine.isBusy)

        try await engine.cancel()
        #expect(!engine.isBusy)

        // Stream should yield nil on next iteration
        var count = 0
        for try await _ in stream {
            count += 1
        }
        #expect(count == 0)
        #expect(stream.stopReason == .cancelled)
    }

    @Test("engine becomes idle after generation completes naturally")
    func idleAfterNaturalCompletion() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        let stream = try await engine.generate(
            with: [1],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 3)
        )
        #expect(engine.isBusy)

        for try await _ in stream {}

        #expect(!engine.isBusy)
        #expect(stream.stopReason == .maxTokens)
    }

    @Test("reset() cancels active generation")
    func resetCancelsGeneration() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        let stream = try await engine.generate(
            with: [1],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 100)
        )
        #expect(engine.isBusy)

        try await engine.reset()
        #expect(!engine.isBusy)
        #expect(engine.resetCalled)

        // Stream should yield nil
        var count = 0
        for try await _ in stream {
            count += 1
        }
        #expect(count == 0)
    }

    @Test("GenerationToken starts not cancelled")
    func tokenStartsNotCancelled() {
        let token = GenerationToken()
        #expect(!token.isCancelled)
    }

    @Test("GenerationToken cancel() sets isCancelled")
    func tokenCancelSetsFlag() {
        let token = GenerationToken()
        token.cancel()
        #expect(token.isCancelled)
    }

    @Test("GenerationToken cancel() is idempotent")
    func tokenCancelIdempotent() {
        let token = GenerationToken()
        token.cancel()
        token.cancel()
        #expect(token.isCancelled)
    }

    // MARK: - Back-to-back turn serialization

    @Test("back-to-back generate() calls do not crash")
    func backToBackGenerate() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50])

        // Turn 1: consume partially (simulates EOS break mid-stream)
        let stream1 = try await engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 5)
        )
        var count1 = 0
        for try await _ in stream1 {
            count1 += 1
            if count1 == 2 { break }
        }

        // Turn 2: immediately start next generation — must not crash
        let stream2 = try await engine.generate(
            with: [1, 2, 3, 10, 20, 4, 5],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 3)
        )
        var count2 = 0
        for try await _ in stream2 {
            count2 += 1
        }
        #expect(count2 == 3)
    }

    @Test("rapid-fire multi-turn stress (10 turns, no delay)")
    func rapidFireMultiTurn() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50])

        for turn in 0..<10 {
            let prompt = Array(0..<(turn + 1)).map { Int32($0) }
            let stream = try await engine.generate(
                with: prompt,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: 3)
            )
            var tokens: [Int32] = []
            for try await output in stream {
                tokens.append(output.tokenId)
            }
            #expect(tokens.count <= 3)
        }
    }

    @Test("generate() after cancel() works cleanly")
    func generateAfterCancel() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])

        let stream1 = try await engine.generate(
            with: [1],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 100)
        )
        _ = stream1  // don't consume at all

        try await engine.cancel()
        #expect(!engine.isBusy)

        // Should work immediately after cancel
        let stream2 = try await engine.generate(
            with: [1, 2],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 2)
        )
        var count = 0
        for try await _ in stream2 {
            count += 1
        }
        #expect(count == 2)
    }
}
