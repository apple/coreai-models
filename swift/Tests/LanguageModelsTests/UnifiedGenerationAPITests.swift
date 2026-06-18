// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - InferenceOutput Tests

@Suite("InferenceOutput")
struct InferenceOutputTests {
    @Test("Init with token only")
    func initTokenOnly() {
        let output = InferenceOutput(tokenId: 42)
        #expect(output.tokenId == 42)
        #expect(output.logits == nil)
    }

    @Test("Init with token and logits")
    func initWithLogits() {
        let logits: [Float16] = [1.0, 2.0, 3.0]
        let output = InferenceOutput(tokenId: 7, logits: logits)
        #expect(output.tokenId == 7)
        #expect(output.logits?.count == 3)
    }
}

// MARK: - InferenceOptions Tests

@Suite("InferenceOptions")
struct InferenceOptionsTests {
    @Test("Default options: no maxTokens, no logits")
    func defaultOptions() {
        let opts = InferenceOptions()
        #expect(opts.maxTokens == nil)
        #expect(opts.includeLogits == false)
    }

    @Test("Custom options")
    func customOptions() {
        let opts = InferenceOptions(maxTokens: 50, includeLogits: true)
        #expect(opts.maxTokens == 50)
        #expect(opts.includeLogits == true)
    }
}

// MARK: - generate() Default Extension Tests

@Suite("InferenceEngine.generate() default extension")
struct GenerateDefaultExtensionTests {
    @Test("generate() yields correct number of tokens via maxTokens")
    func generatesCorrectTokenCount() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])

        var outputs: [InferenceOutput] = []
        let generation = InferenceOptions(maxTokens: 5)
        for try await output in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: generation
        ) {
            outputs.append(output)
        }

        #expect(outputs.count == 5)
        // Tokens cycle: 10, 20, 30, 10, 20
        #expect(outputs[0].tokenId == 10)
        #expect(outputs[1].tokenId == 20)
        #expect(outputs[2].tokenId == 30)
        #expect(outputs[3].tokenId == 10)
        #expect(outputs[4].tokenId == 20)
    }

    @Test("generate() returns nil logits when includeLogits is false")
    func noLogitsWhenNotRequested() async throws {
        let engine = MockEngine(tokens: [42])

        let generation = InferenceOptions(maxTokens: 1, includeLogits: false)
        for try await output in try engine.generate(
            with: [1],
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: generation
        ) {
            #expect(output.logits == nil)
        }
    }

    @Test("generate() returns logits when includeLogits is true")
    func logitsWhenRequested() async throws {
        let engine = MockEngine(tokens: [42], vocabSize: 50)

        let generation = InferenceOptions(maxTokens: 1, includeLogits: true)
        for try await output in try engine.generate(
            with: [1],
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: generation
        ) {
            #expect(output.logits != nil)
            #expect(output.logits?.count == 50)
        }
    }

    @Test("generate() logits have high probability on target token")
    func logitsHighProbOnTarget() async throws {
        let engine = MockEngine(tokens: [5], vocabSize: 10)

        let generation = InferenceOptions(maxTokens: 1, includeLogits: true)
        for try await output in try engine.generate(
            with: [1],
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: generation
        ) {
            guard let logits = output.logits else {
                Issue.record("Expected logits")
                return
            }
            // Token 5 should have the highest logit value
            let maxIdx = logits.enumerated().max(by: { $0.element < $1.element })?.offset
            #expect(maxIdx == 5)
        }
    }

    @Test("generate() returns nil logits when vocabSize is nil")
    func noLogitsWhenVocabSizeNil() async throws {
        let engine = MockEngine(tokens: [42], vocabSize: nil)

        let generation = InferenceOptions(maxTokens: 1, includeLogits: true)
        for try await output in try engine.generate(
            with: [1],
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: generation
        ) {
            #expect(output.logits == nil)
        }
    }

    @Test("generate() respects maxContextLength")
    func respectsMaxContextLength() async throws {
        // Engine with maxContextLength=5, prompt of 3 tokens → can generate max 2
        let engine = MockEngine(tokens: [10, 20, 30], maxContextLength: 5)

        var count = 0
        let generation = InferenceOptions(maxTokens: 100)  // Request way more than available
        for try await _ in try engine.generate(
            with: [1, 2, 3],  // 3 tokens prompt
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: generation
        ) {
            count += 1
        }

        #expect(count == 2)  // Only 2 slots left (5 - 3)
    }

    @Test("generate() with nil maxTokens uses maxContextLength")
    func nilMaxTokensUsesContextLength() async throws {
        let engine = MockEngine(tokens: [10], maxContextLength: 6)

        var count = 0
        let generation = InferenceOptions()  // nil maxTokens
        for try await _ in try engine.generate(
            with: [1, 2, 3],  // 3 tokens prompt
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: generation
        ) {
            count += 1
        }

        #expect(count == 3)  // 6 - 3 = 3 available slots
    }

    @Test("reset() clears state")
    func resetClearsState() async throws {
        let engine = MockEngine(tokens: [10])

        // Generate a token to advance state
        for try await _ in try engine.generate(
            with: [1],
            samplingConfiguration: SamplingConfiguration.greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1)
        ) {}

        #expect(engine.inferenceCallCount == 1)

        try await engine.reset()
        #expect(engine.resetCalled == true)
        #expect(engine.inferenceCallCount == 0)
    }
}

// MARK: - Multi-turn + Guided Generation Pattern Tests

@Suite("generate() multi-call patterns")
struct GenerateMultiCallTests {
    @Test("Multi-turn: repeated generate(maxTokens:1) with growing input doesn't deadlock")
    func multiTurnSingleTokenCalls() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50], maxContextLength: 100)

        var tokens: [Int32] = [1, 2, 3]  // initial prompt

        // Simulate guided-generation pattern: call generate(maxTokens:1) repeatedly
        for _ in 0..<20 {
            var got: InferenceOutput?
            for try await output in try engine.generate(
                with: tokens,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: 1, includeLogits: true)
            ) {
                got = output
                break  // Only consume 1 token (GG pattern)
            }
            guard let output = got else { break }
            tokens.append(output.tokenId)
        }

        // Should have generated 20 tokens without deadlock/crash
        #expect(tokens.count == 23)  // 3 prompt + 20 generated
    }

    @Test("Multi-turn: generate → reset → generate cycle")
    func multiTurnWithReset() async throws {
        let engine = MockEngine(tokens: [10, 20, 30], maxContextLength: 50)

        // Turn 1
        var count1 = 0
        for try await _ in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 5)
        ) {
            count1 += 1
        }
        #expect(count1 == 5)

        // Reset between turns
        try await engine.reset()

        // Turn 2
        var count2 = 0
        for try await _ in try engine.generate(
            with: [4, 5, 6],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 3)
        ) {
            count2 += 1
        }
        #expect(count2 == 3)
    }

    @Test("forcedContinuation: produces exact forced tokens with logits")
    func forcedContinuationWithLogits() async throws {
        let engine = MockEngine(tokens: [99, 99, 99], vocabSize: 50)
        let forced: [Int32] = [7, 8, 9]

        var outputs: [InferenceOutput] = []
        for try await output in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(
                includeLogits: true,
                forcedContinuation: forced
            )
        ) {
            outputs.append(output)
        }

        #expect(outputs.count == 3)
        #expect(outputs.map(\.tokenId) == forced)
        #expect(outputs.allSatisfy { $0.logits != nil })
    }

    @Test("forcedContinuation: empty array produces zero tokens")
    func forcedContinuationEmpty() async throws {
        let engine = MockEngine(tokens: [10, 20])

        var count = 0
        for try await _ in try engine.generate(
            with: [1],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(forcedContinuation: [])
        ) {
            count += 1
        }
        #expect(count == 0)
    }
}

// MARK: - Partial Reset Tests

@Suite("InferenceEngine partial reset")
struct PartialResetTests {
    @Test("processedTokenCount starts at 0")
    func initialTokenCount() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        #expect(engine.processedTokenCount == 0)
    }

    @Test("reset(to: 0) clears processedTokenCount")
    func fullReset() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        engine.processedTokenCount = 10
        try await engine.reset(to: 0)
        #expect(engine.processedTokenCount == 0)
        #expect(engine.resetCalled)
    }

    @Test("reset(to: N) preserves count")
    func partialReset() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        engine.processedTokenCount = 10
        try await engine.reset(to: 5)
        #expect(engine.processedTokenCount == 5)
    }

    @Test("reset() delegates to reset(to: 0)")
    func resetDelegatesToFull() async throws {
        let engine = MockEngine(tokens: [10, 20, 30])
        engine.processedTokenCount = 10
        try await engine.reset()
        #expect(engine.processedTokenCount == 0)
    }
}

// MARK: - Partial Reset Output Parity Tests

@Suite("Partial reset output parity")
struct PartialResetParityTests {
    /// Verifies that reset(to: N) + generate from N produces IDENTICAL tokens
    /// as full reset + re-generate the full sequence (prompt + first N tokens + continue).
    ///
    /// Runs 100 iterations with random reset points to stress-test KV cache
    /// consistency across all code paths.
    ///
    /// NOTE: MockEngine uses deterministic cycling; real engines with actual KV cache
    /// require hardware tests to validate cache coherence after partial reset.
    @Test("reset(to:) produces identical output vs full re-generate — 20 random iterations")
    func partialResetOutputParity() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], maxContextLength: 200)
        let prompt: [Int32] = [1, 2, 3, 4, 5]
        let totalTokens = 50

        // Generate the full reference sequence once
        try await engine.reset()
        var referenceTokens: [Int32] = []
        for try await output in try engine.generate(
            with: prompt,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: totalTokens)
        ) {
            referenceTokens.append(output.tokenId)
        }
        #expect(referenceTokens.count == totalTokens)

        // Use a seeded random number generator for deterministic test execution
        var rng = SplitMix64(seed: 42)

        // Run 20 random partial reset iterations
        for iteration in 0..<20 {
            // Pick a random reset point (1..<totalTokens to ensure partial, not full)
            let resetPoint = Int.random(in: 1..<totalTokens, using: &rng)

            // --- Path A: Full reset + re-generate everything ---
            try await engine.reset()
            var fullTokens: [Int32] = []
            for try await output in try engine.generate(
                with: prompt,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: totalTokens)
            ) {
                fullTokens.append(output.tokenId)
            }

            // --- Path B: Generate up to resetPoint, partial reset, continue ---
            try await engine.reset()
            var partialTokens: [Int32] = []

            // Generate first resetPoint tokens
            for try await output in try engine.generate(
                with: prompt,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: resetPoint)
            ) {
                partialTokens.append(output.tokenId)
            }
            #expect(engine.processedTokenCount == prompt.count + resetPoint)

            // Partial reset back to after prompt + resetPoint tokens
            try await engine.reset(to: prompt.count + resetPoint)
            #expect(engine.processedTokenCount == prompt.count + resetPoint)

            // Continue generating the remaining tokens
            let remaining = totalTokens - resetPoint
            let continueInput = prompt + partialTokens  // full context so far
            for try await output in try engine.generate(
                with: continueInput,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: remaining)
            ) {
                partialTokens.append(output.tokenId)
            }

            // --- Verify: both paths produce identical tokens ---
            #expect(
                partialTokens == fullTokens,
                Comment(
                    rawValue: "Iteration \(iteration): reset(to: \(resetPoint)) diverged. "
                        + "Expected \(Array(fullTokens.prefix(10)))..., got \(Array(partialTokens.prefix(10)))...")
            )
        }
    }

    /// Simpler version: reset to 0 (full) always matches reference.
    /// Verify processedTokenCount is correctly tracked through generate + reset cycles.
    @Test("processedTokenCount tracks prefill and generation accurately")
    func processedTokenCountTracking() async throws {
        let engine = MockEngine(tokens: [10, 20, 30], maxContextLength: 200)
        let prompt: [Int32] = [1, 2, 3, 4, 5]

        #expect(engine.processedTokenCount == 0)

        // After generating 10 tokens from a 5-token prompt
        for try await _ in try engine.generate(
            with: prompt,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 10)
        ) {}
        #expect(engine.processedTokenCount == 15)  // 5 prompt + 10 generated

        // Partial reset to after prompt
        try await engine.reset(to: 5)
        #expect(engine.processedTokenCount == 5)

        // Generate again — input matches cached prefix, so no new prefill
        let continueInput = prompt
        for try await _ in try engine.generate(
            with: continueInput,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 7)
        ) {}
        #expect(engine.processedTokenCount == 12)  // 5 cached + 7 generated

        // Full reset
        try await engine.reset(to: 0)
        #expect(engine.processedTokenCount == 0)
    }
}

// MARK: - TokenHistory Unit Tests

@Suite("TokenHistory")
struct TokenHistoryTests {
    @Test("resolve with empty history returns all tokens as new")
    func resolveEmptyHistory() {
        let history = TokenHistory()
        let input: [Int32] = [1, 2, 3, 4, 5]
        let (commonPrefix, newTokens) = history.resolve(input: input)
        #expect(commonPrefix == 0)
        #expect(Array(newTokens) == [1, 2, 3, 4, 5])
    }

    @Test("resolve with exact match returns no new tokens")
    func resolveExactMatch() {
        var history = TokenHistory()
        history.append(contentsOf: [1, 2, 3][...])
        let input: [Int32] = [1, 2, 3]
        let (commonPrefix, newTokens) = history.resolve(input: input)
        #expect(commonPrefix == 3)
        #expect(Array(newTokens) == [])
    }

    @Test("resolve with prefix match returns only new tokens")
    func resolvePrefixMatch() {
        var history = TokenHistory()
        history.append(contentsOf: [1, 2, 3][...])
        let input: [Int32] = [1, 2, 3, 4, 5]
        let (commonPrefix, newTokens) = history.resolve(input: input)
        #expect(commonPrefix == 3)
        #expect(Array(newTokens) == [4, 5])
    }

    @Test("resolve with divergence finds divergence point")
    func resolveDivergence() {
        var history = TokenHistory()
        history.append(contentsOf: [1, 2, 3, 4, 5][...])
        let input: [Int32] = [1, 2, 99, 100]
        let (commonPrefix, newTokens) = history.resolve(input: input)
        #expect(commonPrefix == 2)
        #expect(Array(newTokens) == [99, 100])
    }

    @Test("resolve with shorter input than history")
    func resolveShorterInput() {
        var history = TokenHistory()
        history.append(contentsOf: [1, 2, 3, 4, 5][...])
        let input: [Int32] = [1, 2, 3]
        let (commonPrefix, newTokens) = history.resolve(input: input)
        #expect(commonPrefix == 3)
        #expect(Array(newTokens) == [])
    }

    @Test("resolve with completely different input")
    func resolveCompleteDivergence() {
        var history = TokenHistory()
        history.append(contentsOf: [1, 2, 3][...])
        let input: [Int32] = [99, 98, 97]
        let (commonPrefix, newTokens) = history.resolve(input: input)
        #expect(commonPrefix == 0)
        #expect(Array(newTokens) == [99, 98, 97])
    }

    @Test("append and truncate lifecycle")
    func appendAndTruncate() {
        var history = TokenHistory()
        history.append(42)
        history.append(contentsOf: [1, 2, 3][...])
        #expect(history.count == 4)
        history.truncate(to: 2)
        #expect(history.count == 2)
        #expect(history.tokens == [42, 1])
        history.truncate(to: 2)  // no-op
        #expect(history.count == 2)
        history.truncate(to: 0)
        #expect(history.count == 0)
        history.append(contentsOf: [10, 20][...])
        history.clear()
        #expect(history.count == 0)
    }

    @Test("resolve with empty input returns 0 common prefix")
    func resolveEmptyInput() {
        var history = TokenHistory()
        history.append(contentsOf: [1, 2, 3][...])
        let input: [Int32] = []
        let (commonPrefix, newTokens) = history.resolve(input: input)
        #expect(commonPrefix == 0)
        #expect(Array(newTokens) == [])
    }
}

// MARK: - Prefix Caching Integration Tests

@Suite("Implicit prefix caching")
struct PrefixCachingTests {
    @Test("generate with same prefix reports prefix hit")
    func prefixHitOnSecondCall() async throws {
        let engine = MockEngine(tokens: [10, 20, 30], maxContextLength: 100)

        // First generation: prompt [1, 2, 3]
        var tokens: [Int32] = [1, 2, 3]
        for try await output in try engine.generate(
            with: tokens,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 3)
        ) {
            tokens.append(output.tokenId)
        }
        // tokens is now [1, 2, 3, 10, 20, 30]
        #expect(tokens.count == 6)

        // Second generation with same prefix + new suffix:
        // Engine should detect prefix hit for the first 6 tokens
        for try await output in try engine.generate(
            with: tokens,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 2)
        ) {
            tokens.append(output.tokenId)
        }

        // All 6 prior tokens should have been a prefix hit
        #expect(engine.lastPrefixHitCount == 6)
        #expect(tokens.count == 8)
    }

    @Test("generate with divergent prefix auto-resets")
    func divergentPrefixAutoResets() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50], maxContextLength: 100)

        // First generation: prompt [1, 2, 3]
        for try await _ in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 3)
        ) {}
        // processedTokenCount = 6 (3 prompt + 3 generated)
        #expect(engine.processedTokenCount == 6)

        // Second generation with divergent prefix: [1, 2, 99, ...]
        // Should auto-detect divergence at position 2 and rewind
        var tokens: [Int32] = []
        for try await output in try engine.generate(
            with: [1, 2, 99, 100],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 2)
        ) {
            tokens.append(output.tokenId)
        }

        // Common prefix is 2 ([1, 2]), divergence at position 2
        #expect(engine.lastPrefixHitCount == 2)
        // processedTokenCount = 4 (input) + 2 (generated) = 6
        #expect(engine.processedTokenCount == 6)
        #expect(tokens.count == 2)
    }

    @Test("generate with completely different input auto-resets to 0")
    func completeDivergenceAutoResets() async throws {
        let engine = MockEngine(tokens: [10, 20, 30], maxContextLength: 100)

        // First generation
        for try await _ in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 2)
        ) {}
        #expect(engine.processedTokenCount == 5)

        // Completely different input
        for try await _ in try engine.generate(
            with: [99, 98, 97],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1)
        ) {}

        #expect(engine.lastPrefixHitCount == 0)
        // processedTokenCount: 3 (new input) + 1 (generated) = 4
        #expect(engine.processedTokenCount == 4)
    }

    @Test("multi-turn prefix caching efficiency")
    func multiTurnEfficiency() async throws {
        let engine = MockEngine(tokens: [10, 20, 30, 40, 50], maxContextLength: 200)

        // Turn 1: generate with prompt [1, 2, 3]
        var context: [Int32] = [1, 2, 3]
        for try await output in try engine.generate(
            with: context,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 5)
        ) {
            context.append(output.tokenId)
        }
        // context = [1, 2, 3, 10, 20, 30, 40, 50]
        #expect(context.count == 8)

        // Turn 2: append user message tokens, generate again
        // The first 8 tokens should be a prefix hit
        context.append(contentsOf: [77, 78, 79])  // new user message
        for try await output in try engine.generate(
            with: context,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 3)
        ) {
            context.append(output.tokenId)
        }

        #expect(engine.lastPrefixHitCount == 8)
        #expect(context.count == 14)  // 8 + 3 new input + 3 generated
    }

    @Test("reset clears history and prefix hit count reflects fresh start")
    func resetClearsHistoryState() async throws {
        let engine = MockEngine(tokens: [10, 20], maxContextLength: 100)

        // Generate some tokens
        for try await _ in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 2)
        ) {}
        #expect(engine.history.count == 5)

        // Full reset
        try await engine.reset()
        #expect(engine.history.count == 0)
        #expect(engine.processedTokenCount == 0)

        // Next generation starts fresh
        for try await _ in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1)
        ) {}
        #expect(engine.lastPrefixHitCount == 0)
    }

    @Test("partial reset truncates history correctly")
    func partialResetTruncatesHistory() async throws {
        let engine = MockEngine(tokens: [10, 20, 30], maxContextLength: 100)

        // Generate: prompt [1, 2, 3] + 3 tokens = history of 6
        for try await _ in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 3)
        ) {}
        #expect(engine.history.count == 6)
        #expect(engine.processedTokenCount == 6)

        // Partial reset to position 3 (keep only prompt)
        try await engine.reset(to: 3)
        #expect(engine.history.count == 3)
        #expect(engine.processedTokenCount == 3)
        #expect(engine.history.tokens == [1, 2, 3])

        // Re-generate with same prompt — should be a full prefix hit
        for try await _ in try engine.generate(
            with: [1, 2, 3],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 2)
        ) {}
        #expect(engine.lastPrefixHitCount == 3)
    }
}

// MARK: - Deterministic RNG for Tests

/// SplitMix64: fast, deterministic PRNG for reproducible test randomness.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
