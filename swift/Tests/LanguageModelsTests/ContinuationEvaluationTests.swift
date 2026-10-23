// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - ContinuationEncoding Tests

@Suite("ContinuationEncoding")
struct ContinuationEncodingTests {
    @Test("Basic continuation encoding splits tokens correctly")
    func basicContinuationEncoding() {
        let tokenizer = MockTokenizer()
        let encoding = ContinuationEncoding(
            context: "Hello",
            continuation: " world",
            tokenizer: tokenizer
        )

        // MockTokenizer encodes as UTF-8 bytes
        // "Hello" = [72, 101, 108, 108, 111] = 5 tokens
        // " world" = [32, 119, 111, 114, 108, 100] = 6 tokens

        #expect(encoding.contextTokens.count == 5)
        #expect(encoding.continuationTokens.count == 6)
        #expect(encoding.tokens.count == 11)
    }

    @Test("Whole tokens equals context + continuation encoding")
    func preservesWholeString() {
        let tokenizer = MockTokenizer()
        let encoding = ContinuationEncoding(
            context: "The answer is",
            continuation: " A",
            tokenizer: tokenizer
        )

        // Verify whole = context + continuation
        let expectedWhole = tokenizer.encode(text: "The answer is A").map { Int32($0) }
        #expect(encoding.tokens == expectedWhole)
    }

    @Test("Empty context produces continuation-only encoding")
    func emptyContext() {
        let tokenizer = MockTokenizer()
        let encoding = ContinuationEncoding(
            context: "",
            continuation: "answer",
            tokenizer: tokenizer
        )

        #expect(encoding.contextTokens.count == 0)
        #expect(encoding.continuationTokens.count == 6)  // "answer" = 6 UTF-8 bytes
        #expect(encoding.tokens.count == 6)
    }

    @Test("Single character continuation produces single token")
    func singleCharacterContinuation() {
        let tokenizer = MockTokenizer()
        let encoding = ContinuationEncoding(
            context: "Answer: ",
            continuation: "A",
            tokenizer: tokenizer
        )

        #expect(encoding.continuationTokens.count == 1)
        #expect(encoding.continuationTokens.first == Int32(UInt8(ascii: "A")))
    }

    @Test("Encoding round-trips correctly through decode")
    func encodingRoundTrip() {
        let tokenizer = MockTokenizer()
        let context = "What is 2+2? Answer: "
        let continuation = "4"

        let encoding = ContinuationEncoding(
            context: context,
            continuation: continuation,
            tokenizer: tokenizer
        )

        let decodedContext = tokenizer.decode(tokens: encoding.contextTokens.map { Int($0) })
        let decodedContinuation = tokenizer.decode(tokens: encoding.continuationTokens.map { Int($0) })
        let decodedWhole = tokenizer.decode(tokens: encoding.tokens.map { Int($0) })

        #expect(decodedContext == context)
        #expect(decodedContinuation == continuation)
        #expect(decodedWhole == context + continuation)
    }

    @Test("Continuation start index is correct for non-merging tokenizer")
    func continuationStartIndex() {
        let tokenizer = MockTokenizer()
        let encoding = ContinuationEncoding(
            context: "Hello",
            continuation: " world",
            tokenizer: tokenizer
        )

        // MockTokenizer doesn't merge, so divergence is at context length
        #expect(encoding.continuationStartIndex == 5)  // "Hello" = 5 bytes
    }
}

// MARK: - ContinuationEvaluationResult Tests

@Suite("ContinuationEvaluationResult")
struct ContinuationEvaluationResultTests {
    @Test("Log probability with uniform distribution")
    func logProbabilityUniformDistribution() {
        // Uniform logits = equal probability for all tokens
        let vocabSize = 10
        let uniformLogits = [Float16](repeating: 0.0, count: vocabSize)

        let result = ContinuationEvaluationResult(
            contextTokens: [1, 2, 3],
            continuationTokens: [0, 1, 2],
            logits: [uniformLogits, uniformLogits, uniformLogits]
        )

        // With uniform distribution, each token has probability 1/vocabSize = 0.1
        // Total log prob = 3 * log(0.1) ≈ -6.908
        let expectedLogProb = 3.0 * log(0.1)
        #expect(abs(result.logProbability() - expectedLogProb) < 0.01)
    }

    @Test("Perplexity equals vocab size for uniform distribution")
    func perplexityUniformDistribution() {
        let vocabSize = 10
        let uniformLogits = [Float16](repeating: 0.0, count: vocabSize)

        let result = ContinuationEvaluationResult(
            contextTokens: [1, 2, 3],
            continuationTokens: [0, 1, 2],
            logits: [uniformLogits, uniformLogits, uniformLogits]
        )

        // Perplexity = exp(-avgLogProb) = exp(-log(0.1)) = 10
        #expect(abs(result.perplexity() - 10.0) < 0.01)
    }

    @Test("High probability target has near-1.0 probability")
    func targetProbabilitiesHighProbability() {
        // Create logits where token 5 has very high probability
        var logits = [Float16](repeating: -100.0, count: 10)
        logits[5] = 100.0

        let result = ContinuationEvaluationResult(
            contextTokens: [1, 2],
            continuationTokens: [5],  // Target is the high-probability token
            logits: [logits]
        )

        let probs = result.targetProbabilities()
        #expect(probs.count == 1)
        #expect(probs[0] > 0.99)
    }

    @Test("Low probability target has near-0.0 probability")
    func targetProbabilitiesLowProbability() {
        // Create logits where token 0 has very high probability
        var logits = [Float16](repeating: -100.0, count: 10)
        logits[0] = 100.0

        let result = ContinuationEvaluationResult(
            contextTokens: [1, 2],
            continuationTokens: [5],  // Target is NOT the high-probability token
            logits: [logits]
        )

        let probs = result.targetProbabilities()
        #expect(probs.count == 1)
        #expect(probs[0] < 0.01)
    }

    @Test("Average log probability divides by token count")
    func averageLogProbability() {
        let vocabSize = 10
        let uniformLogits = [Float16](repeating: 0.0, count: vocabSize)

        let result = ContinuationEvaluationResult(
            contextTokens: [1],
            continuationTokens: [0, 1],
            logits: [uniformLogits, uniformLogits]
        )

        // Average log prob = log(0.1) for uniform over 10 tokens
        let expectedAvgLogProb = log(0.1)
        #expect(abs(result.averageLogProbability() - expectedAvgLogProb) < 0.01)
    }

    @Test("Empty continuation returns zero for all metrics")
    func emptyContinuation() {
        let result = ContinuationEvaluationResult(
            contextTokens: [1, 2, 3],
            continuationTokens: [],
            logits: []
        )

        #expect(result.logProbability() == 0.0)
        #expect(result.averageLogProbability() == 0.0)
        #expect(result.targetProbabilities().isEmpty)
    }

    @Test("Invalid token indices are handled gracefully")
    func invalidTokenIndices() {
        let logits = [Float16](repeating: 0.0, count: 10)

        // Test out of bounds (100) and negative (-1) token indices
        let result = ContinuationEvaluationResult(
            contextTokens: [1, 2],
            continuationTokens: [100, -1],  // Both invalid!
            logits: [logits, logits]
        )

        // Should not crash, should skip invalid tokens
        #expect(result.logProbability() == 0.0)

        // targetProbabilities should return 0.0 for invalid tokens
        let probs = result.targetProbabilities()
        #expect(probs.count == 2)
        #expect(probs[0] == 0.0)  // Out of bounds
        #expect(probs[1] == 0.0)  // Negative
    }

    @Test("Multiple calls return consistent results")
    func consistentResults() {
        let vocabSize = 10
        let uniformLogits = [Float16](repeating: 0.0, count: vocabSize)

        let result = ContinuationEvaluationResult(
            contextTokens: [1, 2, 3],
            continuationTokens: [0, 1, 2],
            logits: [uniformLogits, uniformLogits, uniformLogits]
        )

        // Call multiple methods - results should be consistent
        let logProb1 = result.logProbability()
        let logProb2 = result.logProbability()
        let probs = result.targetProbabilities()
        let perplexity = result.perplexity()

        #expect(logProb1 == logProb2)
        #expect(probs.count == 3)
        #expect(perplexity > 0)
    }
}

// MARK: - Integration Tests

@Suite("ContinuationEvaluation Integration")
struct ContinuationEvaluationIntegrationTests {
    @Test("MMLU-style single answer produces single continuation token")
    func typicalMMLUScenario() {
        let tokenizer = MockTokenizer()

        let context = "Question: What is the capital of France?\nA. Berlin\nB. Paris\nC. London\nD. Madrid\nAnswer: "
        let continuation = "B"

        let encoding = ContinuationEncoding(
            context: context,
            continuation: continuation,
            tokenizer: tokenizer
        )

        // Continuation should be single token for "B"
        #expect(encoding.continuationTokens.count == 1)
    }

    @Test("Multiple answer choices produce unique tokens")
    func multipleAnswerChoicesComparison() {
        let tokenizer = MockTokenizer()
        let context = "Answer: "

        let encodingA = ContinuationEncoding(context: context, continuation: "A", tokenizer: tokenizer)
        let encodingB = ContinuationEncoding(context: context, continuation: "B", tokenizer: tokenizer)
        let encodingC = ContinuationEncoding(context: context, continuation: "C", tokenizer: tokenizer)
        let encodingD = ContinuationEncoding(context: context, continuation: "D", tokenizer: tokenizer)

        // All should have same context length
        #expect(encodingA.contextTokens.count == encodingB.contextTokens.count)
        #expect(encodingB.contextTokens.count == encodingC.contextTokens.count)
        #expect(encodingC.contextTokens.count == encodingD.contextTokens.count)

        // All continuations should be single token
        #expect(encodingA.continuationTokens.count == 1)
        #expect(encodingB.continuationTokens.count == 1)
        #expect(encodingC.continuationTokens.count == 1)
        #expect(encodingD.continuationTokens.count == 1)

        // Continuation tokens should be different
        let tokens = Set([
            encodingA.continuationTokens[0],
            encodingB.continuationTokens[0],
            encodingC.continuationTokens[0],
            encodingD.continuationTokens[0],
        ])
        #expect(tokens.count == 4)
    }

    @Test("Divergence detection handles BPE-style token merging")
    func divergenceDetectionWithMerging() {
        // Simulate a tokenizer where " " + "B" merges into " B" token
        let mergingTokenizer = MergingMockTokenizer()

        let context = "Answer: "  // Ends with space
        let continuation = "B"

        let encoding = ContinuationEncoding(
            context: context,
            continuation: continuation,
            tokenizer: mergingTokenizer
        )

        // MergingMockTokenizer behavior:
        // encode("Answer: ") = [65, 110, 115, 119, 101, 114, 58, 32]  (8 tokens, ends with space=32)
        // encode("Answer: B") = [65, 110, 115, 119, 101, 114, 58, 201] (8 tokens, " B"=201 merged)
        // " B" token = 200 + (B - A) = 200 + 1 = 201
        // Divergence at position 7 (the last token differs)

        // Continuation should include the merged token
        #expect(encoding.continuationTokens.count == 1)

        // The merged token ID should be 201 (200 + offset for B)
        // " A" = 200, " B" = 201, " C" = 202, etc.
        #expect(encoding.continuationTokens[0] == 201)

        // Context tokens should be 7 (up to the divergence point)
        #expect(encoding.contextTokens.count == 7)
        #expect(encoding.continuationStartIndex == 7)
    }

    @Test("Divergence detection when context is subset of whole")
    func divergenceWithShorterContext() {
        // Simulate where context encoding is shorter than whole
        let mergingTokenizer = MergingMockTokenizer()

        // With merging, "Answer:" (no trailing space) + " B" should work
        let encoding = ContinuationEncoding(
            context: "Answer:",
            continuation: " B",
            tokenizer: mergingTokenizer
        )

        // Whole should contain more tokens than if we just concatenated bytes
        #expect(encoding.tokens.count >= encoding.contextTokens.count)
        #expect(encoding.continuationTokens.count >= 1)
    }
}

// MARK: - Raw Token Scoring Tests

/// Records what each `generate` call was fed, so a test can assert on the context and the forced
/// continuation a scoring run hands the engine.
private final class RecordingEngine: MockEngine, @unchecked Sendable {
    private(set) var fedInputs: [[TokenId]] = []
    private(set) var fedContinuations: [[TokenId]?] = []

    override func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        fedInputs.append(input)
        fedContinuations.append(inferenceOptions.forcedContinuation)
        return try await super.generate(
            with: input, samplingConfiguration: samplingConfiguration, inferenceOptions: inferenceOptions)
    }
}

/// How `TextGenerator` scores and generates from pre-tokenized ids. The decisions and refusals of
/// llm-runner's raw-token options are covered in `LLMRunnerTests`.
@Suite("Raw token evaluation")
struct RawTokenEvaluationTests {
    private func makeGenerator(engine: MockEngine) async throws -> TextGenerator {
        try await TextGeneratorBuilder()
            .withInferenceEngine(engine)
            .withSampling(configuration: .greedy)
            .withDecoding(type: .vanilla, parameters: DecodingParameters())
            .withTokenizer(MockTokenizer())
            .build()
    }

    // MARK: Generating

    @Test("Generating from raw tokens writes a tokens document for each generated position")
    func generationWritesTokensDocument() async throws {
        let engine = RecordingEngine(tokens: [10, 20, 30], vocabSize: 100)
        let generator = try await makeGenerator(engine: engine)

        let result = try await generator.generateWithLogits(
            input: .tokens([1, 2, 3, 4]), maxTokens: 3, stopSequences: StopSequences(sequences: []))
        #expect(engine.fedInputs == [[1, 2, 3, 4]])
        #expect(engine.fedContinuations == [nil])

        let tempFile = TempFile()
        defer { tempFile.cleanup() }
        try LogitsWriter.handleOutput(
            logits: result.logits,
            generatedTokenIds: result.tokenIds,
            tokenizer: MockTokenizer(),
            saveLogitsLength: .count(5),
            saveJsonPath: tempFile.path,
            printToConsole: false
        )

        let decoded = try JSONDecoder().decode(LogitsOutput.self, from: try tempFile.readData())
        #expect(decoded.tokens.map(\.tokenId) == [10, 20, 30])
        #expect(decoded.tokens.allSatisfy { $0.topLogits.count == 5 })
        #expect(decoded.tokens.allSatisfy { $0.topLogits.first?.tokenId == $0.tokenId })

        let json = try #require(
            try JSONSerialization.jsonObject(with: try tempFile.readData()) as? [String: Any])
        #expect(Set(json.keys) == ["tokens"])
        let positions = try #require(json["tokens"] as? [[String: Any]])
        #expect(positions.count == 3)
        #expect(positions.allSatisfy { Set($0.keys) == ["token_id", "incremental_text", "top_logits"] })
    }

    // MARK: Scoring

    @Test("Scoring feeds the whole multi-token context and scores each continuation id")
    func scoringFeedsWholeContext() async throws {
        let engine = RecordingEngine(tokens: [10, 20, 30], vocabSize: 100)
        let generator = try await makeGenerator(engine: engine)
        let context: [Int32] = [11, 12, 13, 14, 15]
        let continuation: [Int32] = [21, 22, 23]

        let result = try await generator.evaluateRawTokens(context: context, continuation: continuation)

        #expect(engine.fedInputs == [context])
        #expect(engine.fedContinuations == [continuation])
        #expect(result.contextTokens == context)
        #expect(result.continuationTokens == continuation)
        #expect(result.logits.count == 3)

        let tempFile = TempFile()
        defer { tempFile.cleanup() }
        try LogitsWriter.handleEvaluationOutput(
            result: result,
            context: "(raw tokens: 5 ids)",
            continuation: "(raw continuation: 3 ids)",
            tokenizer: MockTokenizer(),
            saveLogitsLength: .count(5),
            saveJsonPath: tempFile.path,
            printToConsole: false
        )
        let json = try #require(
            try JSONSerialization.jsonObject(with: try tempFile.readData()) as? [String: Any])
        #expect(json["tokens"] == nil)
        #expect(json["context_tokens"] as? [Int] == [11, 12, 13, 14, 15])
        #expect(json["continuation_tokens"] as? [Int] == [21, 22, 23])
        let positions = try #require(json["positions"] as? [[String: Any]])
        #expect(positions.map { $0["target_token"] as? Int } == [21, 22, 23])
        #expect(json["total_log_probability"] != nil)
        #expect(json["perplexity"] != nil)
    }

    @Test("Scoring refuses an empty context or an empty continuation")
    func scoringRefusesEmptySides() async throws {
        let generator = try await makeGenerator(engine: MockEngine(vocabSize: 100))
        await #expect(throws: ContinuationEvaluationError.self) {
            try await generator.evaluateRawTokens(context: [], continuation: [1])
        }
        await #expect(throws: ContinuationEvaluationError.self) {
            try await generator.evaluateRawTokens(context: [1], continuation: [])
        }
    }

    @Test("Text --continuation still feeds its context and forces its continuation")
    func textContinuationUnchanged() async throws {
        let engine = RecordingEngine(tokens: [10, 20, 30], vocabSize: 256)
        let generator = try await makeGenerator(engine: engine)
        let encoding = ContinuationEncoding(context: "Hello", continuation: " world", tokenizer: MockTokenizer())

        let result = try await generator.evaluateContinuation(context: "Hello", continuation: " world")

        #expect(engine.fedInputs == [Array(encoding.tokens[..<encoding.continuationStartIndex])])
        #expect(engine.fedContinuations == [Array(encoding.tokens[encoding.continuationStartIndex...])])
        #expect(result.contextTokens == encoding.contextTokens)
        #expect(result.continuationTokens == encoding.continuationTokens)
        #expect(result.logits.count == encoding.continuationTokens.count)
    }
}
