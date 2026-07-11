// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

/// Generates JSON output constrained to a schema using xgrammar.
///
/// Accepts the `Tokenizer` protocol (swift-transformers) for vocabulary extraction.
///
/// ## Usage
///
/// ```swift
/// import CoreAILanguageModels
///
/// let generator = ConstrainedGenerator(
///     engine: engine,
///     tokenizer: tokenizer,
///     vocabSize: 151936,
///     jsonSchema: schemaString
/// )
///
/// // Generate typed output (schema auto-extracted from Codable type)
/// let person: Person = try await generator.generate(
///     prompt: "Generate a person record as JSON",
///     type: Person.self
/// )
///
/// // Generate raw JSON
/// let json = try await generator.generateJSON(
///     prompt: "Generate a person record as JSON",
///     jsonSchema: schemaString
/// )
/// ```
public struct ConstrainedGenerator: DecodingStrategy {
    private let engine: any InferenceEngine
    private let tokenizer: any Tokenizer
    private let vocabSize: Int
    private let samplingConfig: SamplingConfiguration

    /// The JSON schema used by `decode(from:...)`. Callers that use `generateJSON` directly
    /// pass their schema explicitly; this stored property is only used via the protocol path.
    private let jsonSchema: String

    /// Sentinel token ID used when emitting aggregated output that doesn't correspond to a single token.
    private static let invalidTokenId: Int32 = -1

    /// Initialize with an inference engine and tokenizer.
    ///
    /// - Parameters:
    ///   - engine: The inference engine for generation
    ///   - tokenizer: swift-transformers Tokenizer (vocabulary extracted automatically)
    ///   - vocabSize: Vocabulary size (required — must be supplied from model metadata)
    ///   - jsonSchema: JSON schema string used when generating via `DecodingStrategy.decode`
    ///   - samplingConfig: Sampling configuration (default: greedy)
    public init(
        engine: any InferenceEngine,
        tokenizer: any Tokenizer,
        vocabSize: Int,
        jsonSchema: String,
        samplingConfig: SamplingConfiguration = .greedy
    ) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.jsonSchema = jsonSchema
        self.samplingConfig = samplingConfig
    }

    // MARK: - DecodingStrategy conformance

    /// Stream the complete JSON output as a single `GenerationResult`.
    ///
    /// Unlike `ConstrainedDecodingStrategy` which streams per-token, this emits the
    /// entire JSON string as one result after generation completes. The `tokenId` is
    /// set to `invalidTokenId` since this represents aggregated output, not a single token.
    public func decode(
        from input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        options: InferenceOptions,
        stopSequences: StopSequences
    ) -> ConstrainedGeneratedSequence {
        ConstrainedGeneratedSequence(
            jsonSchema: jsonSchema,
            vocabSize: vocabSize,
            invalidTokenId: Self.invalidTokenId,
            input: input,
            tokenizer: tokenizer,
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            maxTokens: options.maxTokens ?? 200
        )
    }

    // MARK: - Public convenience API

    /// Generate a raw JSON string constrained to a JSON schema.
    public func generateJSON(
        prompt: String,
        jsonSchema: String,
        maxTokens: Int = 200
    ) async throws -> String {
        return try await Self.generateJSONInternal(
            input: .prompt(prompt),
            jsonSchema: jsonSchema,
            tokenizer: self.tokenizer,
            inferenceEngine: self.engine,
            samplingConfiguration: self.samplingConfig,
            vocabSize: self.vocabSize,
            maxTokens: maxTokens
        )
    }

    /// Generate from a schema file path.
    public func generateJSON(
        prompt: String,
        schemaPath: String,
        maxTokens: Int = 200
    ) async throws -> String {
        let schema = try String(contentsOfFile: schemaPath, encoding: .utf8)
        return try await generateJSON(prompt: prompt, jsonSchema: schema, maxTokens: maxTokens)
    }

    // MARK: - Core generation logic

    static func generateJSONInternal(
        input: Input,
        jsonSchema: String,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        vocabSize: Int,
        maxTokens: Int
    ) async throws -> String {
        var session = try ConstrainedGenerationSession(
            jsonSchema: jsonSchema,
            tokenizer: tokenizer,
            vocabSize: vocabSize
        )

        let stopSequences = StopSequences(for: tokenizer)
        var inputTokens = try tokenizeInput(input, tokenizer: tokenizer)

        try await inferenceEngine.reset()

        let generatedTokens = try await runGenerationLoop(
            session: &session,
            inputTokens: &inputTokens,
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            stopSequences: stopSequences,
            maxTokens: maxTokens
        )

        return decodeAndClean(tokens: generatedTokens, tokenizer: tokenizer)
    }

    /// Tokenize input with chat template, falling back to raw encoding.
    private static func tokenizeInput(_ input: Input, tokenizer: any Tokenizer) throws -> [Int32] {
        do {
            return try PromptUtils.maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
                .map(Int32.init)
        } catch {
            switch input {
            case .rawText(let text), .prompt(let text):
                return tokenizer.encode(text: text).map(Int32.init)
            case .tokens(let ids):
                return ids.map(Int32.init)
            }
        }
    }

    /// Run the constrained generation loop: logits → mask → sample → accept.
    private static func runGenerationLoop(
        session: inout ConstrainedGenerationSession,
        inputTokens: inout [Int32],
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        stopSequences: StopSequences,
        maxTokens: Int
    ) async throws -> [Int32] {
        var generatedTokens: [Int32] = []

        for _ in 0..<maxTokens {
            if session.isTerminated { break }

            let options = InferenceOptions(maxTokens: 1, includeLogits: true)

            var rawLogits: [LogitsScalarType]? = nil
            for try await output in try await inferenceEngine.generate(
                with: inputTokens,
                samplingConfiguration: samplingConfiguration,
                inferenceOptions: options
            ) {
                rawLogits = output.logits
                break
            }
            guard let logits = rawLogits else {
                throw ConstrainedGenerationError.generationFailed("No logits returned from engine")
            }

            var maskedLogits = logits
            _ = session.applyMask(to: &maskedLogits)

            let bestToken = CompositeSampler.sample(from: &maskedLogits, config: samplingConfiguration)

            if !session.acceptToken(bestToken) { break }

            inputTokens.append(bestToken)
            generatedTokens.append(bestToken)

            if stopSequences.matches(recentTokens: generatedTokens) { break }
        }

        return generatedTokens
    }

    /// Decode tokens to text and strip EOS token string.
    private static func decodeAndClean(tokens: [Int32], tokenizer: any Tokenizer) -> String {
        let generatedText = tokenizer.decode(tokens: tokens.map(Int.init))
        let eosTokenString = tokenizer.eosTokenId.map { tokenizer.decode(tokens: [$0]) } ?? ""
        let trimmed =
            eosTokenString.isEmpty
            ? generatedText
            : generatedText.replacingOccurrences(of: eosTokenString, with: "")
        return trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - ConstrainedGeneratedSequence

extension ConstrainedGenerator {
    /// Async sequence that emits exactly one `GenerationResult` containing the complete JSON output.
    public struct ConstrainedGeneratedSequence: AsyncSequence {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        let jsonSchema: String
        let vocabSize: Int
        let invalidTokenId: Int32
        let input: Input
        let tokenizer: any Tokenizer
        let inferenceEngine: any InferenceEngine
        let samplingConfiguration: SamplingConfiguration
        let maxTokens: Int

        public func makeAsyncIterator() -> Iterator {
            Iterator(
                jsonSchema: jsonSchema,
                vocabSize: vocabSize,
                invalidTokenId: invalidTokenId,
                input: input,
                tokenizer: tokenizer,
                inferenceEngine: inferenceEngine,
                samplingConfiguration: samplingConfiguration,
                maxTokens: maxTokens
            )
        }
    }
}

extension ConstrainedGenerator.ConstrainedGeneratedSequence {
    public struct Iterator: AsyncIteratorProtocol {
        public typealias Element = GenerationResult
        public typealias Failure = Error

        let jsonSchema: String
        let vocabSize: Int
        let invalidTokenId: Int32
        let input: Input
        let tokenizer: any Tokenizer
        let inferenceEngine: any InferenceEngine
        let samplingConfiguration: SamplingConfiguration
        let maxTokens: Int

        var emitted: Bool = false

        public mutating func next() async throws -> GenerationResult? {
            guard !emitted else {
                return nil
            }
            emitted = true

            try Task.checkCancellation()

            let json = try await ConstrainedGenerator.generateJSONInternal(
                input: input,
                jsonSchema: jsonSchema,
                tokenizer: tokenizer,
                inferenceEngine: inferenceEngine,
                samplingConfiguration: samplingConfiguration,
                vocabSize: vocabSize,
                maxTokens: maxTokens
            )
            return GenerationResult(text: json, tokenId: invalidTokenId, rawLogits: nil)
        }
    }
}
