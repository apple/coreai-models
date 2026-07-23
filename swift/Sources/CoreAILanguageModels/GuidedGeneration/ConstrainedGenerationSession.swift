// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

// MARK: - Constrained Generation Session

/// High-level API for constrained (guided) text generation using JSON schemas.
///
/// `ConstrainedGenerationSession` wraps the low-level xgrammar primitives into a
/// simple API that can be integrated into app code. It manages the grammar matcher
/// lifecycle, bitmask allocation, and logit masking.
///
/// Each session is tied to a specific JSON schema and vocabulary. It tracks
/// the generation state and produces token masks that enforce schema compliance.
public struct ConstrainedGenerationSession: ~Copyable {
    private let tokenizerInfo: TokenizerInfo
    private let compiler: GrammarCompiler
    private let compiledGrammar: CompiledGrammar
    private let matcher: GrammarMatcher
    public let vocabularySize: Int
    private let bitmaskSize: Int
    private var bitmaskBuffer: [Int32]
    // Tracks termination via all-zeros bitmask (xgrammar signals completion this way
    // when no EOS token is present, returning false from fillNextTokenBitmask).
    private var allTokensBlocked = false

    /// The JSON schema this session enforces.
    public let schema: String

    /// Whether the grammar has reached a terminal state (valid JSON complete).
    ///
    /// Returns true either when xgrammar explicitly marks the matcher as terminated,
    /// or when `fillNextTokenBitmask` returns false (all tokens blocked = JSON complete).
    public var isTerminated: Bool {
        matcher.isTerminated || allTokensBlocked
    }

    /// Memory used by the compiled grammar in bytes.
    public var compiledGrammarMemoryBytes: Int {
        compiledGrammar.memorySizeBytes
    }

    /// Initialize a constrained generation session with a JSON schema.
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string
    ///   - vocabulary: The full tokenizer vocabulary (one entry per token ID)
    ///   - vocabType: The vocabulary encoding type (default: `.byteLevel`)
    ///   - stopTokenIds: Token IDs to treat as stop tokens (e.g., EOS).
    ///     xgrammar allows these only at grammar-terminal states (valid JSON complete).
    ///     Pass `nil` to rely on xgrammar defaults.
    /// - Throws: `ConstrainedGenerationError` if the schema is invalid JSON
    public init(
        jsonSchema: String,
        vocabulary: [String],
        vocabType: VocabularyType = .byteLevel,
        stopTokenIds: [Int32]? = nil
    ) throws {
        try Self.validateJSONSchema(jsonSchema)
        let tokenizerInfo = TokenizerInfo(
            vocabulary: vocabulary,
            vocabType: vocabType
        )
        try self.init(jsonSchema: jsonSchema, tokenizerInfo: tokenizerInfo, validated: true)
    }

    /// Initialize with a pre-built `TokenizerInfo` (useful when caching across sessions).
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string
    ///   - tokenizerInfo: Pre-built tokenizer info (must include stop token IDs if needed)
    /// - Throws: `ConstrainedGenerationError` if the schema is invalid JSON,
    ///           `XGrammarError` if schema compilation fails
    public init(
        jsonSchema: String,
        tokenizerInfo: TokenizerInfo
    ) throws {
        try Self.validateJSONSchema(jsonSchema)
        try self.init(jsonSchema: jsonSchema, tokenizerInfo: tokenizerInfo, validated: true)
    }

    /// Internal init that skips validation (caller must validate first).
    private init(
        jsonSchema: String,
        tokenizerInfo: TokenizerInfo,
        validated: Bool
    ) throws {
        self.schema = jsonSchema
        self.tokenizerInfo = tokenizerInfo
        self.compiler = GrammarCompiler(tokenizerInfo: tokenizerInfo)
        self.compiledGrammar = try compiler.compileJSONSchema(jsonSchema)
        self.matcher = GrammarMatcher(compiledGrammar: compiledGrammar)
        self.vocabularySize = tokenizerInfo.vocabularySize
        self.bitmaskSize = (vocabularySize + 31) / 32
        self.bitmaskBuffer = Array(repeating: 0, count: bitmaskSize)
    }

    /// Validate that a string is valid JSON.
    private static func validateJSONSchema(_ schema: String) throws {
        guard let data = schema.data(using: .utf8),
            (try? JSONSerialization.jsonObject(with: data)) != nil
        else {
            throw ConstrainedGenerationError.invalidSchema(schema)
        }
    }

    // MARK: - Tokenizer Init

    /// Initialize from a `Tokenizer` (swift-transformers protocol).
    ///
    /// Extracts vocabulary automatically from the tokenizer.
    ///
    /// - Parameters:
    ///   - jsonSchema: A valid JSON schema string
    ///   - tokenizer: A swift-transformers Tokenizer for vocabulary extraction
    ///   - vocabSize: Size of the tokenizer vocabulary
    ///   - vocabType: The vocabulary encoding type (default: `.byteLevel`)
    ///   - stopTokenIds: Token IDs to treat as stop tokens (e.g., EOS).
    ///     xgrammar allows these tokens only at grammar-terminal states (valid JSON complete),
    ///     blocking them mid-generation. Pass `nil` to rely on xgrammar defaults (not recommended).
    /// - Throws: `ConstrainedGenerationError` if the schema is invalid JSON
    public init(
        jsonSchema: String,
        tokenizer: any Tokenizer,
        vocabSize: Int,
        vocabType: VocabularyType = .byteLevel,
        stopTokenIds: [Int32]? = nil
    ) throws {
        // Extract vocabulary from tokenizer
        var vocabulary: [String] = []
        vocabulary.reserveCapacity(vocabSize)
        for i in 0..<vocabSize {
            // xgrammar handles empty strings for missing token IDs; many tokenizers
            // have gaps in their ID space, so nil here is expected, not an error.
            vocabulary.append(tokenizer.convertIdToToken(i) ?? "")
        }
        try self.init(jsonSchema: jsonSchema, vocabulary: vocabulary, vocabType: vocabType, stopTokenIds: stopTokenIds)
    }

    // MARK: - Raw Bitmask Access (for MaskProviding integration)

    /// Get the bitmask of allowed tokens for the next generation step.
    ///
    /// Each bit in the returned array represents whether a token is allowed (1) or not (0).
    /// Bit `i` in word `i/32` at position `i%32`.
    ///
    /// - Returns: The bitmask array, or nil if the grammar is terminated
    public mutating func nextTokenBitmask() -> [Int32]? {
        if isTerminated { return nil }

        let hasConstraints = bitmaskBuffer.withUnsafeMutableBufferPointer { buffer in
            matcher.fillNextTokenBitmask(buffer.baseAddress!)
        }
        // xgrammar may signal completion either by returning false, or by filling
        // an all-zeros bitmask (no tokens allowed). Both indicate the grammar is done.
        if !hasConstraints || bitmaskBuffer.allSatisfy({ $0 == 0 }) {
            allTokensBlocked = true
            return nil
        }
        return bitmaskBuffer
    }

    // MARK: - Logit Masking

    /// Apply the grammar mask to Float logits in-place (disallowed tokens → `-.infinity`).
    ///
    /// - Parameter logits: Raw logits from the model (modified in-place, length = vocab size)
    /// - Returns: `true` if mask was applied, `false` if terminated (logits unchanged)
    @discardableResult
    public mutating func applyMask(to logits: inout [Float]) -> Bool {
        guard let bitmask = nextTokenBitmask() else { return false }
        _applyBitmask(bitmask, to: &logits, vocabularySize: vocabularySize, negativeInfinity: -.infinity)
        return true
    }

    /// Accept a generated token and advance the grammar state.
    ///
    /// Call this after sampling a token from the masked logits.
    ///
    /// - Parameter tokenId: The token ID that was sampled
    /// - Returns: `true` if the token was accepted, `false` if rejected by the grammar
    @discardableResult
    public mutating func acceptToken(_ tokenId: Int32) -> Bool {
        matcher.acceptToken(tokenId)
    }

    /// Reset the session to the initial state for reuse with the same schema.
    public mutating func reset() {
        matcher.reset()
        allTokensBlocked = false
    }

    /// Fill the bitmask directly into a caller-provided buffer (e.g., a GPU-visible MTLBuffer).
    ///
    /// The caller must ensure the pointer has room for at least `(vocabularySize + 31) / 32`
    /// Int32 words. Returns `false` if the grammar is terminated (buffer untouched).
    public mutating func fillBitmask(into pointer: UnsafeMutablePointer<Int32>) -> Bool {
        if isTerminated { return false }

        let hasConstraints = matcher.fillNextTokenBitmask(pointer)
        if !hasConstraints {
            allTokensBlocked = true
            return false
        }
        // Check for all-zeros (no tokens allowed)
        for i in 0..<bitmaskSize {
            if pointer[i] != 0 { return true }
        }
        allTokensBlocked = true
        return false
    }
}

// MARK: - Float16 Masking

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
extension ConstrainedGenerationSession {
    /// Apply the grammar mask to Float16 logits in-place, setting disallowed tokens to
    /// `-Float16.greatestFiniteMagnitude`.
    ///
    /// - Parameter logits: Raw logits from the model (modified in-place, length = vocab size)
    /// - Returns: `true` if mask was applied, `false` if terminated (logits unchanged)
    @discardableResult
    public mutating func applyMask(to logits: inout [Float16]) -> Bool {
        guard let bitmask = nextTokenBitmask() else { return false }
        _applyBitmask(
            bitmask, to: &logits, vocabularySize: vocabularySize, negativeInfinity: -Float16.greatestFiniteMagnitude)
        return true
    }
}
#endif

// MARK: - Errors

public enum ConstrainedGenerationError: Error, LocalizedError {
    case invalidSchema(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSchema(let schema):
            return "Invalid JSON schema: \(schema.prefix(100))..."
        case .generationFailed(let reason):
            return "Constrained generation failed: \(reason)"
        }
    }
}

// MARK: - Convenience: Schema from File

extension ConstrainedGenerationSession {
    /// Create a session by loading a JSON schema from a file.
    ///
    /// - Parameters:
    ///   - schemaPath: Path to a JSON schema file
    ///   - vocabulary: The full tokenizer vocabulary
    ///   - vocabType: The vocabulary encoding type
    ///   - stopTokenIds: Token IDs to treat as stop tokens (e.g., EOS)
    /// - Throws: If the file can't be read or the schema is invalid
    public init(
        schemaPath: String,
        vocabulary: [String],
        vocabType: VocabularyType = .byteLevel,
        stopTokenIds: [Int32]? = nil
    ) throws {
        let schema = try String(contentsOfFile: schemaPath, encoding: .utf8)
        try self.init(jsonSchema: schema, vocabulary: vocabulary, vocabType: vocabType, stopTokenIds: stopTokenIds)
    }
}

/// Applies bitmask to logits.
private func _applyBitmask<T: BinaryFloatingPoint>(
    _ bitmask: [Int32],
    to logits: inout [T],
    vocabularySize: Int,
    negativeInfinity negInf: T
) {
    let count = min(logits.count, vocabularySize)
    for (wordIndex, mask) in bitmask.enumerated() {
        let base = wordIndex &* 32
        if base >= count { break }
        if mask == 0 {
            let end = min(base + 32, count)
            for i in base..<end { logits[i] = negInf }
        } else if mask == Int32(bitPattern: 0xFFFF_FFFF) {
            continue
        } else {
            for bitIndex in 0..<32 {
                let tokenId = base + bitIndex
                if tokenId >= count { break }
                if (mask & (1 << bitIndex)) == 0 { logits[tokenId] = negInf }
            }
        }
    }
}
