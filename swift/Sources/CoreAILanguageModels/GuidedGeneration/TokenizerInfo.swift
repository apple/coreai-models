// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CXGrammar
import Foundation

public final class TokenizerInfo {
    private let handle: OpaquePointer
    public let vocabulary: [String]
    public let vocabularySize: Int
    public let vocabType: VocabularyType
    public let addPrefixSpace: Bool
    /// Ids a grammar allows only once its JSON is complete, or nil (or empty) when xgrammar detects them from
    /// the vocabulary.
    public let stopTokenIds: [Int32]?

    /// Extract vocabulary from any tokenizer with a vocabulary property
    ///
    /// - Parameter stopTokenIds: Ids to treat as stop tokens (e.g. EOS), which a grammar allows
    ///   only once its JSON is complete. Nil or empty lets xgrammar detect them from the vocabulary, which
    ///   can miss a model's real end of turn.
    public init(
        vocabulary: [String],
        vocabType: VocabularyType = .raw,
        stopTokenIds: [Int32]? = nil,
        addPrefixSpace: Bool = false
    ) {
        self.vocabulary = vocabulary
        self.vocabType = vocabType
        self.addPrefixSpace = addPrefixSpace
        self.stopTokenIds = stopTokenIds

        // Convert Swift strings to C string array
        let cStrings = vocabulary.map { strdup($0) }
        defer {
            for ptr in cStrings { free(ptr) }
        }

        let cStringArray = UnsafeMutablePointer<UnsafePointer<Int8>?>.allocate(capacity: vocabulary.count)
        defer {
            cStringArray.deallocate()
        }

        for (i, cStr) in cStrings.enumerated() {
            cStringArray[i] = UnsafePointer(cStr)
        }

        // Call C bridge — NULL indicates a programming error (corrupt vocabulary)
        let created = { (stops: UnsafePointer<Int32>?, count: Int) in
            xgrammar_tokenizer_info_create(
                cStringArray, Int32(vocabulary.count), vocabType.cValue, stops, Int32(count), addPrefixSpace)
        }
        let handle =
            stopTokenIds.map { ids in ids.withUnsafeBufferPointer { created($0.baseAddress, ids.count) } }
            ?? created(nil, 0)
        guard let handle else {
            preconditionFailure(
                "Failed to create xgrammar TokenizerInfo: invalid vocabulary (\(vocabulary.count) tokens)")
        }

        self.handle = handle
        self.vocabularySize = Int(xgrammar_tokenizer_info_get_vocab_size(handle))
    }

    deinit {
        xgrammar_tokenizer_info_free(handle)
    }

    internal var cHandle: OpaquePointer {
        handle
    }
}

// MARK: - Vocabulary Caching

/// Cache for tokenizer info to avoid repeated vocabulary extraction
public actor TokenizerInfoCache {
    private var cache: [String: TokenizerInfo] = [:]

    public init() {}

    /// Get or create tokenizer info for a model
    ///
    /// - Parameters:
    ///   - modelName: Name of the model (e.g., "Qwen/Qwen2.5-1.5B")
    ///   - vocabulary: The vocabulary array
    /// - Returns: Cached or newly created TokenizerInfo
    public func getOrCreate(
        modelName: String,
        vocabulary: [String],
        vocabType: VocabularyType = .byteLevel
    ) -> TokenizerInfo {
        let cacheKey = "\(modelName)_\(vocabType)_\(vocabulary.count)"
        if let cached = cache[cacheKey] {
            return cached
        }

        let info = TokenizerInfo(
            vocabulary: vocabulary,
            vocabType: vocabType
        )
        cache[cacheKey] = info
        return info
    }

    /// Clear the cache
    public func clear() {
        cache.removeAll()
    }
}

public enum VocabularyType: Sendable, Equatable {
    case raw
    case byteFallback
    case byteLevel

    /// Infers the encoding from the token strings. SentencePiece byte-fallback vocabularies
    /// (Gemma, Llama) carry all 256 `<0xHH>` byte tokens; anything else is treated as
    /// byte-level BPE. A mismatched type makes xgrammar map token ids to the wrong strings.
    public static func detect(vocabulary: [String]) -> VocabularyType {
        var byteTokens = Set<String>()
        for token in vocabulary where token.utf8.count == 6 && token.hasPrefix("<0x") && token.hasSuffix(">") {
            byteTokens.insert(token.uppercased())
            if byteTokens.count == 256 { return .byteFallback }
        }
        return .byteLevel
    }

    var cValue: XGrammarVocabType {
        switch self {
        case .raw: return XGRAMMAR_VOCAB_RAW
        case .byteFallback: return XGRAMMAR_VOCAB_BYTE_FALLBACK
        case .byteLevel: return XGRAMMAR_VOCAB_BYTE_LEVEL
        }
    }
}
