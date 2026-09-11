// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// CLIP BPE tokenizer that reads vocab + merges from an HF-format `tokenizer.json`
// shipped alongside the model bundle.
//
// swift-transformers' AutoTokenizer doesn't support CLIP's BPE variant
// (specifically the `end_of_word_suffix: "</w>"` flag), so we ship this
// CLIP-specific encoder. The on-disk layout still matches HF conventions —
// only the loader is custom.
public struct CLIPTokenizer: Sendable {
    public let encoder: [String: Int32]
    private let decoder: [Int32: String]
    private struct MergePair: Hashable {
        let left: String
        let right: String
    }
    private let bpeRanks: [MergePair: Int]

    public static let sotTokenId: Int32 = 49406
    public static let eotTokenId: Int32 = 49407

    // Byte-level encoding: maps each byte value to a unicode character string.
    // Matches Python's bytes_to_unicode() exactly.
    static let byteEncoder: [Int: String] = {
        var bs =
            Array(Int(("!").utf8.first!)...Int(("~").utf8.first!))
            + Array(0xA1...0xAC)
            + Array(0xAE...0xFF)
        var cs = bs
        var n = 0
        for b in 0..<256 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }
        var result: [Int: String] = [:]
        for (b, c) in zip(bs, cs) {
            if let scalar = Unicode.Scalar(c) {
                result[b] = String(scalar)
            }
        }
        return result
    }()

    private struct TokenizerJSON: Decodable {
        let model: Model
        struct Model: Decodable {
            let vocab: [String: Int32]
            let merges: [[String]]
        }
    }

    /// Load a tokenizer from an HF-format `tokenizer/` directory containing `tokenizer.json`.
    public init(folder: URL) throws {
        let url = folder.appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: url)
        let parsed = try JSONDecoder().decode(TokenizerJSON.self, from: data)

        var merges: [(String, String)] = []
        merges.reserveCapacity(parsed.model.merges.count)
        for pair in parsed.model.merges {
            guard pair.count == 2 else { continue }
            merges.append((pair[0], pair[1]))
        }
        try self.init(vocab: parsed.model.vocab, merges: merges)
    }

    /// Build a tokenizer from a parsed vocab map and ordered merge pairs.
    public init(vocab: [String: Int32], merges: [(String, String)]) throws {
        self.encoder = vocab
        self.decoder = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })

        var ranks: [MergePair: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (i, merge) in merges.enumerated() {
            ranks[MergePair(left: merge.0, right: merge.1)] = i
        }
        self.bpeRanks = ranks
    }

    // MARK: - Encoding

    /// Encode `text` to token IDs, padded/truncated to `contextLength`.
    ///
    /// Pads with `eotTokenId` to match SAM3's `torch.zeros`-then-fill behavior.
    public func encode(_ text: String, contextLength: Int = 77) -> [Int32] {
        let cleaned = whitespaceClean(text).lowercased()
        let wordTokens = tokenize(cleaned)

        var ids: [Int32] = [Self.sotTokenId]
        ids += wordTokens.compactMap { encoder[$0] }
        ids.append(Self.eotTokenId)

        if ids.count > contextLength {
            ids = Array(ids.prefix(contextLength))
            ids[contextLength - 1] = Self.eotTokenId
        }

        while ids.count < contextLength {
            ids.append(Self.eotTokenId)
        }

        return ids
    }

    // MARK: - Private

    private func tokenize(_ text: String) -> [String] {
        var result: [String] = []
        for token in splitTokens(text) {
            let byteEncoded = token.utf8.compactMap { Self.byteEncoder[Int($0)] }.joined()
            let bpeResult = bpe(byteEncoded)
            result += bpeResult.components(separatedBy: " ")
        }
        return result
    }

    // Splits text similarly to SAM3's regex:
    // |'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
    private func splitTokens(_ text: String) -> [String] {
        let contractionSuffixes = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
        var tokens: [String] = []
        var current = text.startIndex

        while current < text.endIndex {
            var matched = false
            for suffix in contractionSuffixes {
                if text[current...].hasPrefix(suffix) {
                    tokens.append(suffix)
                    current = text.index(current, offsetBy: suffix.count)
                    matched = true
                    break
                }
            }
            if matched { continue }

            let ch = text[current]
            if ch.isLetter {
                var end = text.index(after: current)
                while end < text.endIndex && text[end].isLetter {
                    end = text.index(after: end)
                }
                tokens.append(String(text[current..<end]))
                current = end
            } else if ch.isNumber {
                tokens.append(String(ch))
                current = text.index(after: current)
            } else if ch.isWhitespace {
                current = text.index(after: current)
            } else {
                var end = text.index(after: current)
                while end < text.endIndex && !text[end].isWhitespace && !text[end].isLetter
                    && !text[end].isNumber
                {
                    end = text.index(after: end)
                }
                tokens.append(String(text[current..<end]))
                current = end
            }
        }
        return tokens
    }

    private func bpe(_ token: String) -> String {
        var chars = token.map { String($0) }
        guard !chars.isEmpty else { return token }
        chars[chars.count - 1] += "</w>"

        if chars.count == 1 {
            return chars[0]
        }

        var word = chars
        while word.count > 1 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(word.count - 1) {
                if let rank = bpeRanks[MergePair(left: word[i], right: word[i + 1])],
                    rank < bestRank
                {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if bestIdx == -1 { break }

            let first = word[bestIdx]
            let second = word[bestIdx + 1]

            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }

        return word.joined(separator: " ")
    }

    private func whitespaceClean(_ text: String) -> String {
        text.components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}
