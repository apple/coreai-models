// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Limits how many whitespace characters a model can emit between two JSON tokens.
///
/// A grammar that allows any amount of whitespace lets a model that keeps choosing it, typically a
/// newline and indentation after the last value, run on until the token limit without closing the
/// document. The guard removes whitespace-leading tokens from the next-token mask once the run
/// reaches the limit, so the only tokens left continue the document.
///
/// xgrammar can bound whitespace itself (`max_whitespace_cnt`), but it does so by expanding the
/// compiled grammar: with a 262,144-token vocabulary the grammar grows from about 0.3 MiB to about
/// 1.2 GiB for a nested schema at a limit of 32. Applying the same limit to the mask costs one
/// precomputed bitmask per remaining-run length.
///
/// Whitespace inside a string value is not counted. Output whose whitespace stays within the limit
/// is unchanged: the guard only removes tokens that would exceed it.
struct WhitespaceGuard {
    let limit: Int
    private let profile: WhitespaceProfile
    private var state = State()
    private var history: [State] = []

    /// The tracking state after some prefix of the output.
    private struct State {
        var inString = false
        var escaped = false
        /// Whitespace characters since the last non-whitespace character, outside strings.
        var run = 0
    }

    /// Most states kept for ``rollback(_:)``; xgrammar rolls back at most 64 tokens.
    private static let historyLimit = 256

    init(limit: Int, vocabulary: [String], vocabType: VocabularyType) {
        self.limit = limit
        self.profile = WhitespaceProfile(vocabulary: vocabulary, vocabType: vocabType)
    }

    /// Advances the tracking state past `token`.
    mutating func accept(_ tokenId: Int32) {
        history.append(state)
        if history.count > Self.historyLimit { history.removeFirst(history.count - Self.historyLimit) }
        for scalar in profile.scalars(of: Int(tokenId)) { advance(scalar) }
    }

    /// Returns to the state before the last `count` accepted tokens.
    mutating func rollback(_ count: Int) {
        guard count > 0 else { return }
        if count <= history.count {
            state = history[history.count - count]
            history.removeLast(count)
        } else {
            reset()
        }
    }

    mutating func reset() {
        state = State()
        history.removeAll(keepingCapacity: true)
    }

    /// Removes from `mask` the tokens that would push the whitespace run past the limit. Leaves it
    /// unchanged when that would leave no token at all. The unused bits after the last token of the
    /// vocabulary are not tokens, so they do not count as one left over.
    func restrict(_ mask: UnsafeMutablePointer<Int32>) {
        guard !state.inString else { return }
        let blocked = profile.exceeding(remaining: max(0, limit - state.run))
        let last = blocked.count - 1
        var keepsAToken = false
        for word in 0...max(last, 0) where word <= last {
            let survivors = mask[word] & ~blocked[word] & (word == last ? profile.lastWordTokens : -1)
            if survivors != 0 {
                keepsAToken = true
                break
            }
        }
        guard keepsAToken else { return }
        for word in 0..<blocked.count { mask[word] &= ~blocked[word] }
    }

    private mutating func advance(_ scalar: Unicode.Scalar) {
        if state.inString {
            if state.escaped {
                state.escaped = false
            } else if scalar == "\\" {
                state.escaped = true
            } else if scalar == "\"" {
                state.inString = false
                state.run = 0
            }
        } else if scalar == "\"" {
            state.inString = true
            state.run = 0
        } else if WhitespaceProfile.isWhitespace(scalar) {
            state.run += 1
        } else {
            state.run = 0
        }
    }
}

/// What each token of a vocabulary writes, as far as JSON whitespace is concerned.
final class WhitespaceProfile {
    private let vocabulary: [String]
    private let vocabType: VocabularyType
    private let leading: [UInt16]
    private let words: Int
    private var masks: [Int: [Int32]] = [:]

    /// The bits of the last bitmask word that stand for tokens.
    let lastWordTokens: Int32

    init(vocabulary: [String], vocabType: VocabularyType) {
        self.vocabulary = vocabulary
        self.vocabType = vocabType
        self.words = (vocabulary.count + 31) / 32
        let used = vocabulary.count % 32
        self.lastWordTokens = used == 0 ? -1 : Int32(bitPattern: (UInt32(1) << UInt32(used)) &- 1)
        var leading = [UInt16](repeating: 0, count: vocabulary.count)
        for (id, token) in vocabulary.enumerated() {
            var count = 0
            if vocabType == .byteFallback && token.hasPrefix("<0x") {
                // A byte token such as <0x0A>: rare, so take the general path.
                for scalar in Self.scalars(of: token, vocabType) {
                    guard Self.isWhitespace(scalar) else { break }
                    count += 1
                }
            } else {
                for scalar in token.unicodeScalars {
                    guard Self.isWhitespace(Self.mapped(scalar, vocabType)) else { break }
                    count += 1
                }
            }
            leading[id] = UInt16(min(count, Int(UInt16.max)))
        }
        self.leading = leading
    }

    static func isWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        scalar == " " || scalar == "\n" || scalar == "\t" || scalar == "\r"
    }

    /// The characters token `id` writes.
    func scalars(of id: Int) -> [Unicode.Scalar] {
        guard vocabulary.indices.contains(id) else { return [] }
        return Self.scalars(of: vocabulary[id], vocabType)
    }

    /// The characters a vocabulary entry writes: the encoding's markers for space and newline are
    /// replaced by the real characters, and a `<0xHH>` byte token by its ASCII character.
    static func scalars(of token: String, _ vocabType: VocabularyType) -> [Unicode.Scalar] {
        if vocabType == .byteFallback, token.utf8.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">"),
            let byte = UInt8(token.dropFirst(3).dropLast(), radix: 16)
        {
            return [byte < 0x80 ? Unicode.Scalar(byte) : "\u{FFFD}"]
        }
        return token.unicodeScalars.map { mapped($0, vocabType) }
    }

    /// A character of a vocabulary entry with the encoding's marker for space, newline, tab or
    /// carriage return replaced by the real character.
    private static func mapped(_ scalar: Unicode.Scalar, _ vocabType: VocabularyType) -> Unicode.Scalar {
        switch vocabType {
        case .raw:
            return scalar
        case .byteLevel:
            switch scalar.value {
            case 0x0120: return " "  // Ġ
            case 0x010A: return "\n"  // Ċ
            case 0x0109: return "\t"  // ĉ
            case 0x010D: return "\r"  // č
            default: return scalar
            }
        case .byteFallback:
            return scalar.value == 0x2581 ? " " : scalar  // ▁
        }
    }

    /// A bitmask of the tokens that start with more than `remaining` whitespace characters.
    func exceeding(remaining: Int) -> [Int32] {
        if let mask = masks[remaining] { return mask }
        var mask = [Int32](repeating: 0, count: words)
        for id in 0..<leading.count where Int(leading[id]) > remaining {
            mask[id >> 5] |= Int32(bitPattern: 1 << UInt32(id & 31))
        }
        masks[remaining] = mask
        return mask
    }
}
