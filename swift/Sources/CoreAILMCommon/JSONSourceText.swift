// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// The text of a value inside a JSON document, exactly as the document writes it.
///
/// `Codable` doesn't keep the order of an object's keys: a decoder's keyed container lists them in no
/// particular order, and a decoded `Dictionary` iterates in an order that changes from one process
/// to the next. Where the order matters, as in a JSON schema whose `properties` order is the order a
/// grammar generates them in, the value is taken from the source text instead.
///
/// A decoder carries the source in its `userInfo` under ``userInfoKey``. A type that needs a value's
/// text reads it with ``text(at:in:)`` and its own `codingPath`. Outside this module, a request is
/// decoded this way through ``ChatCompletionRequest/init(json:)``.
enum JSONSourceText {
    /// Where a `JSONDecoder` keeps the document it is decoding, as `Data`.
    static let userInfoKey = CodingUserInfoKey(rawValue: "CoreAILMCommon.JSONSourceText")!

    /// Deepest nesting followed, the same as `JSONDecoder`'s.
    static let maxDepth = 512

    /// The text of the value at `path` in `json`, or nil when it can't be found as `JSONDecoder` would
    /// find it.
    ///
    /// `json` is a UTF-8 document that `JSONDecoder` has accepted, as it is when read from a decoder's
    /// `userInfo`; this follows the document only as far as the path needs and does not validate it.
    /// A key in `path` matches an object member by its unescaped name; every coding path that reaches
    /// here is made of object keys. Nil when a key on the path appears more than once in its object,
    /// since the value and the text could then come from different members, and when the document
    /// nests deeper than `JSONDecoder` reads.
    static func text(at path: [any CodingKey], in json: Data) -> String? {
        var bytes = [UInt8](json)
        if bytes.starts(with: [0xEF, 0xBB, 0xBF]) { bytes.removeFirst(3) }
        var scanner = Scanner(bytes: bytes)
        scanner.skipWhitespace()
        guard let range = scanner.find(path[...], depth: 0) else { return nil }
        return String(decoding: bytes[range], as: UTF8.self)
    }

    /// A JSON decoder whose decoding has `json` to hand as its source text.
    static func decoder(for json: Data) -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.userInfo[userInfoKey] = json
        return decoder
    }

    /// The bytes JSON counts as whitespace.
    static let whitespace: Set<UInt8> = [0x20, 0x09, 0x0A, 0x0D]

    private struct Scanner {
        let bytes: [UInt8]
        var index = 0

        init(bytes: [UInt8]) { self.bytes = bytes }

        mutating func skipWhitespace() {
            while index < bytes.count, JSONSourceText.whitespace.contains(bytes[index]) { index += 1 }
        }

        private func at(_ ascii: Unicode.Scalar) -> Bool {
            index < bytes.count && bytes[index] == UInt8(ascii: ascii)
        }

        /// The range of the value at `path` below the value starting at `index`.
        mutating func find(_ path: ArraySlice<any CodingKey>, depth: Int) -> Range<Int>? {
            guard let key = path.first else {
                let start = index
                guard skipValue(depth: depth) else { return nil }
                return start..<index
            }
            guard depth < JSONSourceText.maxDepth, index < bytes.count else { return nil }
            if at("{") { return findMember(key.stringValue, rest: path.dropFirst(), depth: depth + 1) }
            return nil
        }

        private mutating func findMember(
            _ name: String, rest: ArraySlice<any CodingKey>, depth: Int
        ) -> Range<Int>? {
            index += 1
            var found: Range<Int>?
            var matches = 0
            while true {
                skipWhitespace()
                if at("}") {
                    index += 1
                    // A repeated key: the decoded value and this text might not be the same member.
                    return matches == 1 ? found : nil
                }
                guard let keyRange = stringRange() else { return nil }
                skipWhitespace()
                guard at(":") else { return nil }
                index += 1
                skipWhitespace()
                // Every member is read, not just up to the first match: a later one with the same
                // name is the one `JSONDecoder` may have decoded instead.
                if unescaped(keyRange) == name {
                    matches += 1
                    var inner = Scanner(bytes: bytes)
                    inner.index = index
                    found = inner.find(rest, depth: depth)
                }
                guard skipValue(depth: depth) else { return nil }
                skipWhitespace()
                if at(",") {
                    index += 1
                } else if !at("}") {
                    return nil
                }
            }
        }

        /// Moves past one value. False when it can't: the text ends, or nests too deep.
        private mutating func skipValue(depth: Int) -> Bool {
            guard index < bytes.count else { return false }
            if at("\"") { return stringRange() != nil }
            if at("{") || at("[") {
                guard depth < JSONSourceText.maxDepth else { return false }
                let isObject = at("{")
                let close = isObject ? UInt8(ascii: "}") : UInt8(ascii: "]")
                index += 1
                while true {
                    skipWhitespace()
                    guard index < bytes.count else { return false }
                    if bytes[index] == close {
                        index += 1
                        return true
                    }
                    if isObject {
                        guard stringRange() != nil else { return false }
                        skipWhitespace()
                        guard at(":") else { return false }
                        index += 1
                        skipWhitespace()
                    }
                    guard skipValue(depth: depth + 1) else { return false }
                    skipWhitespace()
                    if at(",") {
                        index += 1
                    } else if !(index < bytes.count && bytes[index] == close) {
                        return false
                    }
                }
            }
            // A number, true, false or null: everything up to the next delimiter.
            let start = index
            let delimiters: Set<UInt8> = [UInt8(ascii: ","), UInt8(ascii: "}"), UInt8(ascii: "]")]
            while index < bytes.count, !delimiters.contains(bytes[index]),
                !JSONSourceText.whitespace.contains(bytes[index])
            {
                index += 1
            }
            return index > start
        }

        /// The range of the string starting at `index`, quotes included, and moves past it.
        private mutating func stringRange() -> Range<Int>? {
            guard at("\"") else { return nil }
            let start = index
            index += 1
            while index < bytes.count {
                switch bytes[index] {
                case UInt8(ascii: "\\"):
                    index += 2
                case UInt8(ascii: "\""):
                    index += 1
                    return start..<index
                default:
                    index += 1
                }
            }
            return nil
        }

        /// The string a quoted, possibly escaped, key stands for.
        private func unescaped(_ range: Range<Int>) -> String? {
            let raw = bytes[range]
            guard raw.contains(UInt8(ascii: "\\")) else {
                return String(decoding: raw.dropFirst().dropLast(), as: UTF8.self)
            }
            return try? JSONDecoder().decode(String.self, from: Data(raw))
        }
    }
}

extension ChatCompletionRequest {
    /// Decodes a request body, keeping each `json_schema`'s text as the client wrote it.
    package init(json: Data) throws {
        self = try JSONSourceText.decoder(for: json).decode(Self.self, from: json)
    }
}
