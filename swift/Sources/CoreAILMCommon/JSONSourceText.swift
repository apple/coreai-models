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
/// text reads it with ``text(at:in:)`` and its own `codingPath`.
public enum JSONSourceText {
    /// Where a `JSONDecoder` keeps the document it is decoding, as `Data`.
    public static let userInfoKey = CodingUserInfoKey(rawValue: "CoreAILMCommon.JSONSourceText")!

    /// Deepest nesting followed, the same as `JSONDecoder`'s.
    static let maxDepth = 512

    /// The text of the value at `path` in `json`, or nil when it can't be found as `JSONDecoder` would
    /// find it.
    ///
    /// `json` is a UTF-8 document that `JSONDecoder` has accepted, as it is when read from a decoder's
    /// `userInfo`; this follows the document only as far as the path needs and does not validate it.
    /// A key in `path` matches an object member by its unescaped name, and an integer key indexes an
    /// array. Nil when a key on the path appears more than once in its object, since the value and the
    /// text could then come from different members, and when the document nests deeper than
    /// `JSONDecoder` reads.
    public static func text(at path: [any CodingKey], in json: Data) -> String? {
        var bytes = [UInt8](json)
        if bytes.starts(with: [0xEF, 0xBB, 0xBF]) { bytes.removeFirst(3) }
        var scanner = Scanner(bytes: bytes)
        scanner.skipWhitespace()
        guard let range = scanner.find(path[...], depth: 0) else { return nil }
        return String(decoding: bytes[range], as: UTF8.self)
    }

    /// A JSON decoder whose decoding has `json` to hand as its source text.
    public static func decoder(for json: Data) -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.userInfo[userInfoKey] = json
        return decoder
    }

    /// `json` as strict JSON a grammar compiler takes, and as re-encoding the decoded value would
    /// write it where they differ: a trailing comma, which `JSONDecoder` accepts, is dropped, and a
    /// whole number written with a fraction or an exponent (`5.0`, `1e0`) is written as an integer,
    /// so a schema bound such as `"maxLength": 5.0` stays one a grammar takes as an integer. Every
    /// other number, string and key is left as written, large integers included.
    public static func normalizedForGrammar(_ json: String) -> String {
        let bytes = Array(json.utf8)
        var out: [UInt8] = []
        out.reserveCapacity(bytes.count)
        var index = 0
        while index < bytes.count {
            let byte = bytes[index]
            if byte == UInt8(ascii: "\"") {
                let start = index
                index += 1
                while index < bytes.count, bytes[index] != UInt8(ascii: "\"") {
                    index += bytes[index] == UInt8(ascii: "\\") ? 2 : 1
                }
                index = min(index + 1, bytes.count)
                out += bytes[start..<index]
            } else if byte == UInt8(ascii: "-") || (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte) {
                let start = index
                while index < bytes.count, Self.numberBytes.contains(bytes[index]) { index += 1 }
                let token = bytes[start..<index]
                out += integerForm(of: token) ?? Array(token)
            } else if byte == UInt8(ascii: ","), Self.closes(bytes, after: index + 1) {
                index += 1
            } else {
                out.append(byte)
                index += 1
            }
        }
        return String(decoding: out, as: UTF8.self)
    }

    private static let numberBytes = Set("0123456789+-.eE".utf8)

    /// Whether the next byte from `start` that isn't whitespace closes an object or an array.
    private static func closes(_ bytes: [UInt8], after start: Int) -> Bool {
        var index = start
        while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) { index += 1 }
        return index < bytes.count && (bytes[index] == UInt8(ascii: "}") || bytes[index] == UInt8(ascii: "]"))
    }

    /// The integer `token` is, when it has a fraction or an exponent and is a whole number a `Double`
    /// holds exactly.
    private static func integerForm(of token: ArraySlice<UInt8>) -> [UInt8]? {
        let marks: Set<UInt8> = [UInt8(ascii: "."), UInt8(ascii: "e"), UInt8(ascii: "E")]
        guard token.contains(where: marks.contains),
            let value = Double(String(decoding: token, as: UTF8.self)), value.rounded() == value,
            abs(value) <= 9_007_199_254_740_992
        else { return nil }
        return Array(String(Int64(value)).utf8)
    }

    private struct Scanner {
        let bytes: [UInt8]
        var index = 0

        init(bytes: [UInt8]) { self.bytes = bytes }

        mutating func skipWhitespace() {
            while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) { index += 1 }
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
            if at("["), let position = key.intValue {
                return findElement(position, rest: path.dropFirst(), depth: depth + 1)
            }
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

        private mutating func findElement(
            _ position: Int, rest: ArraySlice<any CodingKey>, depth: Int
        ) -> Range<Int>? {
            index += 1
            var current = 0
            while true {
                skipWhitespace()
                if at("]") { return nil }
                if current == position { return find(rest, depth: depth) }
                guard skipValue(depth: depth) else { return nil }
                skipWhitespace()
                if at(",") {
                    index += 1
                } else if !at("]") {
                    return nil
                }
                current += 1
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
            let delimiters: [UInt8] = [UInt8(ascii: ","), UInt8(ascii: "}"), UInt8(ascii: "]"), 0x20, 0x09, 0x0A, 0x0D]
            while index < bytes.count, !delimiters.contains(bytes[index]) { index += 1 }
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
