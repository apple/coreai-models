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

    /// The text of the value at `path` in `json`, or nil when the path doesn't lead to a value or the
    /// text along it isn't well-formed JSON.
    ///
    /// A key in `path` matches an object member by its unescaped name. An integer key indexes an
    /// array. When an object repeats a key, the last member wins, as with `JSONDecoder`.
    public static func text(at path: [any CodingKey], in json: Data) -> String? {
        let bytes = [UInt8](json)
        var scanner = Scanner(bytes: bytes)
        scanner.skipWhitespace()
        guard let range = scanner.find(path[...]) else { return nil }
        return String(decoding: bytes[range], as: UTF8.self)
    }

    /// A JSON decoder whose decoding has `json` to hand as its source text.
    public static func decoder(for json: Data) -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.userInfo[userInfoKey] = json
        return decoder
    }

    private struct Scanner {
        let bytes: [UInt8]
        var index = 0

        init(bytes: [UInt8]) { self.bytes = bytes }

        mutating func skipWhitespace() {
            while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) { index += 1 }
        }

        /// The range of the value at `path` below the value starting at `index`.
        mutating func find(_ path: ArraySlice<any CodingKey>) -> Range<Int>? {
            guard let key = path.first else {
                let start = index
                guard skipValue() else { return nil }
                return start..<index
            }
            guard index < bytes.count else { return nil }
            switch bytes[index] {
            case UInt8(ascii: "{"):
                return findMember(key.stringValue, rest: path.dropFirst())
            case UInt8(ascii: "["):
                guard let position = key.intValue else { return nil }
                return findElement(position, rest: path.dropFirst())
            default:
                return nil
            }
        }

        private mutating func findMember(_ name: String, rest: ArraySlice<any CodingKey>) -> Range<Int>? {
            index += 1
            var found: Range<Int>?
            skipWhitespace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "}") {
                index += 1
                return nil
            }
            while true {
                skipWhitespace()
                guard let keyRange = stringRange() else { return nil }
                skipWhitespace()
                guard index < bytes.count, bytes[index] == UInt8(ascii: ":") else { return nil }
                index += 1
                skipWhitespace()
                if unescaped(keyRange) == name {
                    var inner = Scanner(bytes: bytes)
                    inner.index = index
                    found = inner.find(rest)
                }
                guard skipValue() else { return nil }
                skipWhitespace()
                guard index < bytes.count else { return nil }
                if bytes[index] == UInt8(ascii: ",") {
                    index += 1
                    continue
                }
                guard bytes[index] == UInt8(ascii: "}") else { return nil }
                index += 1
                return found
            }
        }

        private mutating func findElement(_ position: Int, rest: ArraySlice<any CodingKey>) -> Range<Int>? {
            index += 1
            skipWhitespace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "]") { return nil }
            var current = 0
            while true {
                skipWhitespace()
                if current == position { return find(rest) }
                guard skipValue() else { return nil }
                skipWhitespace()
                guard index < bytes.count, bytes[index] == UInt8(ascii: ",") else { return nil }
                index += 1
                current += 1
            }
        }

        /// Moves past one value. False when the text isn't a well-formed value.
        private mutating func skipValue() -> Bool {
            guard index < bytes.count else { return false }
            switch bytes[index] {
            case UInt8(ascii: "\""):
                return stringRange() != nil
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                let close = bytes[index] == UInt8(ascii: "{") ? UInt8(ascii: "}") : UInt8(ascii: "]")
                index += 1
                skipWhitespace()
                if index < bytes.count, bytes[index] == close {
                    index += 1
                    return true
                }
                while true {
                    skipWhitespace()
                    if close == UInt8(ascii: "}") {
                        guard stringRange() != nil else { return false }
                        skipWhitespace()
                        guard index < bytes.count, bytes[index] == UInt8(ascii: ":") else { return false }
                        index += 1
                        skipWhitespace()
                    }
                    guard skipValue() else { return false }
                    skipWhitespace()
                    guard index < bytes.count else { return false }
                    if bytes[index] == UInt8(ascii: ",") {
                        index += 1
                        continue
                    }
                    guard bytes[index] == close else { return false }
                    index += 1
                    return true
                }
            default:
                // A number, true, false or null: everything up to the next delimiter.
                let start = index
                while index < bytes.count,
                    ![UInt8(ascii: ","), UInt8(ascii: "}"), UInt8(ascii: "]"), 0x20, 0x09, 0x0A, 0x0D].contains(
                        bytes[index])
                {
                    index += 1
                }
                return index > start
            }
        }

        /// The range of the string starting at `index`, quotes included, and moves past it.
        private mutating func stringRange() -> Range<Int>? {
            guard index < bytes.count, bytes[index] == UInt8(ascii: "\"") else { return nil }
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
