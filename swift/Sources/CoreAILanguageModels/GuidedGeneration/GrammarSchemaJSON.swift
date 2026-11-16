// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// A JSON schema written so that a grammar built from it is the same in every process.
///
/// xgrammar generates an object's properties in the order its schema lists them. Encoded with
/// `JSONEncoder`, a schema built from dictionaries lists them in an order that changes from one
/// process to the next, and so would the text a greedy request generates. `GenerationSchema` encodes
/// that way, and records the declared order separately, in `x-order`.
enum GrammarSchemaJSON {
    /// `json` rewritten with every object's `properties` in its `x-order`, then any not listed there
    /// sorted, and every other key sorted. Values keep their meaning, though `JSONSerialization` may
    /// write a number or an escape differently. Nil when `json` isn't a JSON value.
    static func ordered(_ json: Data) -> String? {
        guard let value = try? JSONSerialization.jsonObject(with: json, options: [.fragmentsAllowed]) else {
            return nil
        }
        var out = ""
        guard write(value, propertyOrder: nil, into: &out) else { return nil }
        return out
    }

    /// `json` as the strict JSON a grammar compiler takes. A trailing comma, which Foundation's
    /// parsers accept, is dropped, and a whole number written with a fraction or an exponent (`5.0`,
    /// `1e0`) is written as an integer, so a schema bound such as `"maxLength": 5.0` stays one the
    /// grammar takes as an integer. Every other number, string and key is left as written, large
    /// integers included, and so is the order of every key.
    static func normalized(_ json: String) -> String {
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

    /// The bytes JSON counts as whitespace.
    private static let whitespace: Set<UInt8> = [0x20, 0x09, 0x0A, 0x0D]

    /// Whether the next byte from `start` that isn't whitespace closes an object or an array.
    private static func closes(_ bytes: [UInt8], after start: Int) -> Bool {
        var index = start
        while index < bytes.count, whitespace.contains(bytes[index]) { index += 1 }
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

    /// Appends `value` to `out`. An object's keys go in `propertyOrder` first when it is given (the
    /// `properties` object of a schema with an `x-order`), and sorted otherwise.
    private static func write(_ value: Any, propertyOrder: [String]?, into out: inout String) -> Bool {
        switch value {
        case let object as [String: Any]:
            let order = object["x-order"] as? [String]
            var keys = object.keys.sorted()
            if let propertyOrder {
                var seen = Set<String>()
                let listed = propertyOrder.filter { object[$0] != nil && seen.insert($0).inserted }
                keys = listed + keys.filter { !listed.contains($0) }
            }
            out += "{"
            for (index, key) in keys.enumerated() {
                if index > 0 { out += "," }
                guard fragment(key, into: &out) else { return false }
                out += ":"
                guard write(object[key]!, propertyOrder: key == "properties" ? order : nil, into: &out) else {
                    return false
                }
            }
            out += "}"
            return true
        case let array as [Any]:
            out += "["
            for (index, element) in array.enumerated() {
                if index > 0 { out += "," }
                guard write(element, propertyOrder: nil, into: &out) else { return false }
            }
            out += "]"
            return true
        default:
            return fragment(value, into: &out)
        }
    }

    /// A string, number, boolean or null, written as `JSONSerialization` writes it.
    private static func fragment(_ value: Any, into out: inout String) -> Bool {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]) else {
            return false
        }
        out += String(decoding: data, as: UTF8.self)
        return true
    }
}
