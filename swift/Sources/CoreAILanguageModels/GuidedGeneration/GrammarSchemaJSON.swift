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
