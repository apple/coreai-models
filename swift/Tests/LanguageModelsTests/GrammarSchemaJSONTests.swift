// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import FoundationModels
import Testing

@testable import CoreAILanguageModels

@Suite("GrammarSchemaJSON")
struct GrammarSchemaJSONTests {
    private func ordered(_ json: String) throws -> String {
        try GrammarSchemaJSON.text(of: JSONSerialization.jsonObject(with: Data(json.utf8)), propertyOrder: nil)
    }

    @Test("Properties follow x-order, and every other key is sorted")
    func propertiesFollowXOrder() throws {
        let json =
            #"{"type":"object","x-order":["zeta","name","age"],"properties":{"age":{"type":"integer"},"#
            + #""zeta":{"type":"string"},"name":{"type":"string"}},"required":["zeta","name","age"]}"#
        #expect(
            try ordered(json)
                == #"{"properties":{"zeta":{"type":"string"},"name":{"type":"string"},"age":{"type":"integer"}},"#
                + #""required":["zeta","name","age"],"type":"object","x-order":["zeta","name","age"]}"#)
    }

    @Test("A property x-order leaves out goes after the listed ones, sorted, and arrays keep their order")
    func unlistedPropertiesAndArrays() throws {
        let json = #"{"x-order":["b"],"properties":{"c":{},"b":{},"a":{"enum":["z","y"]}}}"#
        #expect(try ordered(json) == #"{"properties":{"b":{},"a":{"enum":["z","y"]},"c":{}},"x-order":["b"]}"#)
    }

    @Test("Without x-order, properties are sorted")
    func noXOrderSorts() throws {
        #expect(
            try ordered(#"{"properties":{"b":1,"a":true,"c":null}}"#) == #"{"properties":{"a":true,"b":1,"c":null}}"#)
    }

    @Test("Whole numbers written with a fraction or exponent become integers; strings and other numbers stay")
    func wholeNumbers() {
        #expect(
            GrammarSchemaJSON.normalized(#"{"maxLength": 5.0, "m": 1e0, "x": 0.5, "n": -2.0E1, "s": "5.0"}"#)
                == #"{"maxLength": 5, "m": 1, "x": 0.5, "n": -20, "s": "5.0"}"#)
        #expect(GrammarSchemaJSON.normalized("[12345678901234567890, 7]") == "[12345678901234567890, 7]")
        // Past what a Double holds exactly, or infinite, a number stays as written.
        #expect(
            GrammarSchemaJSON.normalized("[1e300, 1e400, 9.3e18, 2.0e15]")
                == "[1e300, 1e400, 9.3e18, 2000000000000000]")
    }

    @Test("Trailing commas are dropped, and commas inside a string stay")
    func trailingCommas() throws {
        let schema = GrammarSchemaJSON.normalized(
            #"{"type":"object","properties":{"a":{"type":"string"},"b":{"enum":[1, 2, ]},},}"#)
        #expect(schema == #"{"type":"object","properties":{"a":{"type":"string"},"b":{"enum":[1, 2 ]}}}"#)
        #expect((try? JSONSerialization.jsonObject(with: Data(schema.utf8))) != nil)
        #expect(GrammarSchemaJSON.normalized(#"{"d": "x\", }"}"#) == #"{"d": "x\", }"}"#)
    }

    @Test("A GenerationSchema comes out in its declared property order, nested types included, every time")
    func generationSchemaInDeclaredOrder() throws {
        let address = DynamicGenerationSchema(
            name: "Address",
            properties: [
                .init(name: "street", schema: .init(type: String.self)),
                .init(name: "city", schema: .init(type: String.self)),
            ])
        let person = DynamicGenerationSchema(
            name: "Person",
            properties: [
                .init(name: "zeta", schema: .init(type: String.self)),
                .init(name: "name", schema: .init(type: String.self), isOptional: true),
                .init(name: "home", schema: .init(referenceTo: "Address")),
                .init(name: "age", schema: .init(type: Int.self)),
            ])
        let schema = try GenerationSchema(root: person, dependencies: [address])
        let text = try GrammarSchemaJSON.ordered(schema)

        // Property names appear in the text in declared order, at the top level and in $defs.
        let positions = ["\"zeta\":{", "\"name\":{", "\"home\":{", "\"age\":{", "\"street\":{", "\"city\":{"]
            .map { text.range(of: $0)?.lowerBound }
        let found = positions.compactMap { $0 }
        try #require(found.count == positions.count, "every property name is in the text")
        #expect(found[found.count - 2] < found[found.count - 1], "Address: street before city")
        #expect(Array(found.dropLast(2)) == found.dropLast(2).sorted(), "Person: zeta, name, home, age")
        // The same text every time, as the grammar-session cache needs; a raw encoding varies even in-process.
        #expect(Set(try (0..<200).map { _ in try GrammarSchemaJSON.ordered(schema) }).count == 1)
    }
}
