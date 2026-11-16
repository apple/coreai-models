// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILMCommon

@Suite("JSONSourceText")
struct JSONSourceTextTests {
    struct Key: CodingKey {
        var stringValue: String
        var intValue: Int?
        init(_ name: String) { stringValue = name }
        init(index: Int) {
            stringValue = "\(index)"
            intValue = index
        }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { self.init(index: intValue) }
    }

    private func text(_ json: String, _ path: [any CodingKey]) -> String? {
        JSONSourceText.text(at: path, in: Data(json.utf8))
    }

    @Test("A nested value comes back exactly as written, key order and spacing included")
    func nestedValueAsWritten() {
        let json = #"{"a": 1, "b": {"z": [1, 2], "y" : {"x":true}}, "c": "d"}"#
        #expect(text(json, [Key("b")]) == #"{"z": [1, 2], "y" : {"x":true}}"#)
        #expect(text(json, [Key("b"), Key("y")]) == #"{"x":true}"#)
        #expect(text(json, [Key("c")]) == #""d""#)
        #expect(text(json, []) == json)
    }

    @Test("Strings holding braces, quotes and escapes don't end a value early")
    func stringsAreOpaque() {
        let json = #"{"s": "a } ] \" , {", "t": {"u": "\\"}, "v": 3}"#
        #expect(text(json, [Key("t")]) == #"{"u": "\\"}"#)
        #expect(text(json, [Key("v")]) == "3")
    }

    @Test("An escaped key matches by its unescaped name")
    func escapedKey() {
        #expect(text(#"{"na\u006de": [true, null]}"#, [Key("name")]) == "[true, null]")
    }

    @Test("An integer key indexes an array")
    func arrayIndex() {
        let json = #"{"items": [{"k": 1}, {"k": 2}, {"k": 3}]}"#
        #expect(text(json, [Key("items"), Key(index: 1)]) == #"{"k": 2}"#)
        #expect(text(json, [Key("items"), Key(index: 3)]) == nil)
    }

    @Test("A repeated key resolves to its last member, as JSONDecoder does")
    func lastMemberWins() {
        #expect(text(#"{"a": 1, "a": {"b": 2}}"#, [Key("a")]) == #"{"b": 2}"#)
    }

    @Test("A missing key or malformed text gives nil")
    func missingOrMalformed() {
        #expect(text(#"{"a": 1}"#, [Key("b")]) == nil)
        #expect(text(#"{"a": 1}"#, [Key("a"), Key("b")]) == nil)
        #expect(text(#"{"a": {"b": 1"#, [Key("a")]) == nil)
        #expect(text(#"{"a" 1}"#, [Key("a")]) == nil)
    }

    // MARK: - A response format's schema

    static let schema =
        #"{"type": "object", "properties": {"zeta": {"type": "string"}, "name": {"type": "string"}, "#
        + #""age": {"type": "integer"}}, "required": ["zeta", "name", "age"]}"#
    static let body =
        #"{"messages":[{"role":"user","content":"Hi"}],"#
        + #""response_format":{"type":"json_schema","json_schema":{"name":"p","schema":"# + schema + "}}}"

    @Test("A schema decoded with its source keeps the client's text, so its properties keep their order")
    func schemaKeepsItsOrder() throws {
        let data = Data(Self.body.utf8)
        let request = try JSONSourceText.decoder(for: data).decode(ChatCompletionRequest.self, from: data)
        #expect(request.responseFormat?.extractedSchema == Self.schema)
    }

    @Test("Without its source, a schema is re-encoded with sorted keys, the same every time")
    func schemaWithoutSourceIsSorted() throws {
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(Self.body.utf8))
        let schema = try #require(request.responseFormat?.extractedSchema)
        #expect(schema.contains(#""properties":{"age":{"type":"integer"},"name":{"type":"string"},"zeta":"#))
        #expect(request.responseFormat?.jsonSchema?.schemaText == nil)
    }

    @Test("A replay line keeps its schema's order too")
    func replayKeepsTheOrder() throws {
        let line = #"{"id":"j","request":"# + Self.body + "}"
        let requests = try ReplayIO.parseRequests(line)
        guard case .chat(let request) = try #require(requests.first).body else {
            Issue.record("expected a chat request")
            return
        }
        #expect(request.responseFormat?.extractedSchema == Self.schema)
    }
}
