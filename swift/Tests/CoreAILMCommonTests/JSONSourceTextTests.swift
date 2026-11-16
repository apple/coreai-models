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

    @Test("A key repeated on the path gives nil, so the text can't disagree with the decoded value")
    func repeatedKeyOnThePath() {
        #expect(text(#"{"a": 1, "a": {"b": 2}}"#, [Key("a")]) == nil)
        #expect(text(#"{"a": {"b": 1}, "a": {"b": 2}}"#, [Key("a"), Key("b")]) == nil)
        // A repeat off the path doesn't matter.
        #expect(text(#"{"x": 1, "x": 2, "a": [3]}"#, [Key("a")]) == "[3]")
    }

    @Test("A trailing comma or a byte order mark, which JSONDecoder accepts, doesn't stop the lookup")
    func lenientInput() {
        #expect(text(#"{"a": [1, 2,], "b": {"c": 3,},}"#, [Key("b")]) == #"{"c": 3,}"#)
        let bom = Data([0xEF, 0xBB, 0xBF] + Array(#"{"a": 7}"#.utf8))
        #expect(JSONSourceText.text(at: [Key("a")], in: bom) == "7")
    }

    @Test("A string ending in a backslash at the end of the text, or nesting past JSONDecoder's limit, gives nil")
    func truncatedAndDeep() {
        #expect(text(#"{"a": "x\"#, [Key("a")]) == nil)
        let deep = String(repeating: "[", count: 600) + String(repeating: "]", count: 600)
        #expect(text(#"{"a": "# + deep + #", "b": 1}"#, [Key("b")]) == nil)
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

    @Test("A request decoded as the server decodes it keeps the schema's text, so its properties keep their order")
    func schemaKeepsItsOrder() throws {
        let request = try ChatCompletionRequest(json: Data(Self.body.utf8))
        #expect(request.responseFormat?.extractedSchema == Self.schema)
    }

    @Test("A schema written with trailing commas comes back as written, for the grammar session to make strict")
    func trailingCommasAreKept() throws {
        let body =
            #"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"schema":"#
            + #"{"type":"object","properties":{"a":{"type":"string"}},}}}}"#
        let data = Data(body.utf8)
        let request = try JSONSourceText.decoder(for: data).decode(ChatCompletionRequest.self, from: data)
        #expect(request.responseFormat?.extractedSchema == #"{"type":"object","properties":{"a":{"type":"string"}},}"#)
    }

    @Test("A schema with a repeated key is the decoded one, re-encoded, rather than another member's text")
    func repeatedSchemaFallsBack() throws {
        let body =
            #"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"#
            + #""schema":{"b":1,"a":2},"schema":{"c":3}}}}"#
        let data = Data(body.utf8)
        let request = try JSONSourceText.decoder(for: data).decode(ChatCompletionRequest.self, from: data)
        #expect(request.responseFormat?.jsonSchema?.schemaText == nil)
        // The first `schema`, as JSONDecoder keeps it, with its keys sorted.
        #expect(request.responseFormat?.extractedSchema == #"{"a":2,"b":1}"#)
    }

    @Test("Without its source, a schema is re-encoded with sorted keys, the same every time")
    func schemaWithoutSourceIsSorted() throws {
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(Self.body.utf8))
        let schema = try #require(request.responseFormat?.extractedSchema)
        #expect(
            schema
                == #"{"properties":{"age":{"type":"integer"},"name":{"type":"string"},"zeta":{"type":"string"}},"#
                + #""required":["zeta","name","age"],"type":"object"}"#)
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
