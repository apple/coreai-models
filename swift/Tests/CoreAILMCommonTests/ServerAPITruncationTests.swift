// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILMCommon

@Suite("Truncation request field")
struct ServerAPITruncationTests {
    private func decode(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    }

    @Test("Absent truncation defaults to off")
    func absentDefaultsOff() throws {
        let req = try decode(#"{"messages":[{"role":"user","content":"Hi"}]}"#)
        #expect(req.truncation == .off)
    }

    @Test("String off / disabled / none decode to off")
    func stringOff() throws {
        for value in ["off", "disabled", "none"] {
            let req = try decode(#"{"messages":[{"role":"user","content":"Hi"}],"truncation":"\#(value)"}"#)
            #expect(req.truncation == .off)
        }
    }

    @Test("String auto decodes to auto")
    func stringAuto() throws {
        let req = try decode(#"{"messages":[{"role":"user","content":"Hi"}],"truncation":"auto"}"#)
        #expect(req.truncation == .auto)
    }

    @Test("Positive integer decodes to tokensAt")
    func integerTokens() throws {
        let req = try decode(#"{"messages":[{"role":"user","content":"Hi"}],"truncation":500}"#)
        #expect(req.truncation == .tokensAt(500))
    }

    @Test("Non-positive integer is rejected")
    func nonPositiveRejected() {
        #expect(throws: (any Error).self) {
            _ = try decode(#"{"messages":[{"role":"user","content":"Hi"}],"truncation":0}"#)
        }
        #expect(throws: (any Error).self) {
            _ = try decode(#"{"messages":[{"role":"user","content":"Hi"}],"truncation":-5}"#)
        }
    }

    @Test("Unknown string is rejected")
    func unknownStringRejected() {
        #expect(throws: (any Error).self) {
            _ = try decode(#"{"messages":[{"role":"user","content":"Hi"}],"truncation":"bogus"}"#)
        }
    }
}
