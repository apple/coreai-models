// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILMCommon
import Foundation
import Testing

/// Coverage for the OpenAI-compatible determinism fields added for reproducible
/// sampling: the request `seed` and the response/chunk `system_fingerprint`.
struct ServerAPISeedTests {
    private func decodeRequest(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    }

    private func encodeToString(_ value: some Encodable) throws -> String {
        String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
    }

    // MARK: - Request seed decoding

    @Test("seed decodes from the request body")
    func seedDecodes() throws {
        let req = try decodeRequest(#"{"messages":[{"role":"user","content":"hi"}],"seed":42}"#)
        #expect(req.seed == 42)
    }

    @Test("seed is nil when absent")
    func seedAbsent() throws {
        let req = try decodeRequest(#"{"messages":[{"role":"user","content":"hi"}]}"#)
        #expect(req.seed == nil)
    }

    // MARK: - Response system_fingerprint encoding

    private func makeResponse(fingerprint: String?) -> ChatCompletionResponse {
        ChatCompletionResponse(
            id: "coreai-1",
            model: "test-model",
            choices: [
                .init(
                    index: 0,
                    message: .init(role: "assistant", content: "hello"),
                    finishReason: "stop")
            ],
            usage: nil,
            systemFingerprint: fingerprint)
    }

    @Test("system_fingerprint is emitted when set")
    func fingerprintEmitted() throws {
        let json = try encodeToString(makeResponse(fingerprint: "fp_abc123"))
        #expect(json.contains(#""system_fingerprint":"fp_abc123""#))
    }

    @Test("system_fingerprint is omitted when nil")
    func fingerprintOmitted() throws {
        let json = try encodeToString(makeResponse(fingerprint: nil))
        #expect(!json.contains("system_fingerprint"))
    }

    // MARK: - Chunk system_fingerprint encoding

    @Test("streaming chunk carries system_fingerprint")
    func chunkFingerprint() throws {
        let chunk = ChatCompletionChunk(
            id: "coreai-1",
            model: "test-model",
            choices: [.init(index: 0, delta: .init(role: "assistant", content: nil), finishReason: nil)],
            systemFingerprint: "fp_xyz")
        let json = try encodeToString(chunk)
        #expect(json.contains(#""system_fingerprint":"fp_xyz""#))
    }
}
