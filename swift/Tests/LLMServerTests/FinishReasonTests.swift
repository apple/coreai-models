// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import Foundation
import Testing

@testable import llm_server

/// A finished stream that reports `stopReason`, `tokensGenerated` and, when recorded, the token IDs.
private struct EndedStream: AsyncSequence, DecodeOutcomeReporting {
    typealias Element = GenerationResult
    typealias Failure = Error
    let stopReason: StopReason?
    var tokensGenerated: Int = 0
    var tokenIds: [Int32]? = nil

    func generatedTokenIds(from start: Int) -> [Int32]? { tokenIds.map { Array($0[start...]) } }

    struct AsyncIterator: AsyncIteratorProtocol {
        mutating func next() async throws -> GenerationResult? { nil }
    }

    func makeAsyncIterator() -> AsyncIterator { AsyncIterator() }
}

/// A finished stream that does not say why it ended or what it took.
private struct SilentStream: AsyncSequence {
    typealias Element = GenerationResult
    typealias Failure = Error

    struct AsyncIterator: AsyncIteratorProtocol {
        mutating func next() async throws -> GenerationResult? { nil }
    }

    func makeAsyncIterator() -> AsyncIterator { AsyncIterator() }
}

@Suite("finish_reason and completion tokens")
struct FinishReasonTests {
    @Test("The stream's stop reason decides, not the token count")
    func reasonOverCount() throws {
        // Ended by a stop sequence on exactly the limit's token: still "stop".
        #expect(
            try reportedFinishReason(
                of: EndedStream(stopReason: .stopSequence("\n")), generated: 8, maxTokens: 8) == "stop")
        #expect(try reportedFinishReason(of: EndedStream(stopReason: .eos), generated: 3, maxTokens: 8) == "stop")
        // Cut off by the limit with fewer results than tokens: still "length".
        #expect(
            try reportedFinishReason(of: EndedStream(stopReason: .maxTokens), generated: 5, maxTokens: 8)
                == "length")
    }

    @Test("A stream that ended on an error, or was cancelled, throws rather than reporting a finished reply")
    func errorAndCancelThrow() {
        #expect(throws: (any Error).self) {
            try reportedFinishReason(of: EndedStream(stopReason: .error), generated: 2, maxTokens: 8)
        }
        #expect(throws: (any Error).self) {
            try reportedFinishReason(of: EndedStream(stopReason: .cancelled), generated: 2, maxTokens: 8)
        }
    }

    @Test("A stream that reports nothing falls back to the count")
    func silentFallsBack() throws {
        #expect(try reportedFinishReason(of: SilentStream(), generated: 8, maxTokens: 8) == "length")
        #expect(try reportedFinishReason(of: SilentStream(), generated: 3, maxTokens: 8) == "stop")
    }

    @Test("Completion tokens are the stream's own count, which includes tokens that yielded no result")
    func tokenCountFromTheStream() {
        #expect(reportedTokenCount(of: EndedStream(stopReason: .eos, tokensGenerated: 5), results: 2) == 5)
        #expect(reportedTokenCount(of: EndedStream(stopReason: .eos), results: 2) == 2)
        #expect(reportedTokenCount(of: SilentStream(), results: 3) == 3)
    }

    @Test("return_token_ids reads the stream's IDs from an offset, and nothing when not asked for")
    func tokenIds() throws {
        let stream = EndedStream(stopReason: .eos, tokensGenerated: 3, tokenIds: [7, 8, 9])
        #expect(try generatedTokenIds(of: stream, from: 0, requested: false) == nil)
        #expect(try generatedTokenIds(of: stream, from: 0, requested: true) == [7, 8, 9])
        #expect(try generatedTokenIds(of: stream, from: 2, requested: true) == [9])
    }

    @Test("return_token_ids fails on a stream that recorded no IDs")
    func tokenIdsMissing() {
        #expect(throws: InferenceRuntimeError.self) {
            try generatedTokenIds(of: EndedStream(stopReason: .eos), from: 0, requested: true)
        }
        #expect(throws: InferenceRuntimeError.self) {
            try generatedTokenIds(of: SilentStream(), from: 0, requested: true)
        }
    }

    @Test("A stream that fails after it started ends with an error frame")
    func errorFrame() throws {
        struct Boom: Error, CustomStringConvertible { var description: String { "the GPU went away" } }
        let frame = streamErrorFrame(Boom())
        #expect(frame.hasPrefix("data: {"))
        #expect(frame.hasSuffix("\n\n"))
        let json = try JSONSerialization.jsonObject(with: Data(frame.dropFirst(6).utf8)) as? [String: Any]
        let error = json?["error"] as? [String: Any]
        #expect(error?["message"] as? String == "the GPU went away")
        #expect(error?["type"] as? String == "server_error")
    }
}
