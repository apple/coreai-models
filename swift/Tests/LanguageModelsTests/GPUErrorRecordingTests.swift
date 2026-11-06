// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Metal
import Testing

@testable import CoreAILanguageModels

@Suite("GPU error recording")
struct GPUErrorRecordingTests {
    @Test("a recorded error replaces the next yield with a throw, and the first error wins")
    func recordedErrorEndsStream() async throws {
        let errors = GPUErrorHolder()
        let (stream, continuation) = AsyncThrowingStream<Int32, any Error>.makeStream()
        let outOfMemory = NSError(
            domain: MTLCommandBufferErrorDomain, code: Int(MTLCommandBufferError.outOfMemory.rawValue))

        errors.yield(1, to: continuation)
        errors.record(outOfMemory)
        errors.record(NSError(domain: "later", code: 1))
        errors.yield(2, to: continuation)

        var tokens: [Int32] = []
        await #expect(throws: outOfMemory) {
            for try await token in stream { tokens.append(token) }
        }
        #expect(tokens == [1])
        #expect(errors.error as NSError? == outOfMemory)

        errors.reset()
        #expect(errors.error == nil)
    }

    @Test("a command buffer made through the wrapper completes with no error recorded")
    func wrapperRecordsNothingOnSuccess() async throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let errors = GPUErrorHolder()
        let queue = ErrorRecordingCommandQueue(try #require(device.makeCommandQueue()), errors: errors)
        let commandBuffer = try #require(queue.makeCommandBuffer())
        // Runs after the wrapper's handler, which was added first.
        await withCheckedContinuation { done in
            commandBuffer.addCompletedHandler { _ in done.resume() }
            commandBuffer.commit()
        }
        #expect(commandBuffer.status == .completed)
        #expect(errors.error == nil)
    }
}
