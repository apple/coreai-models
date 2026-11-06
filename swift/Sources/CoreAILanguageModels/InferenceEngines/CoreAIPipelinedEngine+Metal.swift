// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Metal helpers for CoreAIPipelinedEngine.

import Foundation
import Metal
import Synchronization

// MARK: - GPU Error Recording

/// The first command buffer failure on the engine's queue, for example GPU out of memory.
///
/// Core AI and MPSGraph commit their own command buffers on the queue and don't report one
/// that fails, so its logits are never written. The engine checks this before each yield.
final class GPUErrorHolder: Sendable {
    private let first = Mutex<(any Error)?>(nil)

    var error: (any Error)? { first.withLock { $0 } }

    /// Record `error` unless an earlier one is already recorded.
    func record(_ error: any Error) {
        first.withLock { if $0 == nil { $0 = error } }
    }

    func reset() { first.withLock { $0 = nil } }

    /// Yield `value`, or end the stream with the recorded error instead.
    func yield<T: Sendable>(_ value: T, to continuation: AsyncThrowingStream<T, any Error>.Continuation) {
        if let error { continuation.finish(throwing: error) } else { continuation.yield(value) }
    }
}

/// A command queue that records each failed command buffer made from it in `errors`.
/// Everything else is forwarded to `base`.
final class ErrorRecordingCommandQueue: NSObject, MTLCommandQueue, @unchecked Sendable {
    // @unchecked: both stored properties are thread-safe.
    let base: any MTLCommandQueue
    let errors: GPUErrorHolder

    init(_ base: any MTLCommandQueue, errors: GPUErrorHolder) {
        self.base = base
        self.errors = errors
    }

    // MARK: Forwarding functions
    // MTLCommandQueue requirements, forwarded to `base`. The makeCommandBuffer variants
    // also add the recording handler.

    var label: String? {
        get { base.label }
        set { base.label = newValue }
    }
    var device: any MTLDevice { base.device }

    func makeCommandBuffer() -> (any MTLCommandBuffer)? { recording(base.makeCommandBuffer()) }
    func makeCommandBuffer(descriptor: MTLCommandBufferDescriptor) -> (any MTLCommandBuffer)? {
        recording(base.makeCommandBuffer(descriptor: descriptor))
    }
    func makeCommandBufferWithUnretainedReferences() -> (any MTLCommandBuffer)? {
        recording(base.makeCommandBufferWithUnretainedReferences())
    }
    func insertDebugCaptureBoundary() { base.insertDebugCaptureBoundary() }
    func addResidencySet(_ residencySet: any MTLResidencySet) { base.addResidencySet(residencySet) }
    func removeResidencySet(_ residencySet: any MTLResidencySet) { base.removeResidencySet(residencySet) }
    // swift-format-ignore: AlwaysUseLowerCamelCase
    func __addResidencySets(_ residencySets: UnsafePointer<any MTLResidencySet>, count: Int) {
        base.__addResidencySets(residencySets, count: count)
    }
    // swift-format-ignore: AlwaysUseLowerCamelCase
    func __removeResidencySets(_ residencySets: UnsafePointer<any MTLResidencySet>, count: Int) {
        base.__removeResidencySets(residencySets, count: count)
    }

    // Other selectors a framework calls on the queue go to the real one.
    override func forwardingTarget(for aSelector: Selector!) -> Any? { base }
    override func responds(to aSelector: Selector!) -> Bool {
        super.responds(to: aSelector) || base.responds(to: aSelector)
    }

    // MARK: Recording

    private func recording(_ commandBuffer: (any MTLCommandBuffer)?) -> (any MTLCommandBuffer)? {
        let errors = self.errors
        commandBuffer?.addCompletedHandler { buffer in
            if let error = buffer.error {
                errors.record(error)
            } else if buffer.status == .error {
                errors.record(InferenceRuntimeError.genericError("GPU command buffer failed"))
            }
        }
        return commandBuffer
    }
}
