// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

/// Concrete async sequence returned by `InferenceEngine.generate()`.
///
/// Wraps a pull-based async iterator and tracks why generation ended.
/// Reference type — the consumer reads `stopReason` after iteration, and a
/// decoder may set it explicitly (e.g. `.eos`) before breaking out.
///
/// Two construction modes:
/// - `init(_:)` wraps a pull-based iterator. Generation work runs on the
///   consumer's task as it pulls tokens — no producer `Task`, no unstructured
///   concurrency. This is what the CPU engines use.
/// - `makeStream()` returns a continuation an engine can drive from a
///   background `Task` (e.g. the GPU-pipelined engine, which samples on-device).
///
/// In both modes, natural exhaustion (`next()` returning nil) is recorded as
/// `.maxTokens` unless a reason was already set, and a thrown error/cancellation
/// is recorded as `.error`/`.cancelled`.
public final class InferenceStream: AsyncSequence, Sendable {
    public typealias Element = InferenceOutput

    // MARK: - StopReason

    /// Why token generation terminated.
    public enum StopReason: Sendable, Equatable {
        /// The maximum token limit was reached.
        case maxTokens
        /// An end-of-sequence token was generated.
        case eos
        /// A stop sequence was matched in the output.
        case stopSequence(String)
        /// Generation was cancelled (Task cancellation or explicit cancel).
        case cancelled
        /// An unrecoverable error occurred during generation.
        case error
    }

    // MARK: - Init

    private let makeBase: @Sendable () -> any AsyncIteratorProtocol<InferenceOutput, any Error>
    private let _stopReason: Mutex<StopReason?>

    /// Wrap a pull-based iterator factory. The iterator drives generation lazily
    /// on the consumer's task; no work happens until the stream is iterated.
    init(
        _ makeBase: @escaping @Sendable () -> any AsyncIteratorProtocol<InferenceOutput, any Error>
    ) {
        self.makeBase = makeBase
        self._stopReason = Mutex(nil)
    }

    // MARK: - Public API

    /// Why generation stopped. Nil while the stream is still active.
    /// Guaranteed non-nil after the `for try await` loop exits.
    public var stopReason: StopReason? {
        _stopReason.withLock { $0 }
    }

    // MARK: - Package-internal

    /// Engines and decoders call this when they know why generation ended.
    func setStopReason(_ reason: StopReason) {
        _stopReason.withLock { $0 = reason }
    }

    /// Record a reason only if none was set yet — used to mark natural
    /// exhaustion as `.maxTokens` without clobbering a reason a decoder already
    /// set (e.g. `.eos` before breaking out of the loop).
    private func setStopReasonIfUnset(_ reason: StopReason) {
        _stopReason.withLock { if $0 == nil { $0 = reason } }
    }

    // MARK: - Factory

    /// Create a stream + continuation pair for engines that drive output from a
    /// background `Task` (e.g. the GPU-pipelined engine).
    static func makeStream() -> (
        stream: InferenceStream,
        continuation: AsyncThrowingStream<InferenceOutput, any Error>.Continuation
    ) {
        let (base, continuation) = AsyncThrowingStream<InferenceOutput, any Error>.makeStream()
        return (InferenceStream { base.makeAsyncIterator() }, continuation)
    }

    // MARK: - AsyncSequence

    public struct AsyncIterator: AsyncIteratorProtocol {
        var base: any AsyncIteratorProtocol<InferenceOutput, any Error>
        let stream: InferenceStream

        public mutating func next() async throws -> InferenceOutput? {
            do {
                guard let element = try await base.next() else {
                    stream.setStopReasonIfUnset(.maxTokens)
                    return nil
                }
                return element
            } catch is CancellationError {
                stream.setStopReason(.cancelled)
                throw CancellationError()
            } catch {
                stream.setStopReason(.error)
                throw error
            }
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(base: makeBase(), stream: self)
    }
}
