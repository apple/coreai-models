// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

/// Concrete async sequence returned by `InferenceEngine.generate()`.
///
/// Wraps an `AsyncThrowingStream<InferenceOutput, any Error>` and tracks
/// why generation ended. Reference type — the producer sets `stopReason`
/// and the consumer reads it after iteration.
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

    private let base: AsyncThrowingStream<InferenceOutput, any Error>
    private let _stopReason: Mutex<StopReason?>

    init(base: AsyncThrowingStream<InferenceOutput, any Error>) {
        self.base = base
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

    // MARK: - Factory

    /// Create a stream + continuation pair for engines to drive.
    static func makeStream() -> (
        stream: InferenceStream,
        continuation: AsyncThrowingStream<InferenceOutput, any Error>.Continuation
    ) {
        let (base, continuation) = AsyncThrowingStream<InferenceOutput, any Error>.makeStream()
        return (InferenceStream(base: base), continuation)
    }

    // MARK: - AsyncSequence

    public struct AsyncIterator: AsyncIteratorProtocol {
        var base: AsyncThrowingStream<InferenceOutput, any Error>.AsyncIterator
        let stream: InferenceStream

        public mutating func next() async throws -> InferenceOutput? {
            do {
                guard let element = try await base.next() else {
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
        AsyncIterator(base: base.makeAsyncIterator(), stream: self)
    }
}
