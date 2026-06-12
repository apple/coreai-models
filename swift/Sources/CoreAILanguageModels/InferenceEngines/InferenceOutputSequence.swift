// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

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

/// The async sequence of `InferenceOutput` returned by `InferenceEngine.generate()`.
///
/// Beyond yielding tokens, it records *why* generation ended:
/// - The engine-driven iterator marks `.maxTokens` on natural exhaustion and
///   `.cancelled`/`.error` if iteration throws.
/// - A consumer (decoder) that stops early — e.g. on an EOS token or a matched
///   stop sequence — calls `setStopReason(_:)` before breaking out of the loop.
///
/// Read `stopReason` after the `for try await` loop exits; it is guaranteed
/// non-nil once iteration has run to completion.
public protocol InferenceOutputSequence: AsyncSequence<InferenceOutput, any Error> {
    /// Why generation stopped. Nil while the stream is still active.
    var stopReason: StopReason? { get }

    /// Record why generation stopped. Called by the engine iterator for
    /// engine-driven reasons, or by a decoder that terminates early.
    func setStopReason(_ reason: StopReason)
}

/// Shared, thread-safe slot for a sequence's `StopReason`.
///
/// `InferenceOutputSequence`s are value types, but the reason is written by the
/// iterator (or a producer Task) and read by the caller after iteration. A
/// reference-typed box lets the sequence value, its iterator, and the caller
/// observe the same slot.
final class StopReasonStore: Sendable {
    private let value = Mutex<StopReason?>(nil)

    var stopReason: StopReason? {
        value.withLock { $0 }
    }

    /// Overwrite the recorded reason.
    func set(_ reason: StopReason) {
        value.withLock { $0 = reason }
    }

    /// Record a reason only if none was set yet — used to mark natural
    /// exhaustion as `.maxTokens` without clobbering a reason a decoder already
    /// set (e.g. `.eos` before breaking out of the loop).
    func setIfUnset(_ reason: StopReason) {
        value.withLock { if $0 == nil { $0 = reason } }
    }
}
