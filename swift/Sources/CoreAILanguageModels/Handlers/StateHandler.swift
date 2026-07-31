// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Persistent model state that the engine carries across inference steps.
///
/// Each handler manages one or more named state tensors (e.g., key_cache + value_cache,
/// or a single convolution_state). The engine owns an array of handlers and delegates
/// all state lifecycle to them — allocation, growth, and reset.
///
/// ## MutableViews Lifetime Constraint
///
/// `InferenceFunction.MutableViews.insert` creates a lifetime dependency on each `inout`
/// variable. This means state binding CANNOT be abstracted into a method that takes
/// `inout MutableViews` — the inserts must happen in the same scope as `function.run()`.
/// Handlers therefore expose their arrays directly via subscript for the engine to bind.
public protocol SyncStateHandler {
    /// Names of the states managed by this handler.
    var stateNames: [String] { get }

    /// Number of state arrays managed.
    var stateCount: Int { get }

    /// Current capacity in the sequence/context dimension.
    /// For fixed-size states this equals max capacity.
    var currentCapacity: Int { get }

    /// Whether this state supports in-place truncation (cursor rewind).
    /// KV cache: true (causal mask hides positions beyond the cursor).
    /// Recurrent/conv: false (no independent token axis).
    var supportsTruncation: Bool { get }

    /// Ensure the state can accommodate `contextLength` tokens.
    /// Returns true if reallocation occurred.
    mutating func ensureCapacity(forContextLength contextLength: Int) throws -> Bool

    /// Access a state array by index for binding into MutableViews.
    /// The engine calls this to get name + array pairs for insert.
    subscript(stateIndex index: Int) -> (name: String, array: NDArray) { get set }

    /// Full reset — zero all backing storage, rewind to position 0.
    mutating func reset()

    /// Truncate to a given token position.
    /// Only valid when `supportsTruncation == true`. For KV cache, this is a no-op
    /// on the backing storage (causal mask handles visibility); the engine just
    /// updates its processedTokenCount.
    mutating func truncate(to tokenCount: Int)
}
