// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Persistent model state that the engine carries across inference steps.
///
/// Each handler manages one or more named state tensors (e.g., key_cache + value_cache,
/// or a single convolution_state). The engine owns handlers and delegates all state
/// lifecycle to them — allocation, growth, and reset.
///
/// Handlers are classes (AnyObject) so they own their NDArrays at refcount 1 —
/// `bind(into:)` calls `mutableRawView()` without triggering COW. The loop uses
/// `_overrideLifetime` to express disjoint element access to the compiler.
public protocol SyncStateHandler: AnyObject {
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
    func ensureCapacity(forContextLength contextLength: Int) throws -> Bool

    /// Access a state array by name+index (value copy — use bind(into:) on hot paths).
    subscript(stateIndex index: Int) -> (name: String, array: NDArray) { get set }

    /// Insert all managed states into `views`. Zero-copy: uses reference-backed
    /// storage internally, so mutableRawView() never triggers COW.
    @_lifetime(views: borrow self)
    func bind(into views: inout InferenceFunction.MutableViews)

    /// Full reset — zero all backing storage, rewind to position 0.
    func reset()

    /// Truncate to a given token position.
    func truncate(to tokenCount: Int)
}

// MARK: - Lifetime Helpers

/// Detach lifetime dependencies from MutableViews so it can cross scope
/// boundaries (closures, await). Caller must ensure inserted arrays remain valid.
@inline(__always)
@_unsafeNonescapableResult
@_lifetime(immortal)
func _unsafeEscapeMutableViews(
    _ views: consuming InferenceFunction.MutableViews
) -> InferenceFunction.MutableViews {
    views
}
