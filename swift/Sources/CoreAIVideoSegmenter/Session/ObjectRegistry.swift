// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Insertion-ordered map between client-visible object ids and the dense indices that
/// index per-object storage.
///
/// Port of the `_obj_id_to_idx` / `_obj_idx_to_id` / `obj_ids` trio in
/// `Sam3VideoInferenceSession`. Two invariants:
///
/// * `ids` order is meaningful. Tracker masks come back one row per index, and association,
///   suppression, and the output builder all zip that row order against `ids`.
/// * Removing an object renumbers every index above it, so per-object storage has to be
///   compacted in lockstep. Leaving a hole would misattribute every later object's memory.
struct ObjectRegistry {
    /// Object ids in the order they were first seen.
    private(set) var ids: [Int] = []
    private var indexByID: [Int: Int] = [:]

    var count: Int { ids.count }

    /// Index of `id`, registering it at the end if new.
    ///
    /// Returns the new index and whether it was just created, so the caller can grow its
    /// parallel storage in the same step.
    @discardableResult
    mutating func index(of id: Int) -> (index: Int, isNew: Bool) {
        if let existing = indexByID[id] { return (existing, false) }
        let index = ids.count
        indexByID[id] = index
        ids.append(id)
        return (index, true)
    }

    /// Index of `id` if it is registered.
    func existingIndex(of id: Int) -> Int? { indexByID[id] }

    func id(at index: Int) -> Int { ids[index] }

    /// Remove `id` and report the surviving indices in their old order, so callers can
    /// compact parallel arrays with `newStorage = survivors.map { oldStorage[$0] }`.
    ///
    /// Returns `nil` when `id` was never registered, matching `remove_object`, which
    /// quietly does nothing for an unknown id.
    mutating func remove(_ id: Int) -> [Int]? {
        guard let removed = indexByID[id] else { return nil }
        let survivors = (0..<ids.count).filter { $0 != removed }
        ids = survivors.map { ids[$0] }
        indexByID = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        return survivors
    }
}
