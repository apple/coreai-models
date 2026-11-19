// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Token-embedding rows a drafter has already fetched from the target, bounded so a long session
/// over a large vocabulary cannot grow it without limit. Inserting into a full cache clears it
/// first; the working set of a draft loop is small, so it refills with the tokens in use.
struct EmbeddingCache {
    /// Rows held at most. A row is one backbone-width `[Float]`.
    static let defaultCapacity = 4096

    private var rows: [Int32: [Float]] = [:]
    let capacity: Int

    init(capacity: Int = EmbeddingCache.defaultCapacity) {
        self.capacity = Swift.max(1, capacity)
    }

    var count: Int { rows.count }

    subscript(token: Int32) -> [Float]? { rows[token] }

    mutating func insert(_ row: [Float], for token: Int32) {
        if rows.count >= capacity, rows[token] == nil { rows.removeAll(keepingCapacity: true) }
        rows[token] = row
    }

    mutating func removeAll() { rows.removeAll(keepingCapacity: true) }
}
