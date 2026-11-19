// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// How many requests a server decodes together in one batch: `auto` or a fixed count.
///
/// `auto` uses the model's maximum batch size when the model supports batching, and one request at
/// a time otherwise. A fixed count above 1 needs a model that supports batching with a maximum batch
/// size of at least that count.
///
/// ```swift
/// let requested = ConcurrentRequests("auto")!
/// let rows = try requested.resolve(supportsBatching: true, maxBatchSize: 8)  // 8
/// ```
package enum ConcurrentRequests: Equatable, Sendable {
    case auto
    case count(Int)

    /// Parses `auto` or a positive integer; returns nil for anything else.
    package init?(_ text: String) {
        if text == "auto" {
            self = .auto
        } else if let count = Int(text), count >= 1 {
            self = .count(count)
        } else {
            return nil
        }
    }

    /// The number of requests to decode together for a model; 1 means one request at a time.
    ///
    /// A fixed count above 1 resolves to 1 when the model does not support batching, and throws
    /// ``ConcurrentRequestsError/exceedsMaxBatchSize(requested:maxBatchSize:)`` when it is above the
    /// model's maximum batch size.
    package func resolve(supportsBatching: Bool, maxBatchSize: Int) throws(ConcurrentRequestsError) -> Int {
        guard supportsBatching else { return 1 }
        switch self {
        case .auto:
            return max(1, maxBatchSize)
        case .count(let count):
            guard count <= maxBatchSize else {
                throw .exceedsMaxBatchSize(requested: count, maxBatchSize: maxBatchSize)
            }
            return count
        }
    }
}

/// A concurrent-request count the model cannot serve.
package enum ConcurrentRequestsError: Error, Equatable {
    /// The requested count is above the maximum batch size the model was exported with.
    case exceedsMaxBatchSize(requested: Int, maxBatchSize: Int)
}
