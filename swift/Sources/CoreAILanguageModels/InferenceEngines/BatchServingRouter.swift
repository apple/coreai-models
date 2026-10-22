// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Serving lane of a request under a batched server.
package enum BatchServingLane: Equatable, Sendable {
    /// Requests that share the batch scheduler through a `SchedulerSession`.
    case batched
    /// Requests that run one at a time on their own engine session, such as guided generation.
    case fallbackBatchOne
}

/// Lane selection for chat requests under a batched server.
package enum BatchServingRouter {
    /// Returns `.fallbackBatchOne` for a request with a `response_format` JSON schema, otherwise
    /// `.batched`.
    package static func lane(hasResponseSchema: Bool) -> BatchServingLane {
        hasResponseSchema ? .fallbackBatchOne : .batched
    }
}
