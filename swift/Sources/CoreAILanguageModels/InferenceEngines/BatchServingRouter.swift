// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Which serving lane a request takes under a batched server.
package enum BatchServingLane: Equatable, Sendable {
    /// Served by the batch scheduler through a `SchedulerSession`.
    case batched
    /// Served by a single-request engine that returns per-step logits.
    case fallbackBatchOne
}

/// Chooses the serving lane for a request when batched serving is enabled.
///
/// The batched lane samples inside the scheduler and returns no per-step logits, so a request with a
/// JSON schema takes the fallback lane, where constrained decoding can mask the logits at each step.
/// Loglikelihood scoring also needs logits, but the completions handler sends it to the fallback lane
/// directly without asking the router.
package enum BatchServingRouter {
    /// Returns `.fallbackBatchOne` when the chat request has a `response_format` JSON schema,
    /// otherwise `.batched`.
    package static func lane(hasResponseSchema: Bool) -> BatchServingLane {
        hasResponseSchema ? .fallbackBatchOne : .batched
    }
}
