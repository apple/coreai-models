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
/// Completions scoring does not use the router: with a scheduler it runs through
/// `BatchScheduler.score(_:)`, otherwise on the shared engine.
package enum BatchServingRouter {
    /// Returns `.fallbackBatchOne` when the chat request has a `response_format` JSON schema,
    /// otherwise `.batched`.
    package static func lane(hasResponseSchema: Bool) -> BatchServingLane {
        hasResponseSchema ? .fallbackBatchOne : .batched
    }
}
