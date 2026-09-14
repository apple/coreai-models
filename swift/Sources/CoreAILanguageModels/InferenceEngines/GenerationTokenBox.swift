// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

/// Thread-safe holder for an engine's in-flight `GenerationToken`.
///
/// Both `CoreAISequentialEngine` and `CoreAISequentialVLMEngine` track a single active
/// generation so a new `generate()` can cancel the previous one and the iterator can
/// release the engine when it finishes. This centralizes that `Mutex<GenerationToken?>`
/// bookkeeping so both engines share one implementation.
final class GenerationTokenBox: Sendable {
    private let _token = Mutex<GenerationToken?>(nil)

    /// True while a generation is in flight.
    var isBusy: Bool { _token.withLock { $0 != nil } }

    /// Cancel the active token (if any) and clear it.
    func cancelActive() {
        _token.withLock {
            $0?.cancel()
            $0 = nil
        }
    }

    /// Install `token` as the active generation. Does not cancel any prior token —
    /// callers that need to supersede an in-flight generation call `cancelActive()` first.
    func install(_ token: GenerationToken) {
        _token.withLock { $0 = token }
    }

    /// Clear the active token only if it is `token`. Called by the iterator when
    /// generation finishes or is cancelled, so a newer generation is left untouched.
    func clearIfActive(_ token: GenerationToken) {
        _token.withLock { if $0 === token { $0 = nil } }
    }
}
