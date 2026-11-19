// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization

/// One cached grammar session, as an engine that runs one guided request at a time keeps it.
///
/// Taking the session empties the slot, so two requests never share one, and a cached session is
/// handed out again only for a request with the key it was built for.
package final class ConstrainedSessionSlot: Sendable {
    private let held = Mutex<ConstrainedSessionHandle?>(nil)

    package init() {}

    /// The held session, reset, when it was built for `key`; otherwise a new one from `make`. Either
    /// way the slot is left empty.
    package func checkOut(
        _ key: ConstrainedSessionKey, make: () throws -> ConstrainedSessionHandle
    ) rethrows -> ConstrainedSessionHandle {
        let cached = held.withLock { held -> ConstrainedSessionHandle? in
            defer { held = nil }
            return held?.key == key ? held : nil
        }
        if let cached {
            cached.reset()
            return cached
        }
        return try make()
    }

    /// Keeps `handle` for the next request, in place of any session held now.
    package func checkIn(_ handle: ConstrainedSessionHandle) {
        held.withLock { $0 = handle }
    }
}
