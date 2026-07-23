// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - ContinuousClock Timing Utilities

/// Extension to convert Duration to common time units.
///
/// - Note: This is an internal implementation detail. It is intentionally not
///   `public`: vending members on a standard-library type we don't own would
///   pollute `Duration`'s API surface for every client of this library.
///
/// Example usage:
/// ```swift
/// let start = ContinuousClock.now
/// // ... do work ...
/// let elapsed = (ContinuousClock.now - start).inMilliseconds
/// print("Elapsed: \(elapsed)ms")
/// ```
extension Duration {
    /// Duration in seconds as a Double.
    var inSeconds: Double {
        let (secs, attoseconds) = self.components
        return Double(secs) + Double(attoseconds) / 1e18
    }

    /// Duration in milliseconds as a Double.
    var inMilliseconds: Double {
        inSeconds * 1000.0
    }
}
