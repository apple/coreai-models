// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - Duration Time Conversions

/// Convenience conversions from `Duration` to floating-point time units, shared
/// across the package's command-line tools and profiling code for elapsed-time reporting.
///
/// Example usage:
/// ```swift
/// let start = ContinuousClock.now
/// // ... do work ...
/// let elapsed = (ContinuousClock.now - start).inSeconds
/// print("Elapsed: \(elapsed)s")
/// ```
extension Duration {
    /// Duration in seconds as a `Double`.
    package var inSeconds: Double {
        let (secs, attoseconds) = components
        return Double(secs) + Double(attoseconds) / 1e18
    }

    /// Duration in milliseconds as a `Double`.
    package var inMilliseconds: Double {
        inSeconds * 1000.0
    }
}
