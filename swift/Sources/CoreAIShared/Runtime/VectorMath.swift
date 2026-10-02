// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Numerically stable logistic sigmoid.
///
/// Branches on the sign to keep the exponent argument non-positive, so unbounded
/// detector/segmenter logits stay finite at both extremes (no `NaN`, no underflow to `0`).
@inline(__always)
public func sigmoid(_ x: Float) -> Float {
    if x >= 0 {
        return 1 / (1 + Foundation.exp(-x))
    }
    let e = Foundation.exp(x)
    return e / (1 + e)
}
