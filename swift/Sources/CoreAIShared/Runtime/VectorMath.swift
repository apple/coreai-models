// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Numerically stable logistic sigmoid.
///
/// Branches on the sign so the exponent argument stays non-positive. The naive
/// `1 / (1 + exp(-x))` is fine for `x >= 0`, but on a large *negative* logit `exp(-x)`
/// overflows to `+inf` and the result underflows to exactly `0`; the equivalent
/// `exp(x) / (1 + exp(x))` form stays finite there and returns a tiny positive value.
/// (The opposite naive arrangement would produce `inf / inf = NaN` on large *positive*
/// logits — this one avoids that too.) Detector/segmenter logits are unbounded, so the
/// branch matters.
public func sigmoid(_ x: Float) -> Float {
    if x >= 0 {
        return 1 / (1 + Foundation.exp(-x))
    }
    let e = Foundation.exp(x)
    return e / (1 + e)
}
