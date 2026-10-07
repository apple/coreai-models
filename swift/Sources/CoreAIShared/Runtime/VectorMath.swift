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

/// Cosine similarity of two equal-length vectors, accumulated in `Double` and returned as `Float`.
///
/// Returns `0` for empty, length-mismatched, or zero-norm inputs. The parity harnesses that
/// use this report the value as a diagnostic, so the single documented convention keeps their
/// numbers comparable.
public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    for i in a.indices {
        dot += Double(a[i]) * Double(b[i])
        normA += Double(a[i]) * Double(a[i])
        normB += Double(b[i]) * Double(b[i])
    }
    let denom = (normA * normB).squareRoot()
    return denom > 0 ? Float(dot / denom) : 0
}
