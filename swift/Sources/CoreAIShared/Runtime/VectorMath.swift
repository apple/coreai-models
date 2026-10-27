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
/// Length-mismatched inputs score `0`. For zero-norm inputs the convention matches the parity
/// harnesses: two zero-norm (or empty) vectors are identical and score `1`, while a single
/// zero-norm vector has no direction to compare and scores `0`.
public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return 0 }
    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    for i in a.indices {
        dot += Double(a[i]) * Double(b[i])
        normA += Double(a[i]) * Double(a[i])
        normB += Double(b[i]) * Double(b[i])
    }
    let denom = (normA * normB).squareRoot()
    guard denom > 0 else { return normA == normB ? 1 : 0 }
    return Float(dot / denom)
}
