// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIShared

// MARK: - sigmoid

/// The shared logistic sigmoid is the single home for the two segmenter copies that previously
/// lived in `DetectionDecoder` and `SegmentationPostprocessor`. Detector/segmenter logits are
/// unbounded, so these cases pin the overflow-safe behavior at both extremes.
@Suite("VectorMath")
struct VectorMathTests {
    @Test("sigmoid(0) == 0.5 and is symmetric")
    func knownValues() {
        #expect(sigmoid(0) == 0.5)
        let x: Float = 2.5
        #expect(abs(sigmoid(x) + sigmoid(-x) - 1.0) < 1e-6)
        #expect(abs(sigmoid(2) - 0.880797) < 1e-5)
    }

    @Test("sigmoid stays finite at both extremes")
    func overflowSafe() {
        // Large positive: naive `exp(x)/(1+exp(x))` would be inf/inf = NaN; this returns 1.
        #expect(sigmoid(100) == 1)
        #expect(sigmoid(100).isNaN == false)
        // Large negative: a tiny positive value, never NaN. The naive `1/(1+exp(-x))` underflows
        // to exactly 0 here; the sign-branched form keeps the denormal.
        #expect(sigmoid(-100) > 0)
        #expect(sigmoid(-100) < 1e-40)
        #expect(sigmoid(-100).isNaN == false)
    }
}
