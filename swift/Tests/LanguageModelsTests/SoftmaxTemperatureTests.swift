// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Temperature softmax")
struct SoftmaxTemperatureTests {
    @Test("Temperature zero gives a one-hot at the first maximum instead of NaN")
    func zeroTemperatureIsGreedy() {
        let probs = SpeculativeMath.temperatureSoftmax([1, 3, 3, 2], temperature: 0)
        #expect(probs == [0, 1, 0, 0])
        #expect(SpeculativeMath.temperatureSoftmax([1, 3, 2], temperature: -1) == [0, 1, 0])
        let positive = SpeculativeMath.temperatureSoftmax([1, 3, 2], temperature: 1)
        #expect(abs(positive.reduce(0, +) - 1) < 1e-5)
    }
}
