// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("BatchServingRouter")
struct BatchServingRouterTests {
    @Test(
        "Plain chat batches and a response schema takes the batch=1 lane",
        arguments: [(false, BatchServingLane.batched), (true, .fallbackBatchOne)])
    func lane(hasResponseSchema: Bool, expected: BatchServingLane) {
        #expect(BatchServingRouter.lane(hasResponseSchema: hasResponseSchema) == expected)
    }
}
