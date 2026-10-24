// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("ConcurrentRequests")
struct ConcurrentRequestsTests {
    @Test(
        "Parses auto and positive integers",
        arguments: [
            ("auto", ConcurrentRequests.auto),
            ("1", .count(1)),
            ("8", .count(8)),
            ("0", nil),
            ("-2", nil),
            ("max", nil),
        ] as [(String, ConcurrentRequests?)])
    func parses(text: String, expected: ConcurrentRequests?) {
        #expect(ConcurrentRequests(text) == expected)
    }

    @Test(
        "Resolves to the model's max batch size for auto, and to 1 without batching",
        arguments: [
            (ConcurrentRequests.auto, true, 8, 8),
            (.auto, false, 8, 1),
            (.count(1), true, 8, 1),
            (.count(4), true, 8, 4),
            (.count(8), true, 8, 8),
            (.count(4), false, 1, 1),
        ] as [(ConcurrentRequests, Bool, Int, Int)])
    func resolves(requested: ConcurrentRequests, supportsBatching: Bool, maxBatchSize: Int, expected: Int) throws {
        let rows = try requested.resolve(supportsBatching: supportsBatching, maxBatchSize: maxBatchSize)
        #expect(rows == expected)
    }

    @Test("A count above the model's max batch size throws")
    func rejectsAboveMaxBatchSize() {
        #expect(throws: ConcurrentRequestsError.exceedsMaxBatchSize(requested: 16, maxBatchSize: 8)) {
            try ConcurrentRequests.count(16).resolve(supportsBatching: true, maxBatchSize: 8)
        }
    }
}
