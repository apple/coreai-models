// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("BatchSessionAdapter")
struct BatchSessionAdapterTests {
    private func adapter(
        eos: Set<Int32> = [], maxTokens: Int = 1_000, stops: [[Int32]] = []
    ) -> BatchSessionAdapter {
        BatchSessionAdapter(
            request: BatchRequest(
                promptTokens: [],
                sampling: .greedy,
                eosTokenIds: eos,
                maxTokens: maxTokens,
                stopSequences: StopSequences(sequences: stops),
                decodeStopText: { $0.map(String.init).joined(separator: ",") }))
    }

    @Test("Each row retires on its own criterion and ignores later tokens")
    func retirement() {
        var rows = [adapter(eos: [99]), adapter(maxTokens: 2), adapter(stops: [[7, 8]])]
        var live: [[Bool]] = [[], [], []]
        for token: Int32 in [5, 7, 8, 99] {
            for i in rows.indices { live[i].append(rows[i].accept(token)) }
        }
        // The stop row stays live on 7, a partial match, and retires on 8.
        #expect(live == [[true, true, true, false], [true, false, false, false], [true, true, false, false]])
        #expect(rows.map(\.generated) == [[5, 7, 8, 99], [5, 7], [5, 7, 8]])
        #expect(rows.map(\.stopReason) == [.eos, .maxTokens, .stopSequence("7,8")])
    }
}
