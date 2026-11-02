// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Embedding cache bounds")
struct EmbeddingCacheTests {
    @Test("The embedding cache never holds more than its capacity")
    func embeddingCacheIsBounded() {
        var cache = EmbeddingCache(capacity: 4)
        for t in 0..<100 {
            cache.insert([Float(t)], for: Int32(t))
            #expect(cache.count <= 4)
        }
        #expect(cache.count > 0)
        #expect(cache[99] == [99])
    }

    @Test("A full cache is cleared before the next insert, and lookups of absent tokens are nil")
    func embeddingCacheClearsWhenFull() {
        var cache = EmbeddingCache(capacity: 2)
        cache.insert([1], for: 1)
        cache.insert([2], for: 2)
        #expect(cache[1] == [1] && cache[2] == [2])
        cache.insert([3], for: 3)
        #expect(cache[1] == nil && cache[2] == nil)
        #expect(cache[3] == [3])
        cache.removeAll()
        #expect(cache.count == 0)
    }
}
