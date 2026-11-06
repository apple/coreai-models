// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// With sampling, the whitespace limit must not change a generation until the unlimited generation
/// has itself used more whitespace than the limit allows.
@Suite("Whitespace limit under sampling")
struct WhitespaceLimitSamplingTests {
    private static let vocabulary = [
        "{", "}", "[", "]", ":", ",", "\"", " ", "\n", "\t",
        "name", "age", "email", "city", "true", "false", "null",
        "John", "Jane", "Doe", "Smith", "user", "admin", "guest", "status", "active",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "25",
    ]
    private static let whitespaceTokens: Set<Int32> = [7, 8, 9]
    private static let schema = """
        {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
         "required": ["name", "age"], "additionalProperties": false}
        """

    /// One generation: the logits at each step depend only on `seed` and the step, so two runs see
    /// the same model whatever tokens they chose, and sampling uses the same seeded stream.
    private func generate(
        seed: UInt64, limit: Int?, temperature: Double, whitespaceBoost: Float, steps: Int = 120
    ) throws -> [Int32] {
        var session = try ConstrainedGenerationSession(
            jsonSchema: Self.schema, vocabulary: Self.vocabulary, vocabType: .raw, maxWhitespaceCount: limit)
        var rng = Xoshiro256StarStar(seed: seed)
        var logitsRng = Xoshiro256StarStar(seed: seed &+ 99)
        let config = SamplingConfiguration(temperature: temperature)
        var tokens: [Int32] = []
        for _ in 0..<steps {
            var logits = (0..<Self.vocabulary.count).map { id -> Float in
                let base = Float(Double(logitsRng.next() % 1000) / 250.0)
                return Self.whitespaceTokens.contains(Int32(id)) ? base + whitespaceBoost : base
            }
            guard session.applyMask(to: &logits) else { break }
            let token = CompositeSampler.sample(from: &logits, config: config, using: &rng)
            guard session.acceptToken(token) else { break }
            tokens.append(token)
        }
        return tokens
    }

    @Test("Sampled generations are identical with and without the default limit")
    func identicalWithTheDefaultLimit() throws {
        for seed in 0..<300 as Range<UInt64> {
            let unlimited = try generate(seed: seed, limit: nil, temperature: 0.8, whitespaceBoost: 0)
            let limited = try generate(
                seed: seed, limit: ConstrainedGenerationSession.defaultMaxWhitespaceCount, temperature: 0.8,
                whitespaceBoost: 0)
            #expect(unlimited == limited, "seed \(seed) differs under the default limit")
        }
    }

    @Test("A limit changes a sampled generation only where the unlimited one overran it")
    func divergenceNeedsAnOverrun() throws {
        // Whitespace is made likely, so that runs longer than the limit do happen.
        for limit in [3, 8, 32] {
            var diverged = 0
            for seed in 0..<300 as Range<UInt64> {
                let unlimited = try generate(seed: seed, limit: nil, temperature: 0.8, whitespaceBoost: 2.5)
                let limited = try generate(seed: seed, limit: limit, temperature: 0.8, whitespaceBoost: 2.5)
                guard
                    let index = zip(unlimited, limited).enumerated()
                        .first(where: { $0.element.0 != $0.element.1 })?.offset
                else { continue }
                diverged += 1
                // At the first difference the unlimited run wrote whitespace after `limit` whitespace tokens.
                let before = unlimited[max(0, index - limit)..<index]
                #expect(
                    before.count == limit && before.allSatisfy { Self.whitespaceTokens.contains($0) },
                    "limit \(limit), seed \(seed): diverged at \(index) without a whitespace run of \(limit)")
                #expect(Self.whitespaceTokens.contains(unlimited[index]))
            }
            #expect(diverged > 0, "limit \(limit) was never reached, so the test did not exercise it")
        }
    }
}
