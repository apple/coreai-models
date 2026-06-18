// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("SamplingConfiguration", .serialized)
struct SamplingConfigurationTests {
    // MARK: - Basic Configuration Tests

    @Test("Greedy preset has zero temperature and combined enabled")
    func greedyPreset() {
        let config = SamplingConfiguration.greedy

        #expect(config.temperature == 0)
        #expect(config.combined == true)
        #expect(config.topK == nil)
        #expect(config.topP == nil)
        #expect(config.isGreedy == true)
    }

    @Test("Custom temperature with default combined flag")
    func customTemperature() {
        let config = SamplingConfiguration(temperature: 0.7)

        #expect(config.temperature == 0.7)
        #expect(config.combined == true)
        #expect(config.topK == nil)
        #expect(config.topP == nil)
    }

    @Test("Combined flag can be explicitly disabled")
    func combinedDisabled() {
        let config = SamplingConfiguration(temperature: 0.5, combined: false)

        #expect(config.combined == false)
    }

    // MARK: - TopK and TopP Configuration Tests

    @Test("TopK configuration")
    func topKConfig() {
        let config = SamplingConfiguration(temperature: 0.8, topK: 50)

        #expect(config.temperature == 0.8)
        #expect(config.topK == 50)
        #expect(config.topP == nil)
        #expect(config.isComposite == true)
    }

    @Test("TopP configuration")
    func topPConfig() {
        let config = SamplingConfiguration(temperature: 0.9, topP: 0.95)

        #expect(config.temperature == 0.9)
        #expect(config.topK == nil)
        #expect(config.topP == 0.95)
        #expect(config.isComposite == true)
    }

    @Test("Combined TopK and TopP configuration")
    func combinedTopKTopP() {
        let config = SamplingConfiguration(temperature: 0.8, topK: 50, topP: 0.9)

        #expect(config.topK == 50)
        #expect(config.topP == 0.9)
        #expect(config.isComposite == true)
    }

    @Test("Temperature factory method")
    func temperatureFactory() {
        let config = SamplingConfiguration.temperature(0.7)

        #expect(config.temperature == 0.7)
        #expect(config.topK == nil)
        #expect(config.topP == nil)
    }

    // MARK: - Validation Tests

    @Test("Validation warns for topK=1 with temperature")
    func validationTopK1Warning() {
        let config = SamplingConfiguration(temperature: 0.8, topK: 1)
        config.validateAndWarn()  // Should log warning

        // TopK=1 with temp>0 should trigger warning
        #expect(config.topK == 1)
    }

    @Test("Validation warns for topP=1.0")
    func validationTopP1Warning() {
        let config = SamplingConfiguration(temperature: 0.8, topP: 1.0)
        config.validateAndWarn()  // Should log warning

        // topP=1.0 is effectively disabled
        #expect(config.topP == 1.0)
    }

    @Test("Normalized config removes redundant topP=1.0")
    func normalizedRemovesTopP1() {
        let config = SamplingConfiguration(temperature: 0.8, topP: 1.0)
        let normalized = config.normalized()

        #expect(normalized.topP == nil)
        #expect(normalized.temperature == 0.8)
    }

    @Test("Normalized config removes topK/topP for greedy")
    func normalizedRemovesForGreedy() {
        let config = SamplingConfiguration(temperature: 0, topK: 50, topP: 0.9)
        let normalized = config.normalized()

        #expect(normalized.topK == nil)
        #expect(normalized.topP == nil)
        #expect(normalized.temperature == 0)
    }

    // MARK: - Sampler Tests

    @Test("Fallback sampler selects greedy for zero temperature")
    func fallbackSamplerGreedy() {
        let config = SamplingConfiguration(temperature: 0)
        var logits: [LogitsScalarType] = [0.1, 0.5, 0.2, 0.9, 0.3]

        let token = config.fallbackSampler(from: &logits)

        #expect(token == 3)  // Index of max value 0.9
    }

    @Test("Fallback sampler returns valid index for temperature sampling")
    func fallbackSamplerTemperature() {
        let config = SamplingConfiguration(temperature: 1.0)
        var logits: [LogitsScalarType] = [0.1, 0.5, 0.2, 0.9, 0.3]

        let token = config.fallbackSampler(from: &logits)

        #expect(token >= 0 && token < 5)
    }

    @Test("TopK sampling restricts to top K tokens")
    func topKSamplingRestriction() {
        let config = SamplingConfiguration(temperature: 1.0, topK: 2)
        let logits: [Float16] = [0.1, 0.5, 0.2, 0.9, 0.3]

        // Run multiple times to verify we only get top 2 tokens (indices 3 and 1)
        var sampledTokens = Set<Int32>()
        for _ in 0..<100 {
            var logitsCopy = logits
            let token = config.fallbackSampler(from: &logitsCopy)
            sampledTokens.insert(token)
        }

        // Should only sample from top 2: index 3 (0.9) and index 1 (0.5)
        #expect(sampledTokens.isSubset(of: [1, 3]))
    }

    @Test("TopP sampling excludes low probability tokens")
    func topPSamplingExclusion() {
        // Logits: [10.0, 2.0, 2.0, 2.0]
        // Softmax will be extremely peaked at index 0 (> 0.99)
        // TopP 0.9 should strictly exclude indices 1, 2, 3
        let config = SamplingConfiguration(temperature: 1.0, topP: 0.9)
        let logits: [Float16] = [10.0, 2.0, 2.0, 2.0]

        var sampledTokens = Set<Int32>()
        for _ in 0..<50 {
            var logitsCopy = logits
            let token = config.fallbackSampler(from: &logitsCopy)
            sampledTokens.insert(token)
        }

        // Should only sample index 0
        #expect(sampledTokens == [0])
    }

    @Test("TopP sampling includes multiple tokens within threshold")
    func topPSamplingInclusion() {
        // Logits: [2.0, 2.0, -10.0]
        // Softmax approx: [0.5, 0.5, 0.0]
        // TopP 0.9 should include indices 0 and 1, exclude 2
        let config = SamplingConfiguration(temperature: 1.0, topP: 0.9)
        let logits: [Float16] = [2.0, 2.0, -10.0]

        var sampledTokens = Set<Int32>()
        // We can't pass a stable random seed, however 50 iterations means there is
        // 1/1,000,000,000,000,000 chance this is going to fail (0.5^50)
        for _ in 0..<50 {
            var logitsCopy = logits
            let token = config.fallbackSampler(from: &logitsCopy)
            sampledTokens.insert(token)
        }

        // Should sample both 0 and 1, but never 2
        #expect(sampledTokens.contains(0))
        #expect(sampledTokens.contains(1))
        #expect(!sampledTokens.contains(2))
    }

    // MARK: - MinP Configuration Tests

    @Test("MinP configuration")
    func minPConfig() {
        let config = SamplingConfiguration(temperature: 0.9, minP: 0.05)

        #expect(config.temperature == 0.9)
        #expect(config.minP == 0.05)
        #expect(config.isComposite == true)
    }

    @Test("MinP=nil means isComposite is false (temp only)")
    func minPNilNotComposite() {
        let config = SamplingConfiguration(temperature: 0.9)

        #expect(config.minP == nil)
        #expect(config.isComposite == false)
    }

    @Test("Normalized config removes minP for greedy")
    func normalizedRemovesMinPForGreedy() {
        let config = SamplingConfiguration(temperature: 0, minP: 0.1)
        let normalized = config.normalized()

        #expect(normalized.minP == nil)
    }

    @Test("MinP sampling excludes low relative probability tokens")
    func minPSamplingExclusion() {
        // Logits: [10.0, 9.5, 2.0]
        // After temp=1 softmax: token 0 ≈ 0.62, token 1 ≈ 0.38, token 2 ≈ 0.0003
        // minP=0.1 threshold: 0.1 * 0.62 = 0.062
        // Token 2 (0.0003) < 0.062 → filtered out
        let config = SamplingConfiguration(temperature: 1.0, minP: 0.1)
        let logits: [Float16] = [10.0, 9.5, 2.0]

        var sampledTokens = Set<Int32>()
        for _ in 0..<100 {
            var logitsCopy = logits
            let token = config.fallbackSampler(from: &logitsCopy)
            sampledTokens.insert(token)
        }

        // Token 2 should never be sampled
        #expect(!sampledTokens.contains(2), "minP should filter out token 2")
        // Tokens 0 and 1 should both appear
        #expect(sampledTokens.contains(0))
        #expect(sampledTokens.contains(1))
    }
}
