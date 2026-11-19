// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import TestUtilities
import Testing

@testable import CoreAILanguageModels

@Suite("Sampler backend selection")
struct SamplerBackendTests {
    @Test("The default backend resolves to the reference sampler")
    func defaultResolvesToReference() {
        let config = SamplingConfiguration(temperature: 0.8, topK: 40)
        #expect(config.backend == .default)
        #expect(config.resolvedBackend == .reference)
    }

    @Test("An explicit backend is used for unseeded requests")
    func explicitBackendIsUsed() {
        #expect(SamplingConfiguration(temperature: 0.8, backend: .reference).resolvedBackend == .reference)
        #expect(SamplingConfiguration(temperature: 0.8, backend: .bnns).resolvedBackend == .bnns)
    }

    @Test("Seeded requests always use the reference sampler")
    func seededUsesReference() {
        #expect(SamplingConfiguration(temperature: 0.8, seed: 1).resolvedBackend == .reference)
        #expect(SamplingConfiguration(temperature: 0.8, seed: 1, backend: .reference).resolvedBackend == .reference)
    }

    @Test("The bnns backend with a seed is rejected")
    func bnnsWithSeedIsRejected() async {
        let error = SamplingConfiguration.parameterError(0.8, seed: 1, backend: .bnns)
        #expect(error?.parameter == .backend)
        #expect(SamplingConfiguration.parameterError(0.8, seed: 1, backend: .reference) == nil)
        await #expect(processExitsWith: .failure) {
            _ = SamplingConfiguration(temperature: 0.8, seed: 1, backend: .bnns)
        }
    }

    @Test("normalized() keeps the backend")
    func normalizedKeepsBackend() {
        let config = SamplingConfiguration(temperature: 0.8, topP: 1.0, backend: .bnns)
        #expect(config.normalized().backend == .bnns)
        #expect(config.normalized().topP == nil)
    }

    @Test("Configurations that differ only in backend are not equal")
    func backendIsPartOfEquality() {
        #expect(SamplingConfiguration(temperature: 0.8) != SamplingConfiguration(temperature: 0.8, backend: .bnns))
    }

    @Test(
        "Seeded sampling is unchanged by the backend", arguments: [SamplingConfiguration.Backend.default, .reference])
    func seededSamplingIgnoresBackend(backend: SamplingConfiguration.Backend) {
        var rng = Xoshiro256StarStar(seed: 4)
        let row: [LogitsScalarType] = (0..<4096).map { _ in LogitsScalarType(Float.random(in: -4..<12, using: &rng)) }
        let config = SamplingConfiguration(temperature: 0.8, topK: 50, topP: 0.9, seed: 99, backend: backend)
        let plain = SamplingConfiguration(temperature: 0.8, topK: 50, topP: 0.9, seed: 99)
        for step in 0..<32 {
            var a = row
            var b = row
            #expect(config.sampleToken(from: &a, step: step) == plain.sampleToken(from: &b, step: step))
        }
    }

    @Test("The bnns backend serves greedy requests with the reference token")
    func bnnsGreedyMatchesReference() {
        var rng = Xoshiro256StarStar(seed: 6)
        for _ in 0..<10 {
            let row: [LogitsScalarType] = (0..<32_000).map { _ in
                LogitsScalarType(Float.random(in: -8..<16, using: &rng))
            }
            var a = row
            var b = row
            #expect(
                SamplingConfiguration(temperature: 0, backend: .bnns).cpuSampler(from: &a)
                    == SamplingConfiguration.greedy.cpuSampler(from: &b))
        }
    }

    @Test(
        "The pipelined engine rejects an explicit CPU backend",
        arguments: [SamplingConfiguration.Backend.reference, .bnns])
    func pipelinedRejectsCPUBackend(backend: SamplingConfiguration.Backend) {
        #expect(throws: InferenceRuntimeError.self) {
            try CoreAIPipelinedEngine.rejectCPUBackend(SamplingConfiguration(temperature: 0.8, backend: backend))
        }
    }

    @Test("The pipelined engine accepts the default backend")
    func pipelinedAcceptsDefault() throws {
        try CoreAIPipelinedEngine.rejectCPUBackend(SamplingConfiguration(temperature: 0.8))
    }

    @Test("SamplerUsage counts the implementation that sampled each token")
    func usageCountsImplementations() {
        // Other suites sample concurrently, so check lower bounds on the deltas.
        var rng = Xoshiro256StarStar(seed: 11)
        let row: [LogitsScalarType] = (0..<4096).map { _ in LogitsScalarType(Float.random(in: -4..<12, using: &rng)) }
        var nanRow = row
        nanRow[7] = .nan
        let bnns = SamplingConfiguration(temperature: 0.8, topK: 40, backend: .bnns)
        let before = SamplerUsage.snapshot()
        for _ in 0..<5 {
            var a = row
            _ = bnns.sampleToken(from: &a, step: nil)
        }
        var b = nanRow
        _ = bnns.sampleToken(from: &b, step: nil)
        var c = row
        _ = SamplingConfiguration(temperature: 0.8, backend: .reference).sampleToken(from: &c, step: nil)
        let after = SamplerUsage.snapshot()
        #expect(after.bnns - before.bnns >= 5)
        #expect(after.bnnsFallbacks - before.bnnsFallbacks >= 1)
        #expect(after.reference - before.reference >= 2)
    }

    @Test("SamplerUsage summary reports shares and fallbacks")
    func usageSummary() {
        #expect(
            SamplerUsage.Counts(reference: 25, bnns: 75, bnnsFallbacks: 25).summary
                == "Sampler: bnns 75.0% (75), reference 25.0% (25; 25 BNNS fallbacks)")
        #expect(SamplerUsage.Counts(reference: 10).summary == "Sampler: reference 100.0% (10)")
        #expect(SamplerUsage.Counts().summary == "Sampler: no CPU-sampled tokens")
    }
}
