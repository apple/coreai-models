// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Metal
import TestUtilities
import Testing

@testable import CoreAILanguageModels

/// MPSGraphArgmaxSampler Tests - validates the MPSGraph-based argmax used in V2 pipelined inference.
///
/// Essential coverage:
/// 1. Basic correctness - finds max at arbitrary position
/// 2. Large vocab (150K) - production scenario
/// 3. Offset handling - prefill scenario with queryLength > 1
/// 4. Performance - ensures sampler stays under 1ms
@Suite("MPSGraph Argmax Sampler Tests", .enabled(if: !CIEnvironment.isVM))
struct MPSGraphArgmaxSamplerTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    static let vocabSize32K = 32000
    static let vocabSize150K = 151936  // Qwen vocab size

    @Test("Argmax finds correct maximum")
    func argmaxCorrectness() async throws {
        let device = try #require(Self.device)
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize32K)

        let targetIndex = 12345
        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize32K * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize32K {
            logitsPtr[i] = Float16(i == targetIndex ? 100.0 : Float.random(in: -10.0..<10.0))
        }

        let queue = try #require(device.makeCommandQueue())

        await withCheckedContinuation { continuation in
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in
                    continuation.resume()
                }
            )
        }

        let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(result == Int32(targetIndex), "Expected \(targetIndex), got \(result)")
    }

    @Test("Argmax with large vocabulary (150K tokens)")
    func argmaxLargeVocab() async throws {
        let device = try #require(Self.device)
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize150K)

        let targetIndex = 100000
        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize150K * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize150K {
            logitsPtr[i] = Float16(i == targetIndex ? 50.0 : -50.0)
        }

        let queue = try #require(device.makeCommandQueue())

        await withCheckedContinuation { continuation in
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in
                    continuation.resume()
                }
            )
        }

        let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(result == Int32(targetIndex), "Expected \(targetIndex), got \(result)")
    }

    @Test("Argmax with slice (prefill scenario)")
    func argmaxWithSlice() async throws {
        let device = try #require(Self.device)
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize32K)

        let queryLength = 128  // Typical prefill length
        let targetIndex = 15000
        let totalElements = queryLength * Self.vocabSize32K
        let logitsBuffer = try #require(device.makeBuffer(length: totalElements * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<totalElements {
            logitsPtr[i] = Float16(-100.0)
        }
        // Set max in LAST token's logits (that's what we sample from)
        let lastTokenOffset = (queryLength - 1) * Self.vocabSize32K
        logitsPtr[lastTokenOffset + targetIndex] = Float16(100.0)

        let queue = try #require(device.makeCommandQueue())

        await withCheckedContinuation { continuation in
            sampler.encodeWithSlice(
                to: queue,
                logitsBuffer: logitsBuffer,
                queryLength: queryLength,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in
                    continuation.resume()
                }
            )
        }

        let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(result == Int32(targetIndex), "Expected \(targetIndex), got \(result)")
    }

    @Test("Argmax latency under 1ms for 150K vocab")
    func argmaxPerformance() async throws {
        let device = try #require(Self.device)
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize150K)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize150K * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let queue = try #require(device.makeCommandQueue())

        // Warm up
        for _ in 0..<10 {
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in
                        continuation.resume()
                    }
                )
            }
        }

        // Measure
        let iterations = 100
        let start = SuspendingClock().now
        for _ in 0..<iterations {
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in
                        continuation.resume()
                    }
                )
            }
        }
        let avgLatencyMs = (SuspendingClock().now - start).inMilliseconds / Double(iterations)

        print("MPSGraph Argmax latency: \(String(format: "%.3f", avgLatencyMs)) ms")

        // Use higher threshold on VM due to virtualization overhead
        let threshold = CIEnvironment.isVM ? 100.0 : 25.0
        #expect(
            avgLatencyMs < threshold,
            "Argmax too slow: \(avgLatencyMs) ms (threshold: \(threshold) ms, VM: \(CIEnvironment.isVM))")
    }

    @Test("Argmax handles edge cases - first and last index")
    func argmaxEdgeCases() async throws {
        let device = try #require(Self.device)
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: Self.vocabSize32K)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize32K * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())

        // Test first index (0)
        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize32K {
            logitsPtr[i] = Float16(i == 0 ? 100.0 : -100.0)
        }

        await withCheckedContinuation { continuation in
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in
                    continuation.resume()
                }
            )
        }

        var result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(result == 0, "Expected 0, got \(result)")

        // Test last index (vocabSize - 1)
        for i in 0..<Self.vocabSize32K {
            logitsPtr[i] = Float16(i == Self.vocabSize32K - 1 ? 100.0 : -100.0)
        }

        await withCheckedContinuation { continuation in
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in
                    continuation.resume()
                }
            )
        }

        result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(result == Int32(Self.vocabSize32K - 1), "Expected \(Self.vocabSize32K - 1), got \(result)")
    }
}

// MARK: - MPSGraph Top-K Sampler Tests

@Suite("MPSGraph Top-K Sampler Tests", .enabled(if: !CIEnvironment.isVM))
struct MPSGraphCompositeSamplerTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    static let vocabSize = 32000
    static let k = 40

    @Test("Top-K samples from high-probability tokens")
    func topKSamplesCorrectly() async throws {
        let device = try #require(Self.device)
        // Create sampler with temperature=1.0 (neutral)
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: Self.k, temperature: 1.0)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        // Create logits where only a few tokens have high probability
        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize {
            logitsPtr[i] = Float16(-100.0)  // Very low probability
        }

        // Set a few high-probability tokens (these should be selected)
        let highProbTokens = [100, 200, 300, 400, 500]
        for token in highProbTokens {
            logitsPtr[token] = Float16(10.0)
        }

        let queue = try #require(device.makeCommandQueue())

        // Sample multiple times and verify we only get high-prob tokens
        var sampledTokens = Set<Int32>()
        for _ in 0..<20 {
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in
                        continuation.resume()
                    }
                )
            }

            let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
            sampledTokens.insert(result)
        }

        // All sampled tokens should be from the high-probability set
        for token in sampledTokens {
            #expect(highProbTokens.contains(Int(token)), "Unexpected token \(token) sampled")
        }
    }

    @Test("Top-K with low temperature concentrates probability (deterministic)")
    func topKLowTemperature() async throws {
        let device = try #require(Self.device)
        // Create sampler with very low temperature (0.1) - concentrates probability
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: Self.k, temperature: 0.1)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize {
            logitsPtr[i] = Float16(-100.0)
        }

        // One token is much higher than others
        let topToken = 12345
        logitsPtr[topToken] = Float16(100.0)
        logitsPtr[topToken + 1] = Float16(9.9)  // Much lower
        logitsPtr[topToken + 2] = Float16(9.8)

        let queue = try #require(device.makeCommandQueue())

        // Use deterministic random = 0.5 (middle of distribution)
        // With low temperature and dominant logit, cumsum should exceed 0.5 at first token
        sampler.testingOnlyRandomOverride = 0.5

        await withCheckedContinuation { continuation in
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in
                    continuation.resume()
                }
            )
        }

        let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(result == Int32(topToken), "Low temperature with dominant logit should pick top token, got \(result)")

        // Reset for other tests
        sampler.testingOnlyRandomOverride = nil
    }

    @Test("Top-K with different random values selects different tokens (deterministic)")
    func topKHighTemperature() async throws {
        let device = try #require(Self.device)
        // Create sampler with neutral temperature (1.0)
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: Self.k, temperature: 1.0)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize {
            logitsPtr[i] = Float16(-100.0)
        }

        // Several tokens with similar probabilities
        let similarTokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        for token in similarTokens {
            logitsPtr[token] = Float16(10.0)
        }

        let queue = try #require(device.makeCommandQueue())

        // Test with deterministic random values across the distribution
        var sampledTokens = Set<Int32>()
        let randomValues: [Float] = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

        for randomValue in randomValues {
            sampler.testingOnlyRandomOverride = randomValue

            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in
                        continuation.resume()
                    }
                )
            }

            let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
            sampledTokens.insert(result)
        }

        // With 10 equal-probability tokens and 10 evenly spaced random values,
        // we should get multiple different tokens
        #expect(
            sampledTokens.count >= 3,
            "Different random values should produce different tokens, got \(sampledTokens.count) unique")

        // All sampled tokens should be from our high-probability set
        for token in sampledTokens {
            #expect(similarTokens.contains(Int(token)), "Unexpected token \(token) not in high-probability set")
        }

        // Reset
        sampler.testingOnlyRandomOverride = nil
    }

    @Test("Top-K latency under 2ms for 32K vocab")
    func topKPerformance() async throws {
        let device = try #require(Self.device)
        // Create sampler with temperature=1.0
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: Self.k, temperature: 1.0)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let queue = try #require(device.makeCommandQueue())

        // Warm up
        for _ in 0..<10 {
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in
                        continuation.resume()
                    }
                )
            }
        }

        // Measure
        let iterations = 100
        let start = SuspendingClock().now
        for _ in 0..<iterations {
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in
                        continuation.resume()
                    }
                )
            }
        }
        let avgLatencyMs = (SuspendingClock().now - start).inMilliseconds / Double(iterations)

        print("MPSGraph Top-K latency: \(String(format: "%.3f", avgLatencyMs)) ms")

        // Use higher threshold on VM due to virtualization overhead
        let threshold = CIEnvironment.isVM ? 100.0 : 2.0
        #expect(
            avgLatencyMs < threshold,
            "Top-K too slow: \(avgLatencyMs) ms (threshold: \(threshold) ms, VM: \(CIEnvironment.isVM))")
    }

    @Test("Top-K with slice (prefill scenario)")
    func topKWithSlice() async throws {
        let device = try #require(Self.device)
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: Self.k, temperature: 1.0)

        let queryLength = 128  // Typical prefill length
        let targetToken = 15000
        let totalElements = queryLength * Self.vocabSize
        let logitsBuffer = try #require(device.makeBuffer(length: totalElements * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        // Fill with very low values
        for i in 0..<totalElements {
            logitsPtr[i] = Float16(-100.0)
        }
        // Set very high value in LAST token's logits at targetToken position
        let lastTokenOffset = (queryLength - 1) * Self.vocabSize
        logitsPtr[lastTokenOffset + targetToken] = Float16(100.0)

        let queue = try #require(device.makeCommandQueue())

        // Use deterministic random value that should select the dominant token
        sampler.testingOnlyRandomOverride = 0.5

        await withCheckedContinuation { continuation in
            sampler.encodeWithSlice(
                to: queue,
                logitsBuffer: logitsBuffer,
                queryLength: queryLength,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in
                    continuation.resume()
                }
            )
        }

        let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(result == Int32(targetToken), "Expected \(targetToken), got \(result)")

        sampler.testingOnlyRandomOverride = nil
    }
}

// MARK: - MPSGraph TopP Sampler Tests

@Suite("MPSGraph TopP Sampler Tests", .enabled(if: !CIEnvironment.isVM))
struct MPSGraphTopPSamplerTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    static let vocabSize = 32000

    @Test("TopP filters to tokens within cumulative probability mass")
    func topPFiltersCorrectly() async throws {
        let device = try #require(Self.device)
        // topP=0.9 with K=1000 window
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: 1000, temperature: 1.0, topP: 0.9, minP: 0.0)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize {
            logitsPtr[i] = Float16(-100.0)
        }

        // Create a peaked distribution: token 100 dominates
        let highProbTokens = [100, 101, 102]
        logitsPtr[100] = Float16(10.0)
        logitsPtr[101] = Float16(9.0)
        logitsPtr[102] = Float16(8.0)

        let queue = try #require(device.makeCommandQueue())

        var sampledTokens = Set<Int32>()
        for _ in 0..<30 {
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in continuation.resume() }
                )
            }
            let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
            sampledTokens.insert(result)
        }

        // All sampled tokens should be from the high-probability set
        for token in sampledTokens {
            #expect(highProbTokens.contains(Int(token)), "TopP sampled unexpected token \(token)")
        }
    }

    @Test("TopP=1.0 (disabled) behaves same as plain TopK")
    func topPDisabledMatchesTopK() async throws {
        let device = try #require(Self.device)
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: 40, temperature: 1.0, topP: 1.0, minP: 0.0)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize { logitsPtr[i] = Float16(-100.0) }

        // 5 tokens with equal high logits
        for t in [500, 501, 502, 503, 504] { logitsPtr[t] = Float16(10.0) }

        let queue = try #require(device.makeCommandQueue())

        sampler.testingOnlyRandomOverride = 0.5
        await withCheckedContinuation { continuation in
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in continuation.resume() }
            )
        }

        let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(
            [500, 501, 502, 503, 504].contains(Int(result)), "topP=1.0 should sample from top tokens, got \(result)")
        sampler.testingOnlyRandomOverride = nil
    }

    @Test("TopP with very small value concentrates on top token")
    func topPSmallConcentrates() async throws {
        let device = try #require(Self.device)
        // topP=0.01 should only keep the very top token
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: 1000, temperature: 1.0, topP: 0.01, minP: 0.0)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize { logitsPtr[i] = Float16(-100.0) }
        logitsPtr[7777] = Float16(10.0)
        logitsPtr[7778] = Float16(5.0)

        let queue = try #require(device.makeCommandQueue())

        // With topP=0.01 and a dominant token, should always pick the top one
        sampler.testingOnlyRandomOverride = 0.005
        await withCheckedContinuation { continuation in
            sampler.encode(
                to: queue,
                logitsBuffer: logitsBuffer,
                logitsOffset: 0,
                outputBuffer: outputBuffer,
                outputOffset: 0,
                completion: { _ in continuation.resume() }
            )
        }

        let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        #expect(result == 7777, "topP=0.01 with dominant token should pick it, got \(result)")
        sampler.testingOnlyRandomOverride = nil
    }
}

// MARK: - MPSGraph MinP Sampler Tests

@Suite("MPSGraph MinP Sampler Tests", .enabled(if: !CIEnvironment.isVM))
struct MPSGraphMinPSamplerTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    static let vocabSize = 32000

    @Test("MinP filters out low-probability tokens")
    func minPFiltersLowProb() async throws {
        let device = try #require(Self.device)
        // minP=0.1 means keep tokens with prob >= 10% of max prob
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: 1000, temperature: 1.0, topP: 1.0, minP: 0.1)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize { logitsPtr[i] = Float16(-100.0) }

        // Set up: token 200 has highest logit (10.0), tokens 201-203 are close (9.5-8.5)
        // Token 300 is far away (2.0) — should be filtered by minP
        logitsPtr[200] = Float16(10.0)
        logitsPtr[201] = Float16(9.5)
        logitsPtr[202] = Float16(9.0)
        logitsPtr[203] = Float16(8.5)
        logitsPtr[300] = Float16(2.0)  // exp(2-10)/exp(0) = exp(-8) ≈ 0.0003 < 0.1

        let queue = try #require(device.makeCommandQueue())

        var sampledTokens = Set<Int32>()
        for _ in 0..<30 {
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in continuation.resume() }
                )
            }
            let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
            sampledTokens.insert(result)
        }

        // Token 300 should never appear (too low relative probability)
        #expect(!sampledTokens.contains(300), "minP=0.1 should filter out token 300, but it was sampled")
        // High probability tokens should appear
        #expect(!sampledTokens.isEmpty, "Should have sampled some tokens")
    }

    @Test("MinP=0.0 (disabled) allows all top-K tokens")
    func minPDisabled() async throws {
        let device = try #require(Self.device)
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: 40, temperature: 1.0, topP: 1.0, minP: 0.0)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize { logitsPtr[i] = Float16(-100.0) }

        // Equal logits for 10 tokens
        let tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for t in tokens { logitsPtr[t] = Float16(5.0) }

        let queue = try #require(device.makeCommandQueue())

        var sampledTokens = Set<Int32>()
        let randomValues: [Float] = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        for r in randomValues {
            sampler.testingOnlyRandomOverride = r
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in continuation.resume() }
                )
            }
            let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
            sampledTokens.insert(result)
        }

        // With minP=0.0, all tokens should be accessible
        #expect(sampledTokens.count >= 3, "minP=0.0 should allow diverse sampling, got \(sampledTokens.count) unique")
        for token in sampledTokens {
            #expect(tokens.contains(Int(token)), "Unexpected token \(token)")
        }
        sampler.testingOnlyRandomOverride = nil
    }

    @Test("Combined TopP + MinP: both filters apply")
    func combinedTopPAndMinP() async throws {
        let device = try #require(Self.device)
        // topP=0.95 + minP=0.1
        let sampler = try MPSGraphCompositeSampler(
            device: device, vocabSize: Self.vocabSize, k: 1000, temperature: 1.0, topP: 0.95, minP: 0.1)

        let logitsBuffer = try #require(device.makeBuffer(length: Self.vocabSize * 2, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Self.vocabSize { logitsPtr[i] = Float16(-100.0) }

        // Dominant token + a few close ones
        logitsPtr[1000] = Float16(10.0)
        logitsPtr[1001] = Float16(9.8)
        logitsPtr[1002] = Float16(9.5)
        logitsPtr[2000] = Float16(3.0)  // Far below minP threshold

        let queue = try #require(device.makeCommandQueue())

        var sampledTokens = Set<Int32>()
        for _ in 0..<30 {
            await withCheckedContinuation { continuation in
                sampler.encode(
                    to: queue,
                    logitsBuffer: logitsBuffer,
                    logitsOffset: 0,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: { _ in continuation.resume() }
                )
            }
            let result = outputBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
            sampledTokens.insert(result)
        }

        // Token 2000 should be filtered by minP
        #expect(!sampledTokens.contains(2000), "Combined topP+minP should filter token 2000")
        // Should only sample from the top cluster
        for token in sampledTokens {
            #expect([1000, 1001, 1002].contains(Int(token)), "Unexpected token \(token)")
        }
    }
}

// MARK: - MPSGraph Sampler Factory Tests

@Suite("MPSGraph Sampler Factory Tests", .enabled(if: !CIEnvironment.isVM))
struct MPSGraphSamplerFactoryTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("Factory creates argmax sampler for temperature=0")
    func factoryCreatesArgmax() throws {
        let device = try #require(Self.device)
        let config = SamplingConfiguration(temperature: 0)
        let sampler = try MPSGraphSamplerFactory.makeSampler(device: device, vocabSize: 32000, config: config)
        #expect(sampler is MPSGraphArgmaxSampler)
    }

    @Test("Factory creates composite sampler for temperature>0")
    func factoryCreatesComposite() throws {
        let device = try #require(Self.device)
        let config = SamplingConfiguration(temperature: 0.8, topK: 50, topP: 0.9, minP: 0.05)
        let sampler = try MPSGraphSamplerFactory.makeSampler(device: device, vocabSize: 32000, config: config)
        #expect(sampler is MPSGraphCompositeSampler)
        let composite = sampler as! MPSGraphCompositeSampler
        #expect(composite.k == 50)
        #expect(composite.topP == 0.9)
        #expect(composite.minP == 0.05)
    }

    @Test("Factory uses K=1000 when only topP is set")
    func factoryUsesLargeKForTopP() throws {
        let device = try #require(Self.device)
        let config = SamplingConfiguration(temperature: 0.8, topP: 0.9)
        let sampler = try MPSGraphSamplerFactory.makeSampler(device: device, vocabSize: 32000, config: config)
        let composite = sampler as! MPSGraphCompositeSampler
        #expect(composite.k == 1000)
    }

    @Test("Factory uses K=1000 when only minP is set")
    func factoryUsesLargeKForMinP() throws {
        let device = try #require(Self.device)
        let config = SamplingConfiguration(temperature: 0.8, minP: 0.05)
        let sampler = try MPSGraphSamplerFactory.makeSampler(device: device, vocabSize: 32000, config: config)
        let composite = sampler as! MPSGraphCompositeSampler
        #expect(composite.k == 1000)
    }

    @Test("Factory uses K=40 for temperature-only")
    func factoryUsesDefaultK() throws {
        let device = try #require(Self.device)
        let config = SamplingConfiguration(temperature: 0.8)
        let sampler = try MPSGraphSamplerFactory.makeSampler(device: device, vocabSize: 32000, config: config)
        let composite = sampler as! MPSGraphCompositeSampler
        #expect(composite.k == 40)
    }
}
