// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Sliding-window ring and batch alignment

@Suite("Sliding-window ring rewind with batch-aligned restarts")
struct AlignedRingRewindTests {
    // Gemma 4: window 512, ring 576, 8-wide decode batch.
    let ring = SlidingRing(depth: 576, window: 512)

    @Test("A rewind whose batch-aligned restart needs overwritten keys is refused")
    func alignedRestartIsCounted() {
        // Processed 1000, target 935: 65 positions back is allowed by the unaligned rule, but the
        // next forward restarts at 928 and its first query needs key 928 - 511 = 417, while the
        // oldest key still resident is 1000 - 576 = 424.
        #expect(ring.allowsRewind(processed: 1000, to: 935))
        #expect(!ring.allowsRewind(processed: 1000, to: 935, alignment: 8))
        // A target on a batch boundary restarts there, so 64 back is fine.
        #expect(ring.allowsRewind(processed: 1000, to: 936, alignment: 8))
    }

    @Test("Whenever the aligned rule allows a rewind, every key the restart needs is resident")
    func allowedRewindsKeepNeededKeys() {
        for alignment in [1, 4, 8, 64] {
            for processed in [600, 1000, 4097] {
                for target in 1...processed {
                    guard ring.allowsRewind(processed: processed, to: target, alignment: alignment) else {
                        continue
                    }
                    let restart = target / alignment * alignment
                    let oldestNeeded = max(0, restart - ring.window + 1)
                    let oldestResident = max(0, processed - ring.depth)
                    #expect(
                        oldestNeeded >= oldestResident,
                        "processed \(processed) target \(target) alignment \(alignment)")
                }
            }
        }
    }

    @Test("A verify block of the limit rolls back to its anchor from any position")
    func verifyLimitIsSafeAtEveryPosition() {
        for alignment in [1, 8] {
            let limit = ring.maxVerifyBlockLength(alignment: alignment)
            for base in 600..<(600 + 4 * 64) {
                // Anchor at `base`; the block holds `limit` tokens and all drafts are rejected.
                #expect(
                    ring.allowsRewind(processed: base + limit, to: base + 1, alignment: alignment),
                    "base \(base) limit \(limit) alignment \(alignment)")
            }
            // One more token fails at some position.
            let failing = (600..<(600 + 4 * 64)).contains {
                !ring.allowsRewind(processed: $0 + limit + 1, to: $0 + 1, alignment: alignment)
            }
            #expect(failing, "limit \(limit) is not tight for alignment \(alignment)")
        }
        #expect(ring.maxVerifyBlockLength(alignment: 8) == 59)
    }
}

// MARK: - Reset validation

@Suite("Static-shape reset validation")
struct ResetValidationTests {
    @Test("A target index in [0, processed] is accepted")
    func inRange() throws {
        try StaticShapeEngine.validateResetTarget(0, processed: 0)
        try StaticShapeEngine.validateResetTarget(0, processed: 10)
        try StaticShapeEngine.validateResetTarget(10, processed: 10)
    }

    @Test("A negative index or one past the processed count throws an invalid-argument error")
    func outOfRange() {
        for (target, processed) in [(-1, 10), (11, 10), (1, 0), (Int.min, 5)] {
            #expect(throws: InferenceRuntimeError.self) {
                try StaticShapeEngine.validateResetTarget(target, processed: processed)
            }
        }
    }
}

@Suite("In-memory engine reset")
struct InMemoryEngineResetTests {
    @Test("A negative index or one past the processed count throws")
    func outOfRange() async throws {
        let model = HashEngine(model: HashModel(vocab: 8, levels: 1000, salt: 1))
        try await model.advance([1, 2, 3])
        await #expect(throws: InferenceRuntimeError.self) { try await model.reset(to: -1) }
        await #expect(throws: InferenceRuntimeError.self) { try await model.reset(to: 4) }
        #expect(model.processedTokenCount == 3)
    }

    @Test("A refused reset leaves the committed tokens and the next verify intact")
    func refusedResetChangesNothing() async throws {
        let hashModel = HashModel(vocab: 8, levels: 1000, salt: 2)
        let model = HashEngine(model: hashModel)
        try await model.advance([1, 2, 3, 4])
        model.refusesResets = true
        await #expect(throws: InferenceRuntimeError.self) { try await model.reset(to: 1) }
        #expect(model.committedTokens == [1, 2, 3, 4])
        #expect(model.hashes.count == 4)
        let out = try await model.forwardPerPositionArgmax([5])
        #expect(out.count == 1)
        #expect(model.processedTokenCount == 5)
    }
}

// MARK: - Engine state (needs an exported static-shape bundle)

/// Loads the bundle named by `COREAI_TEST_STATIC_BUNDLE`: a static-shape Gemma 4 export with a
/// 576-deep sliding ring and the metadata layout this tree reads. The tests are skipped without it.
private func loadStaticShapeEngine() async throws -> StaticShapeEngine {
    let path = ProcessInfo.processInfo.environment["COREAI_TEST_STATIC_BUNDLE"] ?? ""
    let bundle = try LanguageModelBundle(from: path)
    try bundle.modelBundle.validateModelAssets()
    let engine = try await EngineFactory.createEngine(
        bundle: bundle, options: EngineOptions(tensorData: bundle.tensorData))
    return try #require(engine as? StaticShapeEngine)
}

@Suite(
    "Static-shape engine state", .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_STATIC_BUNDLE"] != nil))
struct StaticShapeEngineStateTests {
    @Test("reset(to:) past the processed count throws and changes nothing")
    func resetBeyondProcessed() async throws {
        let engine = try await loadStaticShapeEngine()
        try await engine.advance(Array(repeating: 100, count: 10))
        await #expect(throws: InferenceRuntimeError.self) { try await engine.reset(to: 20) }
        #expect(engine.processedTokenCount == 10)
    }

    @Test("reset(to:) with a negative index throws and changes nothing")
    func resetNegative() async throws {
        let engine = try await loadStaticShapeEngine()
        try await engine.advance(Array(repeating: 100, count: 10))
        await #expect(throws: InferenceRuntimeError.self) { try await engine.reset(to: -1) }
        #expect(engine.processedTokenCount == 10)
    }

    @Test("A reset the sliding ring refuses leaves the engine consistent")
    func failedResetKeepsEngineConsistent() async throws {
        let engine = try await loadStaticShapeEngine()
        // 700 tokens exceed the 576-deep ring, so rewinding to 100 needs overwritten keys.
        try await engine.advance(Array(repeating: 100, count: 700))
        await #expect(throws: InferenceRuntimeError.self) { try await engine.reset(to: 100) }
        #expect(engine.processedTokenCount == 700)
        let out = try await engine.forwardPerPositionArgmax([5])
        #expect(out.count == 1)
        #expect(engine.processedTokenCount == 701)
    }

    @Test("A verify block wider than the widest graph throws instead of returning misaligned rows")
    func wideVerifyBlockThrows() async throws {
        let engine = try await loadStaticShapeEngine()
        try await engine.advance(Array(repeating: 100, count: 3))
        let tooWide = Array(repeating: Int32(7), count: 4 * engine.maxVerifyBlockLength)
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await engine.forwardPerPositionArgmax(tooWide)
        }
        #expect(engine.processedTokenCount == 3)
    }

    @Test("A verify block of the limit returns one argmax per token")
    func verifyBlockAtLimitReturnsAllRows() async throws {
        let engine = try await loadStaticShapeEngine()
        try await engine.advance(Array(repeating: 100, count: 3))
        let block = Array(repeating: Int32(7), count: engine.maxVerifyBlockLength)
        let out = try await engine.forwardPerPositionArgmax(block)
        #expect(out.count == block.count)
    }
}

@Suite("Speculative sequence buffer")
struct SpeculativeSequenceTests {
    @Test("A buffer that matches the processed count is accepted")
    func inSync() throws {
        try StaticShapeEngine.validateSpeculativeSequence(count: 0, processed: 0)
        try StaticShapeEngine.validateSpeculativeSequence(count: 12, processed: 12)
    }

    @Test("A buffer that disagrees with the processed count throws")
    func outOfSync() {
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.validateSpeculativeSequence(count: 0, processed: 5)
        }
        #expect(throws: InferenceRuntimeError.self) {
            try StaticShapeEngine.validateSpeculativeSequence(count: 9, processed: 4)
        }
    }
}
