// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// Model-backed check that the two ways `CoreAISequentialEngine` scores a forced continuation
/// agree.
///
/// One path runs context plus continuation through `main` as one batch. The other is the
/// generation path: the context goes through prefill (the `prefill` graph when the asset has
/// one), then each continuation id goes through `main` one at a time. An asset with a prefill
/// graph beside a static-width `main` can only take the second path, so it must give the same
/// answer as the first.
///
/// Gated, because it needs a compiled bundle. Set `FORCED_CONTINUATION_PARITY_BUNDLE` to an LLM
/// bundle directory (the one `llm-runner --model` takes). The ids below must be in its vocabulary.
private enum ForcedContinuationFixture {
    static let bundlePath = ProcessInfo.processInfo.environment["FORCED_CONTINUATION_PARITY_BUNDLE"]
    static var isAvailable: Bool { bundlePath != nil }

    static let context: [Int32] = [101, 102, 103, 104, 105, 106, 107]
    static let continuation: [Int32] = [108, 109, 110]

    /// Largest allowed difference between the two paths at any logit. Both read fp16 logits, and
    /// one `main` batch and single-token `main` passes reduce in a different order. Measured on
    /// 2026-09-25: 0.018 on Qwen3-0.6B (logits near 20) and 0.078 on a random-weight
    /// mistral-tiny with a prefill graph (logits near 0.2). The same 0.078 appears when every id
    /// goes through `main` one at a time with no prefill graph, so it is `main`'s batch numerics.
    static let logitTolerance: Float = 0.1
}

@Suite(
    "Forced continuation path parity (on-device)",
    .enabled(if: ForcedContinuationFixture.isAvailable)
)
struct ForcedContinuationPathParityTests {
    private func score(
        _ engine: CoreAISequentialEngine, context: [Int32], forced: [Int32], inOneBatch: Bool
    ) async throws -> [InferenceOutput] {
        engine.scoresForcedContinuationInOneBatch = inOneBatch
        try await engine.reset()
        let sequence = try await engine.generate(
            with: context,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(
                maxTokens: forced.count, includeLogits: true, forcedContinuation: forced))
        var outputs: [InferenceOutput] = []
        for try await output in sequence {
            outputs.append(output)
        }
        try await engine.reset()
        return outputs
    }

    private func logits(_ outputs: [InferenceOutput]) throws -> [[Float]] {
        try outputs.map { try #require($0.logits).map { Float($0) } }
    }

    private func largestDifference(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    /// The chosen id, and how far it leads the runner-up.
    private func choice(_ logits: [Float]) -> (index: Int, margin: Float) {
        let index = logits.indices.max { logits[$0] < logits[$1] } ?? -1
        let runnerUp = logits.indices.filter { $0 != index }.map { logits[$0] }.max() ?? -.infinity
        return (index, logits[index] - runnerUp)
    }

    @Test("One main batch and the step path give the same logits and the same choices")
    func batchAndStepPathsAgree() async throws {
        let bundle = try LanguageModelBundle(from: try #require(ForcedContinuationFixture.bundlePath))
        let engine = try #require(
            try await EngineFactory.createEngine(
                bundle: bundle, options: EngineOptions(variant: "coreai-sequential")) as? CoreAISequentialEngine,
            "Fixture must resolve to CoreAISequentialEngine")
        let decided = engine.scoresForcedContinuationInOneBatch
        defer { engine.scoresForcedContinuationInOneBatch = decided }

        let context = ForcedContinuationFixture.context
        let forced = ForcedContinuationFixture.continuation
        let batched = try await score(engine, context: context, forced: forced, inOneBatch: true)
        let stepped = try await score(engine, context: context, forced: forced, inOneBatch: false)

        #expect(batched.map(\.tokenId) == forced)
        #expect(stepped.map(\.tokenId) == forced)

        let tolerance = ForcedContinuationFixture.logitTolerance
        var largest: Float = 0
        for (b, s) in zip(try logits(batched), try logits(stepped)) {
            #expect(b.count == s.count)
            largest = max(largest, largestDifference(b, s))
            // A lead inside the tolerance is a tie at fp16, and either path may break it.
            let (bIndex, bMargin) = choice(b)
            let (sIndex, sMargin) = choice(s)
            if bMargin > tolerance || sMargin > tolerance {
                #expect(bIndex == sIndex)
            }
        }
        #expect(largest <= tolerance, "largest logit difference \(largest)")

        // Every id through `main` one at a time: context [first id], the rest forced. With a
        // prefill graph this shares nothing with the step path before `main`, so it checks the
        // prefill graph itself. Measured: 0 on mistral-tiny, 0.018 on Qwen3-0.6B (no prefill
        // graph, so the step path's context is one `main` batch).
        let reference = try logits(
            try await score(
                engine, context: [context[0]], forced: Array(context.dropFirst()) + forced, inOneBatch: false)
        ).suffix(forced.count)
        let stepToReference = zip(try logits(stepped), reference).map(largestDifference).max() ?? 0
        print(
            "Forced continuation parity: engine decided oneBatch=\(decided), batch-to-step \(largest), "
                + "step-to-single-token \(stepToReference)")
        #expect(stepToReference <= tolerance, "step-to-single-token difference \(stepToReference)")
    }
}
