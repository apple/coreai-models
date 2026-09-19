// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Fixtures

/// Builds a `GenerationSessionState` backed by the in-module `MockStateHandler`
/// (defined in `StateHandlerTests.swift`), so state round-trip and prefix-cache
/// behavior can be exercised without a compiled model asset.
private func makeSessionState(hasNonTruncatableStates: Bool = false) -> GenerationSessionState {
    let kv = MockStateHandler(names: ["key_cache", "value_cache"], shape: [1, 8, 16, 32])
    return GenerationSessionState(
        kvCache: kv,
        additionalStates: nil,
        hasNonTruncatableStates: hasNonTruncatableStates
    )
}

// MARK: - Session State Round-Trip (model-free)

@Suite("GenerationSessionState round-trip")
struct GenerationSessionStateTests {
    @Test("two sessions are independent and do not alias")
    func sessionsDoNotAlias() {
        // Distinct sessions must carry their own cursor + history. Mutating one
        // leaves the other untouched, so one engine can serve them sequentially.
        let first = makeSessionState()
        let second = makeSessionState()

        first.processedTokenCount = 5
        first.history.append(contentsOf: [1, 2, 3, 4, 5][...])

        #expect(first !== second)
        #expect(second.processedTokenCount == 0)
        #expect(second.history.count == 0)
        #expect(second.history.tokens.isEmpty)
    }
}

// MARK: - Prefix Reset Plan (model-free)

/// `CoreAISequentialEngine.prefixResetPlan` is the pure decision extracted from
/// `generate(with:sessionState:)`. It maps (commonPrefix, history/cursor, input,
/// hybrid?) to a reset target and prefix hit count. These cases pin every branch of
/// that routing so the extraction stays behavior-preserving.
@Suite("Prefix reset plan routing")
struct PrefixResetPlanTests {
    private struct Case {
        let name: String
        let commonPrefix: Int
        let historyCount: Int
        let processedTokenCount: Int
        let inputCount: Int
        let hasNonTruncatableStates: Bool
        let expectedResetTo: Int?
        let expectedHitCount: Int
    }

    @Test("routing maps each branch to its reset target and hit count")
    func routingTable() {
        let cases: [Case] = [
            Case(
                name: "empty history resolves as all-new, no reset",
                commonPrefix: 0, historyCount: 0, processedTokenCount: 0, inputCount: 3,
                hasNonTruncatableStates: false, expectedResetTo: nil, expectedHitCount: 0),
            Case(
                name: "exact re-submission rewinds one token to reseed",
                commonPrefix: 3, historyCount: 3, processedTokenCount: 3, inputCount: 3,
                hasNonTruncatableStates: false, expectedResetTo: 2, expectedHitCount: 3),
            Case(
                name: "extension beyond history keeps history, no reset",
                commonPrefix: 3, historyCount: 3, processedTokenCount: 3, inputCount: 5,
                hasNonTruncatableStates: false, expectedResetTo: nil, expectedHitCount: 3),
            Case(
                name: "mid-sequence divergence triggers a full reset",
                commonPrefix: 2, historyCount: 4, processedTokenCount: 4, inputCount: 4,
                hasNonTruncatableStates: false, expectedResetTo: 0, expectedHitCount: 2),
            Case(
                name: "first-token divergence triggers a full reset",
                commonPrefix: 0, historyCount: 3, processedTokenCount: 3, inputCount: 3,
                hasNonTruncatableStates: false, expectedResetTo: 0, expectedHitCount: 0),
            Case(
                name: "input shorter than history rewinds one token",
                commonPrefix: 3, historyCount: 5, processedTokenCount: 5, inputCount: 3,
                hasNonTruncatableStates: false, expectedResetTo: 2, expectedHitCount: 3),
            Case(
                name: "hybrid full-resets on rewind (commonPrefix < history)",
                commonPrefix: 2, historyCount: 4, processedTokenCount: 4, inputCount: 5,
                hasNonTruncatableStates: true, expectedResetTo: 0, expectedHitCount: 0),
            Case(
                name: "hybrid full-resets on re-submission (processed >= input)",
                commonPrefix: 3, historyCount: 3, processedTokenCount: 3, inputCount: 3,
                hasNonTruncatableStates: true, expectedResetTo: 0, expectedHitCount: 0),
            Case(
                name: "hybrid pure-extension keeps state, no reset",
                commonPrefix: 3, historyCount: 3, processedTokenCount: 3, inputCount: 5,
                hasNonTruncatableStates: true, expectedResetTo: nil, expectedHitCount: 0),
        ]

        for c in cases {
            let plan = CoreAISequentialEngine.prefixResetPlan(
                commonPrefix: c.commonPrefix,
                historyCount: c.historyCount,
                processedTokenCount: c.processedTokenCount,
                inputCount: c.inputCount,
                hasNonTruncatableStates: c.hasNonTruncatableStates)
            #expect(plan.resetTo == c.expectedResetTo, "\(c.name): resetTo")
            #expect(plan.prefixHitCount == c.expectedHitCount, "\(c.name): prefixHitCount")
        }
    }
}

// MARK: - Single-Active-Generation Contract (model-free)

/// A `GenerationSessionState` is single-owner: only one active generation may drive
/// it at a time. The shim enforces this with a `GenerationTokenBox`, cancelling any
/// prior token before installing a new one (see generate(with:sessionState:)).
@Suite("Single-active-generation contract")
struct SingleActiveGenerationTests {
    @Test("shim supersede sequence cancels the previous generation")
    func supersedeCancelsPrevious() {
        // Mirrors generate(with:sessionState:): cancelActive() then install(newToken).
        let box = GenerationTokenBox()
        let first = GenerationToken()
        box.install(first)

        // New generation supersedes.
        box.cancelActive()
        let second = GenerationToken()
        box.install(second)

        #expect(first.isCancelled)
        #expect(!second.isCancelled)
        #expect(box.isBusy)
    }
}

// MARK: - Idempotent Capability Routing (model-free)

/// A minimal `IdempotentEngine` conformer used purely to prove the capability-signal
/// routing (`engine is any IdempotentEngine`) matches the mechanism the codebase uses
/// for `ConstrainedGenerationCapable`.
private final class IdempotentMockEngine: MockEngine, IdempotentEngine, @unchecked Sendable {
    func makeSessionState() throws -> GenerationSessionState {
        GenerationSessionState(
            kvCache: MockStateHandler(names: ["key_cache", "value_cache"], shape: [1, 4]),
            additionalStates: nil,
            hasNonTruncatableStates: false
        )
    }

    func generate(
        with input: [TokenId],
        sessionState: GenerationSessionState,
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> MockEngine.GenerationSequence {
        // Capability signal only — delegate to the base mock's streaming path.
        try await generate(
            with: input,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions
        )
    }
}

@Suite("IdempotentEngine capability routing")
struct IdempotentEngineRoutingTests {
    /// Runtime metatype conformance check — mirrors `engine as? any IdempotentEngine`
    /// but on the type, so real engines can be probed without a compiled model.
    private func reportsIdempotent(_ type: Any.Type) -> Bool {
        type is any IdempotentEngine.Type
    }

    @Test("an IdempotentEngine conformer routes as idempotent")
    func conformerReportsCapability() {
        let engine: any InferenceEngine = IdempotentMockEngine()
        #expect(engine is any IdempotentEngine)
        #expect(engine as? any IdempotentEngine != nil)
    }

    @Test("a plain engine does not route as idempotent")
    func plainEngineDoesNotReportCapability() {
        let engine: any InferenceEngine = MockEngine()
        #expect(!(engine is any IdempotentEngine))
        #expect(engine as? any IdempotentEngine == nil)
    }

    @Test("CoreAISequentialEngine is the idempotent engine")
    func sequentialEngineIsIdempotent() {
        #expect(reportsIdempotent(CoreAISequentialEngine.self))
    }

    @Test("the other engines do not report the idempotent capability")
    func otherEnginesAreNotIdempotent() {
        #expect(!reportsIdempotent(CoreAIPipelinedEngine.self))
        #expect(!reportsIdempotent(StaticShapeEngine.self))
        #expect(!reportsIdempotent(CoreAISequentialVLMEngine.self))
    }
}

// MARK: - Parity Scaffold (Phase 2 — requires an on-device compiled model)

/// Model-backed parity checks for the idempotent-engine refactor.
///
/// These are deliberately gated: they need a compiled `.aimodel` asset and therefore
/// cannot run in model-free CI. Provide the fixture via environment variables:
///
///   IDEMPOTENT_PARITY_CONFIG — path to the model's config JSON
///   IDEMPOTENT_PARITY_MODEL  — path to the `.aimodel` / `.aimodelc` asset
///
/// When either is unset the whole suite is skipped. This is a real (non-faked)
/// scaffold: run it on-device to validate that (1) the idempotent path is
/// bit-identical to the pre-refactor engine-owned shim, and (2) a snapshot →
/// restore → continue sequence equals an uninterrupted run.
private enum ParityFixture {
    static let configPath = ProcessInfo.processInfo.environment["IDEMPOTENT_PARITY_CONFIG"]
    static let modelPath = ProcessInfo.processInfo.environment["IDEMPOTENT_PARITY_MODEL"]
    static var isAvailable: Bool { configPath != nil && modelPath != nil }

    /// Deterministic prompt tokens. Kept small and model-agnostic; the parity
    /// assertions compare the engine against itself, so exact token IDs only need
    /// to be in-vocabulary for the fixture model.
    static let prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
    static let halfTokens = 8
}

@Suite(
    "Idempotent engine parity (Phase 2, on-device)",
    .enabled(if: ParityFixture.isAvailable)
)
struct IdempotentEngineParityTests {
    /// Builds a sequential (idempotent) engine from the fixture asset.
    private func makeSequentialEngine() async throws -> CoreAISequentialEngine {
        let configData = try Data(contentsOf: URL(fileURLWithPath: ParityFixture.configPath!))
        let modelURL = URL(fileURLWithPath: ParityFixture.modelPath!)
        let engine = try await EngineFactory.createEngine(
            config: configData,
            modelURL: modelURL,
            options: EngineOptions(variant: "coreai-sequential")
        )
        let sequential = try #require(
            engine as? CoreAISequentialEngine,
            "Fixture must resolve to CoreAISequentialEngine")
        return sequential
    }

    private func collect(_ sequence: CoreAISequentialEngine.GenerationSequence) async throws -> [Int32] {
        var tokens: [Int32] = []
        for try await output in sequence {
            tokens.append(output.tokenId)
        }
        return tokens
    }

    @Test("idempotent generate(with:sessionState:) matches the engine-owned shim bit-for-bit")
    func idempotentMatchesShim() async throws {
        let engine = try await makeSequentialEngine()
        let options = InferenceOptions(maxTokens: ParityFixture.halfTokens)

        // Engine-owned shim path (pre-refactor behavior).
        let shimTokens = try await collect(
            engine.generate(
                with: ParityFixture.prompt,
                samplingConfiguration: .greedy,
                inferenceOptions: options))

        // Idempotent path over a caller-owned, freshly minted session.
        let session = try engine.makeSessionState()
        let idempotentTokens = try await collect(
            engine.generate(
                with: ParityFixture.prompt,
                sessionState: session,
                samplingConfiguration: .greedy,
                inferenceOptions: options))

        #expect(idempotentTokens == shimTokens)
        #expect(!idempotentTokens.isEmpty)
    }

    @Test("snapshot → restore → continue equals an uninterrupted run")
    func snapshotRestoreContinueEqualsUninterrupted() async throws {
        let engine = try await makeSequentialEngine()
        let n = ParityFixture.halfTokens

        // Uninterrupted reference: one session generates 2N tokens straight through.
        let referenceSession = try engine.makeSessionState()
        let reference = try await collect(
            engine.generate(
                with: ParityFixture.prompt,
                sessionState: referenceSession,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: 2 * n)))
        #expect(reference.count == 2 * n)

        // Interrupted: generate the first N, then "restore" by replaying the exact
        // prefix (prompt + firstHalf) against the same session — implicit prefix
        // caching rewinds the KV to the checkpoint — and continue for N more.
        let session = try engine.makeSessionState()
        let firstHalf = try await collect(
            engine.generate(
                with: ParityFixture.prompt,
                sessionState: session,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: n)))
        #expect(firstHalf == Array(reference.prefix(n)))

        let continuationInput = ParityFixture.prompt + firstHalf
        let secondHalf = try await collect(
            engine.generate(
                with: continuationInput,
                sessionState: session,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: n)))

        // Prefix reuse must have kicked in on the continue (at least the prompt).
        #expect(engine.lastPrefixHitCount >= ParityFixture.prompt.count)

        #expect(firstHalf + secondHalf == reference)
    }

    /// `reset(to:)` contract. Gated with the rest of the parity suite because it
    /// needs a live engine (and therefore a compiled model); it cannot run model-free.
    /// The out-of-range branch is a `precondition` (a trap, not a throw), so it is not
    /// exercised here — only the hybrid partial-reset `throw` and the valid path are.
    @Test("reset(to:) enforces its hybrid partial-reset contract")
    func resetContract() async throws {
        let engine = try await makeSequentialEngine()

        // Advance the session so the cursor is non-zero.
        _ = try await collect(
            engine.generate(
                with: ParityFixture.prompt,
                samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: ParityFixture.halfTokens)))
        #expect(engine.processedTokenCount > 0)

        if engine.hasRecurrentState {
            // Hybrid recurrent state cannot be partially rewound.
            await #expect(throws: InferenceRuntimeError.self) {
                try await engine.reset(to: 1)
            }
            // Full reset is always allowed.
            try await engine.reset(to: 0)
            #expect(engine.processedTokenCount == 0)
        } else {
            // Non-hybrid: partial reset moves the cursor back.
            try await engine.reset(to: 1)
            #expect(engine.processedTokenCount == 1)
            try await engine.reset(to: 0)
            #expect(engine.processedTokenCount == 0)
        }
    }
}
