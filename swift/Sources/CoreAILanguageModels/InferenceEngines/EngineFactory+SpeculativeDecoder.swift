// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

extension EngineFactory {
    /// How a draft-head decoder is configured. Every field defaults to the value the head was tuned
    /// with.
    package struct DraftHeadOptions: Sendable {
        /// Drafts per round, 1 to ``SpeculativeDefaults/maxDraftTokens``. nil drafts one less than the
        /// target's verify tile is wide, the most one verify pass scores, and
        /// ``SpeculativeDefaults/numDraftTokens`` when the target has no tile.
        package var draftTokens: Int?

        /// Propose only as many drafts as fit in the target's current verify tile, so every verify is
        /// one forward.
        package var alignDraftsToVerifyTile: Bool

        /// Adapt the draft count to acceptance.
        package var adaptiveDraftCount: Bool

        /// Also propose continuations found in the context, from n-grams of at least this many
        /// tokens. nil turns prompt lookup off.
        package var promptLookupMinNgram: Int?

        /// Centroids whose candidates the head's scan scores. nil for ``DraftScan/defaultTopK``.
        package var drafterTopK: Int?

        /// A draft-head pack (`draft_head.safetensors` from `models/gemma4/export_draft_pack.py`) to run the
        /// head on the CPU from, with ``CPUDrafter``. nil runs the bundle's draft head program.
        package var cpuPack: URL?

        package init(
            draftTokens: Int? = nil, alignDraftsToVerifyTile: Bool = true, adaptiveDraftCount: Bool = false,
            promptLookupMinNgram: Int? = nil, drafterTopK: Int? = nil, cpuPack: URL? = nil
        ) {
            self.draftTokens = draftTokens
            self.alignDraftsToVerifyTile = alignDraftsToVerifyTile
            self.adaptiveDraftCount = adaptiveDraftCount
            self.promptLookupMinNgram = promptLookupMinNgram
            self.drafterTopK = drafterTopK
            self.cpuPack = cpuPack
        }

        /// The draft count for a target whose verify tile is `verifyTileWidth` wide.
        package func resolvedDraftTokens(verifyTileWidth: Int?) -> Int {
            if let draftTokens { return draftTokens }
            if let width = verifyTileWidth, width > 1 { return width - 1 }
            return SpeculativeDefaults.numDraftTokens
        }
    }

    /// The bundle's draft head, after checking that the target can feed it.
    ///
    /// - Throws: When the bundle ships no draft head, or when the target's decode graphs do not emit
    ///   the hidden states the head seeds from.
    static func requireDraftHead(bundle: LanguageModelBundle, targetEmitsHiddenStates: Bool) throws
        -> DrafterComponent
    {
        guard let component = try bundle.drafter else {
            throw InferenceRuntimeError.invalidArgument(
                "\(bundle.name) has no draft head; add one with models/gemma4/export_drafter.py --bundle")
        }
        try requireHiddenStates(bundle: bundle, targetEmitsHiddenStates: targetEmitsHiddenStates)
        return component
    }

    /// Throws unless the target's decode graphs emit the hidden states a draft head seeds from.
    static func requireHiddenStates(bundle: LanguageModelBundle, targetEmitsHiddenStates: Bool) throws {
        guard targetEmitsHiddenStates else {
            throw InferenceRuntimeError.invalidArgument(
                "\(bundle.name)'s decode graphs do not emit hidden_states, which the draft head seeds from; "
                    + "re-export the model with models/gemma4/export.py --emit-hidden-states")
        }
    }

    /// The draft head in `pack` on the CPU, connected to a target through `sharedKV` and
    /// `embedding`. The bundle need not declare a draft head.
    ///
    /// - Throws: When the target emits no hidden states, when the pack is malformed or sized for
    ///   another target, or when the target's shared caches do not fit the head.
    static func makeCPUDrafter(
        pack: URL, bundle: LanguageModelBundle, targetEmitsHiddenStates: Bool, backboneHiddenSize: Int,
        topK: Int?,
        sharedKV:
            @escaping (StaticShapeEngine.SharedKVCacheKind, (StaticShapeEngine.SharedKVCache) -> Void) throws ->
            Void,
        embedding: @escaping (Int32) async throws -> [Float]
    ) throws -> CPUDrafter {
        try requireHiddenStates(bundle: bundle, targetEmitsHiddenStates: targetEmitsHiddenStates)
        let cpu = try CPUDrafter(
            pack: pack, backboneHiddenSize: backboneHiddenSize, maxContextLength: bundle.maxContextLength,
            vocabSize: bundle.vocabSize, topK: topK)
        // A head that does not fit the target's caches fails here, not at the first draft.
        try cpu.checkTarget(sharedKV)
        cpu.sharedKVProvider = sharedKV
        cpu.embeddingProvider = embedding
        return cpu
    }

    /// A speculative decoder that verifies on `target` and drafts with `bundle`'s draft head
    /// (``LanguageModelBundle/drafter``), or with the head in ``DraftHeadOptions/cpuPack`` on the CPU.
    ///
    /// - Throws: When the bundle ships no draft head and no pack is given, when `target` does not
    ///   emit the hidden states the head seeds from, or when the head does not match the target.
    package static func createSpeculativeDecoder(
        target: StaticShapeEngine, bundle: LanguageModelBundle, options: DraftHeadOptions = DraftHeadOptions()
    ) async throws -> SpeculativeDecoder {
        if let problem = SpeculativeDecodingOptions.draftCountProblem(options.draftTokens) {
            throw InferenceRuntimeError.invalidArgument(problem)
        }
        let sharedKV: (StaticShapeEngine.SharedKVCacheKind, (StaticShapeEngine.SharedKVCache) -> Void) throws -> Void =
            { [weak target] kind, body in
                guard let target else { throw InferenceRuntimeError.invalidState("the target engine was released") }
                try target.withSharedKVCache(kind, body)
            }
        let embedding: (Int32) async throws -> [Float] = { [weak target] token in
            guard let target else { throw InferenceRuntimeError.invalidState("the target engine was released") }
            return try await target.drafterEmbedding(token)
        }
        let drafter: any DrafterEngine
        if let pack = options.cpuPack {
            drafter = try makeCPUDrafter(
                pack: pack, bundle: bundle, targetEmitsHiddenStates: target.emitsHiddenStates,
                backboneHiddenSize: target.backboneHiddenSize, topK: options.drafterTopK,
                sharedKV: sharedKV, embedding: embedding)
        } else {
            let component = try requireDraftHead(bundle: bundle, targetEmitsHiddenStates: target.emitsHiddenStates)
            let head = try await StaticShapeDrafter.load(
                component: component, backboneHiddenSize: target.backboneHiddenSize,
                maxContextLength: bundle.maxContextLength, vocabSize: bundle.vocabSize, topK: options.drafterTopK)
            head.sharedKVProvider = sharedKV
            head.embeddingProvider = embedding
            drafter = head
        }
        let decoder = SpeculativeDecoder(
            drafter: drafter, target: target,
            numDraftTokens: options.resolvedDraftTokens(verifyTileWidth: target.verifyTileWidth),
            alignDraftsToVerifyTile: options.alignDraftsToVerifyTile,
            adaptiveDraftCount: options.adaptiveDraftCount)
        if let minNgram = options.promptLookupMinNgram {
            decoder.promptLookup = PromptLookup(
                maxNgram: max(SpeculativeDefaults.lookupMaxNgram, minNgram), minNgram: minNgram)
        }
        return decoder
    }
}
