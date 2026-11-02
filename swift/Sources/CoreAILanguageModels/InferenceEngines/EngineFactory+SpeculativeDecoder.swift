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

        package init(
            draftTokens: Int? = nil, alignDraftsToVerifyTile: Bool = true, adaptiveDraftCount: Bool = false,
            promptLookupMinNgram: Int? = nil, drafterTopK: Int? = nil
        ) {
            self.draftTokens = draftTokens
            self.alignDraftsToVerifyTile = alignDraftsToVerifyTile
            self.adaptiveDraftCount = adaptiveDraftCount
            self.promptLookupMinNgram = promptLookupMinNgram
            self.drafterTopK = drafterTopK
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
        guard targetEmitsHiddenStates else {
            throw InferenceRuntimeError.invalidArgument(
                "\(bundle.name)'s decode graphs do not emit hidden_states, which the draft head seeds from; "
                    + "re-export the model with models/gemma4/export.py --emit-hidden-states")
        }
        return component
    }

    /// A speculative decoder that verifies on `target` and drafts with `bundle`'s draft head
    /// (``LanguageModelBundle/drafter``).
    ///
    /// - Throws: When the bundle ships no draft head, when `target` does not emit the hidden states
    ///   the head seeds from, or when the head does not match the target.
    package static func createSpeculativeDecoder(
        target: StaticShapeEngine, bundle: LanguageModelBundle, options: DraftHeadOptions = DraftHeadOptions()
    ) async throws -> SpeculativeDecoder {
        let component = try requireDraftHead(bundle: bundle, targetEmitsHiddenStates: target.emitsHiddenStates)
        if let problem = SpeculativeDecodingOptions.draftCountProblem(options.draftTokens) {
            throw InferenceRuntimeError.invalidArgument(problem)
        }
        let drafter = try await StaticShapeDrafter.load(
            component: component, backboneHiddenSize: target.backboneHiddenSize,
            maxContextLength: bundle.maxContextLength, vocabSize: bundle.vocabSize, topK: options.drafterTopK)
        drafter.sharedKVProvider = { [weak target] kind, body in
            guard let target else { throw InferenceRuntimeError.invalidState("the target engine was released") }
            try target.withSharedKVCache(kind, body)
        }
        drafter.embeddingProvider = { [weak target] token in
            guard let target else { throw InferenceRuntimeError.invalidState("the target engine was released") }
            return try await target.drafterEmbedding(token)
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
