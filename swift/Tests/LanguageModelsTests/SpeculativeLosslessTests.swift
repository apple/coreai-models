// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

/// Greedy speculative decoding must emit exactly what plain greedy decoding of the target emits,
/// whatever the draft count, verify width, context length or proposal source.
@Suite("Speculative decoding equals plain greedy")
struct SpeculativeLosslessTests {
    @Test("Random configurations, including a verify width narrower than the draft count")
    func randomConfigurations() async throws {
        var rng = SplitMix(42)
        var failures: [String] = []
        for caseIndex in 0..<600 {
            let vocab = 2 + Int(rng.next() % 40)
            let levels = [1, 2, 3, 5, 1000][Int(rng.next() % 5)]
            let model = HashModel(vocab: vocab, levels: levels, salt: rng.next())
            let tile: Int? = [nil, 4, 8, 16][Int(rng.next() % 4)]
            let block: Int? = [nil, 8, 16][Int(rng.next() % 3)]
            let verifyLimit = [Int.max, 2, 5, 8, 16][Int(rng.next() % 5)]
            let target = HashEngine(
                model: model, block: block, tile: tile, maxVerifyBlockLength: verifyLimit)
            let drafter: any DrafterEngine
            if rng.next() % 3 == 0 {
                let draftModel = HashModel(
                    vocab: vocab, levels: levels, salt: rng.next() % 2 == 0 ? model.salt : rng.next())
                drafter = AutoregressiveDrafter(engine: HashEngine(model: draftModel))
            } else {
                drafter = FuzzDrafter(model: model, seed: rng.next())
            }
            let decoder = SpeculativeDecoder(
                drafter: drafter, target: target, numDraftTokens: 1 + Int(rng.next() % 40),
                alignDraftsToVerifyTile: rng.next() % 2 == 0, adaptiveDraftCount: rng.next() % 2 == 0)
            decoder.alignProposalsToForward = rng.next() % 2 == 0
            if rng.next() % 3 == 0 {
                let minNgram = 1 + Int(rng.next() % 3)
                decoder.promptLookup = PromptLookup(
                    maxNgram: minNgram + Int(rng.next() % 3), minNgram: minNgram,
                    maxTokens: 1 + Int(rng.next() % 20))
            }
            for runIndex in 0..<3 {
                let plen = 1 + Int(rng.next() % 20)
                let prompt = (0..<plen).map { _ in Int32(rng.next() % UInt64(vocab)) }
                let stops = Set((0..<Int(rng.next() % 3)).map { _ in Int32(rng.next() % UInt64(vocab)) })
                let maxNew = 1 + Int(rng.next() % 60)
                let expected = plainGreedy(model: model, prompt: prompt, maxNew: maxNew, stops: stops)
                var got: [Int32] = []
                do {
                    _ = try await decoder.generateGreedy(
                        promptTokens: prompt, maxNewTokens: maxNew, stopTokenIds: stops,
                        onToken: { got.append($0) })
                } catch {
                    failures.append("case \(caseIndex) run \(runIndex) verifyLimit \(verifyLimit): \(error)")
                    break
                }
                if got != expected {
                    failures.append(
                        "case \(caseIndex) run \(runIndex) vocab \(vocab) levels \(levels) tile \(String(describing: tile)) block \(String(describing: block)) verifyLimit \(verifyLimit)\n  expected \(expected)\n  got      \(got)"
                    )
                    break
                }
            }
        }
        #expect(failures.isEmpty, "\(failures.count) mismatches; first: \(failures.first ?? "")")
    }

    @Test("A draft count far above the verify width is clamped to it")
    func draftCountAboveVerifyWidth() async throws {
        let model = HashModel(vocab: 30, levels: 1000, salt: 5)
        let decoder = SpeculativeDecoder(
            drafter: FuzzDrafter(model: model, seed: 3),
            target: HashEngine(model: model, maxVerifyBlockLength: 8), numDraftTokens: 100)
        var got: [Int32] = []
        _ = try await decoder.generateGreedy(
            promptTokens: [1, 2, 3], maxNewTokens: 50, stopTokenIds: [], onToken: { got.append($0) })
        #expect(got == plainGreedy(model: model, prompt: [1, 2, 3], maxNew: 50, stops: []))
    }

    @Test("A verify width of one runs plain decoding")
    func verifyWidthOfOne() async throws {
        let model = HashModel(vocab: 30, levels: 1000, salt: 6)
        let decoder = SpeculativeDecoder(
            drafter: FuzzDrafter(model: model, seed: 4),
            target: HashEngine(model: model, maxVerifyBlockLength: 1),
            numDraftTokens: 4)
        var got: [Int32] = []
        _ = try await decoder.generateGreedy(
            promptTokens: [1], maxNewTokens: 20, stopTokenIds: [], onToken: { got.append($0) })
        #expect(got == plainGreedy(model: model, prompt: [1], maxNew: 20, stops: []))
    }

    @Test("Generation runs to the end of the context and equals plain greedy there")
    func contextBoundary() async throws {
        var failures: [String] = []
        for maxContext in [12, 20, 33, 40] {
            for lookup in [false, true] {
                for numDraft in [3, 7, 30] {
                    let model = HashModel(vocab: 50, levels: 1000, salt: UInt64(maxContext))
                    let target = HashEngine(model: model, maxContextLength: maxContext)
                    let decoder = SpeculativeDecoder(
                        drafter: FuzzDrafter(model: model, seed: 1, maxContextLength: maxContext),
                        target: target, numDraftTokens: numDraft)
                    if lookup { decoder.promptLookup = PromptLookup() }
                    let prompt: [Int32] = [1]
                    var got: [Int32] = []
                    _ = try await decoder.generateGreedy(
                        promptTokens: prompt, maxNewTokens: 1000, stopTokenIds: [],
                        onToken: { got.append($0) })
                    let expected = plainGreedy(
                        model: model, prompt: prompt, maxNew: 1000, stops: [], capacity: maxContext)
                    if got != expected {
                        failures.append(
                            "maxContext \(maxContext) lookup \(lookup) drafts \(numDraft): got \(got.count) tokens, plain \(expected.count)"
                        )
                    }
                }
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }
}
