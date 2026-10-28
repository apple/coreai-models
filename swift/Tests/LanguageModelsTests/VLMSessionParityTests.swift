// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - VLM Session Parity (on-device, requires a compiled VLM bundle)

/// Model-backed checks for vision-language generation on `CoreAISequentialEngine` sessions and
/// its prefix-cache rule (reuse stops at the first image placeholder).
///
/// Gated on environment variables because they need a compiled VLM bundle:
///
///   VLM_PARITY_BUNDLE — path to a separate-embedding VLM bundle directory (metadata.json kind=vlm)
///   VLM_PARITY_IMAGE  — path to an image file
///
/// Results are also printed (`VLM_PARITY …` lines) so on-device runs can be compared across
/// machines and models.
private enum VLMParityFixture {
    static let bundlePath = ProcessInfo.processInfo.environment["VLM_PARITY_BUNDLE"]
    static let imagePath = ProcessInfo.processInfo.environment["VLM_PARITY_IMAGE"]
    static var isAvailable: Bool { bundlePath != nil && imagePath != nil }

    static let steps = 24
    static let question1 = "Describe this image in one sentence."
    static let question2 = "What colors stand out the most?"
    static let followUpText = "\nWhich colors stand out the most?"
}

@Suite(
    "VLM session parity (on-device)",
    .serialized,
    .enabled(if: VLMParityFixture.isAvailable)
)
struct VLMSessionParityTests {
    private struct Fixture {
        let engine: CoreAISequentialEngine
        let tokenizer: any Tokenizer
        let image: InputEmbeddings
        let imageTokenId: Int32
    }

    private struct Run {
        var tokens: [Int32] = []
        var logits: [[LogitsScalarType]] = []
    }

    private func makeFixture() async throws -> Fixture {
        let bundle = try LanguageModelBundle(at: URL(fileURLWithPath: VLMParityFixture.bundlePath!))
        let visionConfig = try #require(bundle.visionConfig, "Fixture must be a VLM bundle")
        let engine = try await EngineFactory.makeSequentialVLMEngine(bundle: bundle)
        #expect(engine.acceptsMultimodalInput)
        let tokenizer = try await bundle.loadTokenizer()
        let image = try await engine.encodeImage(at: URL(fileURLWithPath: VLMParityFixture.imagePath!))
        return Fixture(
            engine: engine, tokenizer: tokenizer, image: image, imageTokenId: visionConfig.imageTokenId)
    }

    /// Chat-templated tokens for `messages`, with the user turn at `imageTurn` carrying the image:
    /// the template's single image token is expanded to the image's placeholder count.
    private func promptTokens(
        _ messages: [[String: String]], imageTurn: Int = 0, fixture: Fixture
    ) throws -> [Int32] {
        let imageToken = fixture.tokenizer.convertIdToToken(Int(fixture.imageTokenId)) ?? "<|image_pad|>"
        var templated = messages
        templated[imageTurn]["content"] = "\(imageToken)\n" + (messages[imageTurn]["content"] ?? "")
        let raw = try fixture.tokenizer.applyChatTemplate(messages: templated.map { $0 as Message })
        var result: [Int32] = []
        var expanded = false
        for token in raw.map(Int32.init) {
            if token == fixture.imageTokenId {
                if !expanded {
                    result += [Int32](repeating: fixture.imageTokenId, count: fixture.image.tokenCount)
                    expanded = true
                }
                continue
            }
            result.append(token)
        }
        try #require(expanded, "Chat template dropped the image token")
        return result
    }

    private func collect(_ sequence: CoreAISequentialEngine.GenerationSequence) async throws -> Run {
        var run = Run()
        for try await output in sequence {
            run.tokens.append(output.tokenId)
            if let logits = output.logits { run.logits.append(logits) }
        }
        return run
    }

    private var options: InferenceOptions {
        InferenceOptions(maxTokens: VLMParityFixture.steps, logits: .lastPosition)
    }

    private func multimodal(
        _ tokens: [Int32], image: InputEmbeddings? = nil, session: GenerationSessionState, fixture: Fixture
    ) async throws -> Run {
        try await collect(
            fixture.engine.generate(
                with: image ?? fixture.image, tokens: tokens, sessionState: session,
                samplingConfiguration: .greedy, inferenceOptions: options))
    }

    private func textOnly(
        _ tokens: [Int32], session: GenerationSessionState, fixture: Fixture
    ) async throws -> Run {
        try await collect(
            fixture.engine.generate(
                with: tokens, sessionState: session,
                samplingConfiguration: .greedy, inferenceOptions: options))
    }

    /// A solid-color image, distinct from the fixture image.
    private func solidImage(red: CGFloat, green: CGFloat, blue: CGFloat) throws -> CGImage {
        let context = try #require(
            CGContext(
                data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(red: red, green: green, blue: blue, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        return try #require(context.makeImage())
    }

    private func imageStart(of tokens: [Int32], _ fixture: Fixture) throws -> Int {
        try #require(tokens.firstIndex(of: fixture.imageTokenId))
    }

    private struct Comparison: CustomStringConvertible {
        /// Leading steps where both runs chose the same token.
        var sameTokens: Int
        var steps: Int
        /// Largest absolute logit difference over the steps whose inputs were still identical
        /// (up to and including the first divergence).
        var maxAbsDiff: Float
        /// At the first divergence: the reference run's top-1 minus top-2 logit gap. A small
        /// gap means the runs disagreed on a near-tie.
        var divergenceMargin: Float?

        var description: String {
            "sameTokens=\(sameTokens)/\(steps) maxAbsDiffBeforeDivergence=\(maxAbsDiff) "
                + "divergenceMargin=\(divergenceMargin.map { "\($0)" } ?? "none")"
        }
    }

    /// Compare `run` against `reference` on the steps where both saw identical inputs.
    private func compare(_ run: Run, _ reference: Run) -> Comparison {
        let sameTokens = zip(run.tokens, reference.tokens).prefix { $0 == $1 }.count
        let comparable = min(sameTokens + 1, run.logits.count, reference.logits.count)
        var maxAbsDiff: Float = 0
        for step in 0..<comparable {
            for (p, q) in zip(run.logits[step], reference.logits[step]) {
                maxAbsDiff = max(maxAbsDiff, abs(Float(p) - Float(q)))
            }
        }
        var margin: Float?
        if sameTokens < min(run.tokens.count, reference.tokens.count), sameTokens < reference.logits.count {
            let sorted = reference.logits[sameTokens].map(Float.init).sorted(by: >)
            margin = sorted[0] - sorted[1]
        }
        return Comparison(
            sameTokens: sameTokens, steps: reference.tokens.count, maxAbsDiff: maxAbsDiff,
            divergenceMargin: margin)
    }

    private func decode(_ tokens: [Int32], _ fixture: Fixture) -> String {
        fixture.tokenizer.decode(tokens: tokens.map(Int.init)).replacingOccurrences(of: "\n", with: "\\n")
    }

    @Test("AT-MULTITURN: a templated second turn on a used session equals a fresh session")
    func multiturnTemplatedMatchesFresh() async throws {
        let fixture = try await makeFixture()
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let session = try fixture.engine.makeSessionState()
        let answer1 = try await multimodal(turn1, session: session, fixture: fixture)

        let turn2 = try promptTokens(
            [
                ["role": "user", "content": VLMParityFixture.question1],
                ["role": "assistant", "content": fixture.tokenizer.decode(tokens: answer1.tokens.map(Int.init))],
                ["role": "user", "content": VLMParityFixture.question2],
            ], fixture: fixture)
        let reused = try await multimodal(turn2, session: session, fixture: fixture)
        let hit = fixture.engine.lastPrefixHitCount
        let fresh = try await multimodal(turn2, session: try fixture.engine.makeSessionState(), fixture: fixture)
        let start = try imageStart(of: turn2, fixture)

        print("VLM_PARITY turn1: \(decode(answer1.tokens, fixture))")
        print("VLM_PARITY templated-turn2 prefixHit=\(hit)/\(turn2.count) \(compare(reused, fresh))")
        // Reuse stops at the image, so the used session re-prefills and matches a fresh one bit for bit.
        #expect(hit <= start)
        #expect(reused.tokens == fresh.tokens)
        #expect(reused.logits == fresh.logits)
        #expect(!reused.tokens.isEmpty)
    }

    @Test("AT-MULTITURN: the same image extended with follow-up text equals a fresh session")
    func multiturnSameImageMatchesFresh() async throws {
        let fixture = try await makeFixture()
        let followUp = fixture.tokenizer.encode(text: VLMParityFixture.followUpText, addSpecialTokens: false)
            .map(Int32.init)
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let start = try imageStart(of: turn1, fixture)

        let session = try fixture.engine.makeSessionState()
        let answer1 = try await multimodal(turn1, session: session, fixture: fixture)

        // An exact extension of turn 1: every token matches, and reuse is still cut at the image.
        let extended = turn1 + answer1.tokens + followUp
        let reused = try await multimodal(extended, session: session, fixture: fixture)
        #expect(fixture.engine.lastPrefixHitCount == start)
        let fresh = try await multimodal(extended, session: try fixture.engine.makeSessionState(), fixture: fixture)

        print("VLM_PARITY same-image-extension prefixHit=\(start)/\(extended.count) \(compare(reused, fresh))")
        #expect(reused.tokens == fresh.tokens)
        #expect(reused.logits == fresh.logits)
    }

    @Test("a different image at the same positions is never served from the cached prefix")
    func differentImageNotReused() async throws {
        let fixture = try await makeFixture()
        let followUp = fixture.tokenizer.encode(text: VLMParityFixture.followUpText, addSpecialTokens: false)
            .map(Int32.init)
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)

        let session = try fixture.engine.makeSessionState()
        let answer1 = try await multimodal(turn1, session: session, fixture: fixture)

        let other = try await fixture.engine.encodeImage(cgImage: try solidImage(red: 0.9, green: 0.1, blue: 0.1))
        try #require(other.tokenCount == fixture.image.tokenCount)

        // Identical tokens (same placeholder count), different image.
        let extended = turn1 + answer1.tokens + followUp
        let swapped = try await multimodal(extended, image: other, session: session, fixture: fixture)
        let swapHit = fixture.engine.lastPrefixHitCount
        let start = try imageStart(of: extended, fixture)
        #expect(swapHit <= start)
        let fresh = try await multimodal(
            extended, image: other, session: try fixture.engine.makeSessionState(), fixture: fixture)

        print("VLM_PARITY swapped-image: \(decode(swapped.tokens, fixture))")
        #expect(swapped.tokens == fresh.tokens)
        #expect(swapped.logits == fresh.logits)
    }

    @Test("an image after the cached text reuses the text and prefills the image on the session")
    func imageAfterTextReusesText() async throws {
        let fixture = try await makeFixture()
        let textPrompt = try fixture.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "List three colors."]]
        ).map(Int32.init)

        let session = try fixture.engine.makeSessionState()
        let answer = try await textOnly(textPrompt, session: session, fixture: fixture)

        // An exact extension of the text turn that adds an image and a question.
        let newline = fixture.tokenizer.encode(text: "\n", addSpecialTokens: false).map(Int32.init)
        let question = fixture.tokenizer.encode(
            text: "\n" + VLMParityFixture.question1, addSpecialTokens: false
        ).map(Int32.init)
        let extended =
            textPrompt + answer.tokens + newline
            + [Int32](repeating: fixture.imageTokenId, count: fixture.image.tokenCount) + question
        let start = try imageStart(of: extended, fixture)

        let reused = try await multimodal(extended, session: session, fixture: fixture)
        let hit = fixture.engine.lastPrefixHitCount
        let fresh = try await multimodal(extended, session: try fixture.engine.makeSessionState(), fixture: fixture)

        // Measured, not asserted equal: the reused text KV was built token by token during decode.
        print(
            "VLM_PARITY image-after-text prefixHit=\(hit)/\(extended.count) imageStart=\(start) "
                + "\(compare(reused, fresh))")
        print("VLM_PARITY image-after-text: \(decode(reused.tokens, fixture))")
        #expect(hit == textPrompt.count + answer.tokens.count - 1)
        #expect(hit <= start)
        #expect(!reused.tokens.isEmpty)
    }

    @Test("text-only: re-requesting a prefix on the VLM decoder rewinds and reproduces the run")
    func textOnlyRewindRoundTrip() async throws {
        let fixture = try await makeFixture()
        let prompt = try fixture.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "List three colors."]]
        ).map(Int32.init)

        let session = try fixture.engine.makeSessionState()
        let first = try await textOnly(prompt, session: session, fixture: fixture)

        // Identical re-request: the session rewinds one token to reseed instead of full-resetting.
        let again = try await textOnly(prompt, session: session, fixture: fixture)
        #expect(fixture.engine.lastPrefixHitCount == prompt.count)
        print("VLM_PARITY text-rewind \(compare(again, first))")
        #expect(again.tokens == first.tokens)

        // A text-only request after an image turn does not reuse the image KV.
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let start = try imageStart(of: turn1, fixture)
        _ = try await multimodal(turn1, session: session, fixture: fixture)
        let afterImage = try await textOnly(prompt, session: session, fixture: fixture)
        #expect(fixture.engine.lastPrefixHitCount <= start)
        #expect(afterImage.tokens == first.tokens)
    }
}
