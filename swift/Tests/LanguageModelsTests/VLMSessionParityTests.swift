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

/// Bundle and image paths from the environment, and the prompts the tests use.
private enum VLMParityFixture {
    static let bundlePath = ProcessInfo.processInfo.environment["VLM_PARITY_BUNDLE"]
    static let imagePath = ProcessInfo.processInfo.environment["VLM_PARITY_IMAGE"]
    static var isAvailable: Bool { bundlePath != nil && imagePath != nil }

    static let steps = 24
    static let question1 = "Describe this image in one sentence."
    static let question2 = "What colors stand out the most?"
    static let followUpText = "\nWhich colors stand out the most?"
}

/// Model-backed checks for vision-language generation on `CoreAISequentialEngine` sessions and
/// its prefix-cache rule (a multimodal request rewinds to its first image placeholder; a text-only
/// follow-up continues on the cached image).
///
/// Gated on environment variables because they need a compiled VLM bundle:
///
///   VLM_PARITY_BUNDLE — path to a separate-embedding VLM bundle directory (metadata.json kind=vlm)
///   VLM_PARITY_IMAGE  — path to an image file
///
/// A rewound session's logits are printed (`VLM_PARITY …` lines), not asserted equal to a fresh
/// session's: its cached prefix was computed in a different prefill batch, or token by token during
/// decode, and this model's numerics depend on the batch.
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
        let engine = try #require(
            try await CoreAIRunner(bundle: bundle).makeInferenceEngine() as? CoreAISequentialEngine,
            "Fixture must resolve to CoreAISequentialEngine")
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

    /// Runs a multimodal request. `resumesAt` is the KV position the session must resume from,
    /// checked before the first step.
    private func multimodal(
        _ tokens: [Int32], image: InputEmbeddings? = nil, session: GenerationSessionState, resumesAt: Int? = nil,
        fixture: Fixture
    ) async throws -> Run {
        let sequence = try await fixture.engine.generate(
            with: image ?? fixture.image, tokens: tokens, sessionState: session,
            samplingConfiguration: .greedy, inferenceOptions: options)
        if let resumesAt { #expect(session.processedTokenCount == resumesAt) }
        return try await collect(sequence)
    }

    /// Runs a text-only request. `resumesAt` is checked as in `multimodal`.
    private func textOnly(
        _ tokens: [Int32], session: GenerationSessionState, resumesAt: Int? = nil,
        inferenceOptions: InferenceOptions? = nil, fixture: Fixture
    ) async throws -> Run {
        let sequence = try await fixture.engine.generate(
            with: tokens, sessionState: session,
            samplingConfiguration: .greedy, inferenceOptions: inferenceOptions ?? options)
        if let resumesAt { #expect(session.processedTokenCount == resumesAt) }
        return try await collect(sequence)
    }

    /// Length of the shared prefix of `a` and `b`.
    private func sharedPrefix(_ a: [Int32], _ b: [Int32]) -> Int {
        zip(a, b).prefix { $0 == $1 }.count
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

    @Test("a templated second turn on a used session rewinds to the image")
    func multiturnTemplatedRewinds() async throws {
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
        let start = try imageStart(of: turn2, fixture)
        let reused = try await multimodal(turn2, session: session, resumesAt: start, fixture: fixture)
        let hit = fixture.engine.lastPrefixHitCount
        let fresh = try await multimodal(turn2, session: try fixture.engine.makeSessionState(), fixture: fixture)

        print("VLM_PARITY turn1: \(decode(answer1.tokens, fixture))")
        print("VLM_PARITY templated-turn2 prefixHit=\(hit)/\(turn2.count) \(compare(reused, fresh))")
        #expect(hit == start)
    }

    @Test("the same image extended with follow-up text rewinds to the image")
    func multiturnSameImageRewinds() async throws {
        let fixture = try await makeFixture()
        let followUp = fixture.tokenizer.encode(text: VLMParityFixture.followUpText, addSpecialTokens: false)
            .map(Int32.init)
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let start = try imageStart(of: turn1, fixture)

        let session = try fixture.engine.makeSessionState()
        let answer1 = try await multimodal(turn1, session: session, fixture: fixture)

        // An exact extension of turn 1: reuse is cut at the image, and the session rewinds there.
        let extended = turn1 + answer1.tokens + followUp
        let reused = try await multimodal(extended, session: session, resumesAt: start, fixture: fixture)
        let hit = fixture.engine.lastPrefixHitCount
        let fresh = try await multimodal(extended, session: try fixture.engine.makeSessionState(), fixture: fixture)

        print("VLM_PARITY same-image-extension prefixHit=\(hit)/\(extended.count) \(compare(reused, fresh))")
        #expect(hit == start)
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

        // Identical tokens (same placeholder count): the original image, then a different one.
        let extended = turn1 + answer1.tokens + followUp
        let start = try imageStart(of: extended, fixture)
        let original = try await multimodal(extended, session: session, resumesAt: start, fixture: fixture)
        let swapped = try await multimodal(extended, image: other, session: session, resumesAt: start, fixture: fixture)
        let swapHit = fixture.engine.lastPrefixHitCount

        print("VLM_PARITY swapped-image prefixHit=\(swapHit)/\(extended.count)")
        print("VLM_PARITY original-image: \(decode(original.tokens, fixture))")
        print("VLM_PARITY swapped-image: \(decode(swapped.tokens, fixture))")
        #expect(swapHit == start)
        #expect(swapped.tokens != original.tokens)
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

        // The text turn's last sampled token is not in the KV cache yet.
        let cached = textPrompt.count + answer.tokens.count - 1
        let reused = try await multimodal(extended, session: session, resumesAt: cached, fixture: fixture)
        let hit = fixture.engine.lastPrefixHitCount
        let fresh = try await multimodal(extended, session: try fixture.engine.makeSessionState(), fixture: fixture)

        print(
            "VLM_PARITY image-after-text prefixHit=\(hit)/\(extended.count) imageStart=\(start) "
                + "\(compare(reused, fresh))")
        print("VLM_PARITY image-after-text: \(decode(reused.tokens, fixture))")
        #expect(hit == cached)
    }

    @Test("a text-only re-request reseeds one token back, and rewinds to the template after an image turn")
    func textOnlyRewindRoundTrip() async throws {
        let fixture = try await makeFixture()
        let prompt = try fixture.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "List three colors."]]
        ).map(Int32.init)

        let session = try fixture.engine.makeSessionState()
        let first = try await textOnly(prompt, session: session, fixture: fixture)

        let again = try await textOnly(prompt, session: session, resumesAt: prompt.count - 1, fixture: fixture)
        #expect(fixture.engine.lastPrefixHitCount == prompt.count)
        print("VLM_PARITY text-rewind \(compare(again, first))")

        // A different text-only request after an image turn rewinds to the shared template prefix.
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let shared = sharedPrefix(prompt, turn1)
        let start = try imageStart(of: turn1, fixture)
        try #require(shared <= start)
        _ = try await multimodal(turn1, session: session, fixture: fixture)
        let afterImage = try await textOnly(prompt, session: session, resumesAt: shared, fixture: fixture)
        #expect(fixture.engine.lastPrefixHitCount == shared)
        print("VLM_PARITY text-after-image \(compare(afterImage, first))")
    }

    @Test("a text-only follow-up after an image turn continues on the cached image")
    func textFollowUpReusesImage() async throws {
        let fixture = try await makeFixture()
        let followUp = fixture.tokenizer.encode(text: VLMParityFixture.followUpText, addSpecialTokens: false)
            .map(Int32.init)
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let session = try fixture.engine.makeSessionState()

        // Image placeholders that are not in the KV cache need the image.
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await textOnly(turn1, session: session, fixture: fixture)
        }

        let answer1 = try await multimodal(turn1, session: session, fixture: fixture)
        let extended = turn1 + answer1.tokens + followUp
        let cached = turn1.count + answer1.tokens.count - 1
        let reused = try await textOnly(extended, session: session, resumesAt: cached, fixture: fixture)
        let hit = fixture.engine.lastPrefixHitCount
        let fresh = try await multimodal(extended, session: try fixture.engine.makeSessionState(), fixture: fixture)

        print("VLM_PARITY text-follow-up prefixHit=\(hit)/\(extended.count) \(compare(reused, fresh))")
        print("VLM_PARITY text-follow-up: \(decode(reused.tokens, fixture))")
        #expect(hit == cached)
    }

    @Test("a templated text-only second turn rewinds past the cached image")
    func templatedTextTurnReusesImage() async throws {
        let fixture = try await makeFixture()
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let session = try fixture.engine.makeSessionState()
        let answer1 = try await multimodal(turn1, session: session, fixture: fixture)

        // The template re-encodes the assistant turn, so the match ends somewhere past the image.
        let turn2 = try promptTokens(
            [
                ["role": "user", "content": VLMParityFixture.question1],
                ["role": "assistant", "content": fixture.tokenizer.decode(tokens: answer1.tokens.map(Int.init))],
                ["role": "user", "content": VLMParityFixture.question2],
            ], fixture: fixture)
        let shared = sharedPrefix(turn2, turn1 + answer1.tokens.dropLast())
        let imageEnd = try imageStart(of: turn1, fixture) + fixture.image.tokenCount
        try #require(shared >= imageEnd)

        let reply = try await textOnly(turn2, session: session, resumesAt: shared, fixture: fixture)
        print("VLM_PARITY templated-text-turn2 prefixHit=\(fixture.engine.lastPrefixHitCount)/\(turn2.count)")
        print("VLM_PARITY templated-text-turn2: \(decode(reply.tokens, fixture))")
        #expect(fixture.engine.lastPrefixHitCount == shared)
    }

    @Test("a rejected text-only request leaves the session's generation and hit count alone")
    func rejectedRequestHasNoSideEffects() async throws {
        let fixture = try await makeFixture()
        let followUp = fixture.tokenizer.encode(text: VLMParityFixture.followUpText, addSpecialTokens: false)
            .map(Int32.init)
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let start = try imageStart(of: turn1, fixture)
        let session = try fixture.engine.makeSessionState()
        let answer1 = try await multimodal(turn1, session: session, fixture: fixture)

        let sequence = try await fixture.engine.generate(
            with: fixture.image, tokens: turn1 + answer1.tokens + followUp, sessionState: session,
            samplingConfiguration: .greedy, inferenceOptions: options)
        let iterator = sequence.makeAsyncIterator()
        _ = try await iterator.next()
        let processed = session.processedTokenCount

        // Diverges at position 0, so none of its placeholders are cached.
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await fixture.engine.generate(
                with: followUp + turn1, sessionState: session, samplingConfiguration: .greedy,
                inferenceOptions: options)
        }
        #expect(session.lastPrefixHitCount == start)
        #expect(fixture.engine.lastPrefixHitCount == start)
        #expect(session.processedTokenCount == processed)

        while try await iterator.next() != nil {}
        #expect(sequence.stopReason != .cancelled)
    }

    @Test("forced-continuation scoring after an image turn runs only the uncached tokens")
    func forcedContinuationAfterImage() async throws {
        let fixture = try await makeFixture()
        let turn1 = try promptTokens([["role": "user", "content": VLMParityFixture.question1]], fixture: fixture)
        let session = try fixture.engine.makeSessionState()
        let answer1 = try await multimodal(turn1, session: session, fixture: fixture)

        // `turn1` is a strict prefix of the history: reseed its last token, then score the answer.
        let scored = try await textOnly(
            turn1, session: session, resumesAt: turn1.count - 1,
            inferenceOptions: .eval(maxTokens: answer1.tokens.count, forcedContinuation: answer1.tokens),
            fixture: fixture)
        // As in decode, the last forced token is not run.
        #expect(session.processedTokenCount == turn1.count + answer1.tokens.count - 1)
        #expect(session.history.count == session.processedTokenCount)

        let agree = zip(scored.logits, answer1.tokens).filter { logits, token in
            logits.indices.max { logits[$0] < logits[$1] } == Int(token)
        }.count
        print("VLM_PARITY forced-continuation argmax agrees \(agree)/\(answer1.tokens.count)")
        #expect(scored.logits.count == answer1.tokens.count)
        #expect(agree * 2 >= answer1.tokens.count)
    }
}
