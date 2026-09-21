// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

#if canImport(CoreAI)
import CoreAI
#endif

@Suite("Multimodal types")
struct MultimodalTypeTests {
    #if canImport(CoreAI)
    @Test("InputEmbeddings wraps NDArray with positions")
    func embeddedInputBasics() throws {
        let embeddings = NDArray(
            shape: [1, 256, 2048],
            scalarType: .float16
        )
        let input = try InputEmbeddings(
            embeddings: embeddings,
            embeddingPositions: 5..<261
        )
        #expect(input.tokenCount == 256)
        #expect(input.embeddingPositions.count == 256)
    }
    #endif

    @Test("VisionConfig decodes from snake_case JSON")
    func visionConfigDecode() throws {
        let json = """
            {"image_size": 896, "patch_size": 14, "image_token_count": 256, "image_token_id": 255999}
            """
        let config = try JSONDecoder().decode(VisionConfig.self, from: json.data(using: .utf8)!)
        #expect(config.imageSize == 896)
        #expect(config.patchSize == 14)
        #expect(config.imageTokenCount == 256)
        #expect(config.imageTokenId == 255999)
    }

    @Test("LanguageConfig decodes with vision block")
    func languageConfigWithVision() throws {
        let json = """
            {
                "tokenizer": "google/gemma-3",
                "vocab_size": 262144,
                "max_context_length": 8192,
                "vision": {
                    "image_size": 896,
                    "patch_size": 14,
                    "image_token_count": 256,
                    "image_token_id": 255999
                }
            }
            """
        let config = try JSONDecoder().decode(LanguageConfig.self, from: json.data(using: .utf8)!)
        #expect(config.vision != nil)
        #expect(config.vision?.imageSize == 896)
        #expect(config.vision?.imageTokenCount == 256)
    }

    @Test("LanguageConfig decodes without vision block")
    func languageConfigWithoutVision() throws {
        let json = """
            {
                "tokenizer": "Qwen/Qwen3-0.6B",
                "vocab_size": 151936,
                "max_context_length": 32768
            }
            """
        let config = try JSONDecoder().decode(LanguageConfig.self, from: json.data(using: .utf8)!)
        #expect(config.vision == nil)
    }

    @Test("VisionConfig decodes with video fields")
    func visionConfigWithVideoFields() throws {
        let json = """
            {
                "image_size": 384,
                "patch_size": 14,
                "image_token_count": 729,
                "image_token_id": 255999,
                "max_video_frames": 16,
                "tokens_per_frame": 729
            }
            """
        let config = try JSONDecoder().decode(VisionConfig.self, from: json.data(using: .utf8)!)
        #expect(config.maxVideoFrames == 16)
        #expect(config.tokensPerFrame == 729)
        #expect(config.imageSize == 384)
    }

    @Test("VisionConfig backwards compatible without video fields")
    func visionConfigWithoutVideoFields() throws {
        let json = """
            {
                "image_size": 896,
                "patch_size": 14,
                "image_token_count": 256,
                "image_token_id": 255999
            }
            """
        let config = try JSONDecoder().decode(VisionConfig.self, from: json.data(using: .utf8)!)
        #expect(config.maxVideoFrames == nil)
        #expect(config.tokensPerFrame == nil)
        #expect(config.imageSize == 896)
        #expect(config.imageTokenCount == 256)
    }
}

// MARK: - CoreAIVLMExecutor stop token union

/// `CoreAIVLMExecutor.respond(...)` needs a live `CoreAISequentialVLMEngine`, which only
/// initializes against real compiled `.aimodel` assets, so the full "generation stops on
/// an additional stop token" path can't be exercised at the unit level here (no fake/mock
/// engine seam exists for that final class). `stopTokenSet` is the exact union logic
/// `respond()` uses to decide when to stop, extracted so it's directly testable; the tests
/// below prove additional stop tokens resolved at load are actually unioned into the set the
/// executor checks against each generated token, not just stored and unused.
@Suite("CoreAIVLMExecutor.stopTokenSet")
struct VLMStopTokenSetTests {
    private static let vocab: [String: Int] = [
        "<eos>": 2,
        "<|im_end|>": 4,
    ]

    private static func tokenizer(vocab: [String: Int] = vocab) -> any Tokenizer {
        MockTokenizer(vocab: vocab)
    }

    @Test("Main EOS is always included")
    func includesMainEos() {
        let stopTokens = CoreAIVLMExecutor.stopTokenSet(
            tokenizer: Self.tokenizer(vocab: [:]), additionalStopTokenIds: [])
        #expect(stopTokens == [2])
    }

    @Test("<|im_end|> is included when present in the vocab")
    func includesImEndWhenPresent() {
        let stopTokens = CoreAIVLMExecutor.stopTokenSet(
            tokenizer: Self.tokenizer(), additionalStopTokenIds: [])
        #expect(stopTokens == [2, 4])
    }

    @Test("Additional stop tokens resolved at load are unioned into the stop set")
    func unionsAdditionalStopTokens() {
        // Gemma's <end_of_turn> (106) and Phi's <|end|> (200020) aren't in this
        // tokenizer's vocab at all -- they only reach the stop set via
        // `additionalStopTokenIds`, mirroring what LanguageConfig.additionalStopTokenIds
        // resolves from tokenizer_config.json / tokenizer.json at load.
        let stopTokens = CoreAIVLMExecutor.stopTokenSet(
            tokenizer: Self.tokenizer(vocab: [:]), additionalStopTokenIds: [106, 200_020])
        #expect(stopTokens == [2, 106, 200_020])
    }

    @Test("Additional stop tokens combine with <|im_end|> without duplication")
    func unionsWithImEnd() {
        let stopTokens = CoreAIVLMExecutor.stopTokenSet(
            tokenizer: Self.tokenizer(), additionalStopTokenIds: [4, 106])
        #expect(stopTokens == [2, 4, 106])
    }

    @Test("Empty additional stop tokens leave the base set unchanged")
    func emptyAdditionalStopTokensNoOp() {
        let stopTokens = CoreAIVLMExecutor.stopTokenSet(
            tokenizer: Self.tokenizer(), additionalStopTokenIds: [])
        #expect(stopTokens == [2, 4])
    }
}
