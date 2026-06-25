// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

#if canImport(CoreAI)
import CoreAI
#endif
@testable import CoreAILanguageModels

@Suite("Multimodal types")
struct MultimodalTypeTests {
    #if canImport(CoreAI)
    @Test("EmbeddedInput wraps NDArray with positions")
    func embeddedInputBasics() {
        let embeddings = NDArray(
            shape: [1, 256, 2048],
            scalarType: .float16
        )
        let input = EmbeddedInput(
            embeddings: embeddings,
            imageTokenPositions: 5..<261
        )
        #expect(input.tokenCount == 256)
        #expect(input.imageTokenPositions.count == 256)
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
}
