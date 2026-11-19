// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

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

    @Test("VisionConfig decodes a Gemma4-style size ladder")
    func visionConfigWithImageSizes() throws {
        let json = """
            {
                "image_size": 1152,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "image_token_count": 576,
                "image_token_id": 258880,
                "boi_token_id": 255999,
                "eoi_token_id": 258882,
                "pad_token_id": 0,
                "image_sizes": [384, 768, 1152],
                "token_merge": "decoder_gather"
            }
            """
        let config = try JSONDecoder().decode(VisionConfig.self, from: json.data(using: .utf8)!)
        #expect(config.poolingKernelSize == 3)
        #expect(config.padTokenId == 0)
        #expect(config.imageSizes == [384, 768, 1152])
        #expect(config.tokenMerge == .decoderGather)
        #expect(config.blockPixels == 48)
        #expect(config.pooledGrid(imageSize: 1152) == 24)
        #expect(config.softTokenCount(imageSize: 384) == 64)
        #expect(VisionConfig.encoderFunctionName(imageSize: 768) == "encode_image_768")
    }

    @Test("VisionConfig backwards compatible without a size list (Qwen-style)")
    func visionConfigWithoutImageSizes() throws {
        let json = """
            {"image_size": 896, "patch_size": 14, "image_token_count": 256, "image_token_id": 255999}
            """
        let config = try JSONDecoder().decode(VisionConfig.self, from: json.data(using: .utf8)!)
        #expect(config.imageSizes == nil)
        #expect(config.tokenMerge == .separateEmbedding)
        #expect(config.poolingKernelSize == nil)
    }
}
