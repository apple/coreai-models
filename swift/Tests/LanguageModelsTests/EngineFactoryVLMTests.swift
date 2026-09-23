// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels
@testable import CoreAIShared

@Suite("EngineFactory VLM construction")
struct EngineFactoryVLMTests {
    /// Write a metadata.json (and nothing else) into a temp bundle dir and return its URL.
    /// `requireModelURL` only needs the asset key present, so no real `.aimodel` files are needed
    /// to exercise config assembly.
    private static func tempBundle(_ metadata: String, named name: String = "vlm") throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "EngineFactoryVLMTests-\(UUID().uuidString)/\(name)"
        )
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try metadata.write(
            to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        return dir
    }

    private static let vlmMetadata = """
        {
          "metadata_version": "0.2",
          "kind": "vlm",
          "name": "qwen3-vl-2b",
          "assets": {
            "main": "main.aimodel",
            "vision": "vision.aimodel",
            "embedding": "embed.aimodel"
          },
          "language": {
            "tokenizer": "Qwen/Qwen3-VL-2B-Instruct",
            "vocab_size": 151936,
            "max_context_length": 8192,
            "function_map": { "main": ["main"] }
          },
          "vision": {
            "image_size": 448,
            "patch_size": 16,
            "image_token_count": 256,
            "image_token_id": 151655
          }
        }
        """

    @Test("makeVLMConfig assembles base + vision from a VLM bundle")
    func assemblesConfig() throws {
        let url = try Self.tempBundle(Self.vlmMetadata)
        let bundle = try LanguageBundle(at: url)
        let mainURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        let config = try EngineFactory.makeVLMConfig(
            bundle: bundle, languageModelURL: mainURL, options: EngineOptions())

        #expect(config.name == "qwen3-vl-2b")
        #expect(config.base.tokenizer == "Qwen/Qwen3-VL-2B-Instruct")
        #expect(config.vocabSize == 151936)
        #expect(config.maxContextLength == 8192)
        #expect(config.function == "main")
        #expect(config.base.serializedModel == [mainURL.path])
        // Vision config surfaces from the bundle.
        #expect(config.visionConfig.imageSize == 448)
        #expect(config.visionConfig.imageTokenId == 151655)
        #expect(config.visionConfig.imageTokenCount == 256)
    }

    @Test("makeVLMConfig applies chunking overrides to the base config")
    func appliesChunkingOverrides() throws {
        let url = try Self.tempBundle(Self.vlmMetadata)
        let bundle = try LanguageBundle(at: url)
        let mainURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        let options = EngineOptions(prefillChunkSize: 128, prefillChunkThreshold: 256)
        let config = try EngineFactory.makeVLMConfig(
            bundle: bundle, languageModelURL: mainURL, options: options)

        // Overrides land on `base` and surface through VLMModelConfig's forwarding accessors.
        #expect(config.base.prefillChunkSize == 128)
        #expect(config.base.chunkThreshold == 256)
        #expect(config.prefillChunkSize == 128)
        #expect(config.chunkThreshold == 256)
    }
}
