// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Chunking Configuration", .serialized)
struct ChunkingConfigTests {
    @Test("ModelConfig defaults: memory-based chunkSize, threshold = 2x chunkSize")
    func modelConfigDefaults() throws {
        let config = makeTestConfig()
        #expect(config.prefillChunkSize >= 2048)
        #expect(config.chunkThreshold == config.prefillChunkSize * 2)
    }

    @Test("Init and applyChunkingOverrides set chunk params")
    func overrides() throws {
        let viaInit = ModelConfig(
            name: "test", tokenizer: "test", vocabSize: 1000,
            maxContextLength: 4096, serializedModel: ["test.aimodel"],
            function: "main", prefillChunkSize: 128, prefillChunkThreshold: 384
        )
        #expect(viaInit.prefillChunkSize == 128)
        #expect(viaInit.chunkThreshold == 384)

        var viaApply = makeTestConfig()
        viaApply.applyChunkingOverrides(prefillChunkSize: 64, prefillChunkThreshold: 256)
        #expect(viaApply.prefillChunkSize == 64)
        #expect(viaApply.chunkThreshold == 256)
    }

    @Test("Override of 0 or negative falls through to default")
    func invalidOverrideFallsThrough() throws {
        var config = makeTestConfig()
        config.applyChunkingOverrides(prefillChunkSize: 0, prefillChunkThreshold: -1)
        #expect(config.prefillChunkSize >= 2048)
        #expect(config.chunkThreshold == config.prefillChunkSize * 2)
    }

    @Test("LanguageConfig decodes prefill_chunk_size from JSON")
    func languageConfigDecoding() throws {
        let json = """
            {
                "tokenizer": "test/model",
                "vocab_size": 32000,
                "max_context_length": 4096,
                "embedded_tokenizer": true,
                "prefill_chunk_size": 256,
                "prefill_chunk_threshold": 512
            }
            """
        let config = try JSONDecoder().decode(LanguageConfig.self, from: Data(json.utf8))
        #expect(config.prefillChunkSize == 256)
        #expect(config.prefillChunkThreshold == 512)
    }

    @Test("Codable round-trip does not persist chunking overrides")
    func codableRoundTrip() throws {
        var original = makeTestConfig()
        original.applyChunkingOverrides(prefillChunkSize: 64, prefillChunkThreshold: 128)

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ModelConfig.self, from: data)

        #expect(decoded.prefillChunkSize >= 2048)
        #expect(decoded.chunkThreshold == decoded.prefillChunkSize * 2)
    }

    @Test("Deprecated env var sets chunkSize and threshold, override beats it")
    func deprecatedEnvVar() throws {
        setenv("COREAI_CHUNK_THRESHOLD", "128", 1)
        defer { unsetenv("COREAI_CHUNK_THRESHOLD") }

        let config = makeTestConfig()
        #expect(config.prefillChunkSize == 128)
        #expect(config.chunkThreshold == 128)

        var overridden = makeTestConfig()
        overridden.applyChunkingOverrides(prefillChunkSize: 256, prefillChunkThreshold: 512)
        #expect(overridden.prefillChunkSize == 256)
        #expect(overridden.chunkThreshold == 512)
    }

    @Test("defaultPrefillChunkSize returns a power-of-two >= 2048")
    func memoryBasedDefault() {
        let size = defaultPrefillChunkSize()
        #expect(size >= 2048)
        #expect(size & (size - 1) == 0)
    }

    @Test("VLMModelConfig surfaces chunking overrides applied to its base config")
    func vlmConfigForwardsOverrides() throws {
        // The VLM engine applies EngineOptions overrides onto `base` and reconstructs
        // the VLMModelConfig; its prefill decision then reads config.prefillChunkSize /
        // config.chunkThreshold. Verify those overrides surface through the wrapper.
        var base = makeTestConfig()
        base.applyChunkingOverrides(prefillChunkSize: 128, prefillChunkThreshold: 384)
        let vlm = VLMModelConfig(base: base, visionConfig: makeTestVisionConfig())
        #expect(vlm.prefillChunkSize == 128)
        #expect(vlm.chunkThreshold == 384)

        // Absent overrides fall through to the memory-based default (2x threshold).
        let plain = VLMModelConfig(base: makeTestConfig(), visionConfig: makeTestVisionConfig())
        #expect(plain.prefillChunkSize >= 2048)
        #expect(plain.chunkThreshold == plain.prefillChunkSize * 2)
    }

    // MARK: - Helpers

    private func makeTestConfig() -> ModelConfig {
        ModelConfig(
            name: "test", tokenizer: "test", vocabSize: 1000,
            maxContextLength: 4096, serializedModel: ["test.aimodel"],
            function: "main"
        )
    }

    private func makeTestVisionConfig() -> VisionConfig {
        VisionConfig(
            imageSize: 224, patchSize: 14, imageTokenCount: 256, imageTokenId: 0
        )
    }
}
