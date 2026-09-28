// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Unified Config Parsing (Core AI + static-shape formats)

@Suite("Unified ModelConfigHandler")
struct UnifiedConfigHandlerTests {
    // MARK: - Parsing both formats through single handler

    @Test("Parses CoreAI-format JSON (with model_definition)")
    func parseCoreAIFormat() throws {
        let json = """
            {
                "name": "Qwen3-0.6B-4bit",
                "engine": "coreai",
                "tokenizer": "Qwen/Qwen3-0.6B",
                "vocab_size": 151936,
                "max_context_length": 4096,
                "function": "main",
                "source": { "model_definition": "torch", "hf_model_id": "Qwen/Qwen3-0.6B" },
                "serialized_model": ["qwen3_0_6b_4bit.aimodel"]
            }
            """
        let config = try ModelConfig(parsing: Data(json.utf8))

        #expect(config.source?.modelDefinition == .pyTorch)
        #expect(config.inputMode == nil)
    }

    @Test("Parses static-shape format JSON (no model_definition)")
    func parseStaticShapeFormat() throws {
        let json = """
            {
                "name": "Qwen3-0.6B-static",
                "engine": "coreai",
                "tokenizer": "Qwen/Qwen3-0.6B",
                "vocab_size": 151936,
                "max_context_length": 4096,
                "function": "main",
                "input_mode": "all-zeros",
                "source": { "hf_model_id": "Qwen/Qwen3-0.6B" },
                "serialized_model": ["qwen3_0_6b.aimodel"]
            }
            """
        let config = try ModelConfig(parsing: Data(json.utf8))

        #expect(config.source?.modelDefinition == nil)
        #expect(config.inputMode == .allZeros)
    }

    // MARK: - Resolved accessors

    @Test("resolvedModelDefinition defaults to .pyTorch when nil")
    func resolvedModelDefinitionDefault() {
        let config = ModelConfig(
            name: "t", tokenizer: "t",
            vocabSize: 100, maxContextLength: 512,
            source: ModelSource(hfModelId: "t"),
            serializedModel: ["t.aimodel"],
            function: "main"
        )
        #expect(config.resolvedModelDefinition == .pyTorch)
    }

    // MARK: - Codable round-trip

    @Test("Config survives encode → decode round-trip")
    func codableRoundtrip() throws {
        let original = ModelConfig(
            name: "roundtrip",
            tokenizer: "test/tokenizer",
            vocabSize: 32000,
            maxContextLength: 2048,
            source: ModelSource(hfModelId: "test/model", modelDefinition: .pyTorch),
            serializedModel: ["model.aimodel"],
            function: "main",
            inputMode: .random
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ModelConfig.self, from: data)

        #expect(decoded.name == original.name)
        #expect(decoded.inputMode == original.inputMode)
        #expect(decoded.source?.modelDefinition == original.source?.modelDefinition)
    }
}
