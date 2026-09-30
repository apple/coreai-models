// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels
@testable import CoreAIShared

/// Engine-support tests: the wiring between a bundle on disk and the engine the
/// factory builds from it.
///
/// What is *not* covered here: whether the input handlers actually attach, and
/// whether a decode step runs. Both need a real multi-gigabyte asset, because
/// `InferenceFunctionDescriptor` cannot be constructed outside the runtime. Those
/// belong in an integration test gated on a bundle path; everything below runs
/// from a temp directory with no asset at all.
@Suite("Engine support")
struct EngineSupportTests {
    // MARK: - Helpers

    /// Writes a bundle directory containing `metadata.json` plus any named
    /// sidecar files, and returns its URL.
    private static func bundle(
        metadata: String,
        files: [String] = []
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "EngineSupportTests-\(UUID().uuidString)/model"
        )
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try metadata.write(
            to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        for file in files {
            try Data("stub".utf8).write(to: dir.appending(path: file))
        }
        return dir
    }

    /// A Gemma-4-shaped bundle: sliding window, dual RoPE, soft cap, PLE sidecar.
    private static func gemmaMetadata(
        assets: String = #""main": "model.aimodel", "per_layer_embeddings": "model_ple.safetensors""#,
        extraLanguage: String = ""
    ) -> String {
        """
        {
          "metadata_version": "0.2",
          "kind": "llm",
          "name": "gemma_4_e2b_it_static",
          "assets": { \(assets) },
          "language": {
            "tokenizer": "google/gemma-4-E2B-it",
            "vocab_size": 262144,
            "max_context_length": 131072,
            "overrides": {
              "sliding_window": 512,
              "final_logit_softcapping": 30.0,
              "rope": {
                "sliding_head_dim": 256,
                "global_head_dim": 512,
                "sliding_rope_theta": 10000.0,
                "global_rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25
              }
            }\(extraLanguage)
          }
        }
        """
    }

    // MARK: - Sidecar resolution

    @Test("A declared PLE sidecar resolves through the assets role map")
    func perLayerEmbeddingsResolvesFromAssets() throws {
        let url = try Self.bundle(
            metadata: Self.gemmaMetadata(), files: ["model_ple.safetensors"])
        let resolved = try LanguageModelBundle(at: url).tensorData

        let key = EngineOptions.TensorDataKey.perLayerEmbeddings
        #expect(resolved[key]?.lastPathComponent == "model_ple.safetensors")
    }

    @Test("A bundle that ships no sidecar reports none")
    func noSidecarReportsEmpty() throws {
        let url = try Self.bundle(metadata: Self.gemmaMetadata(assets: #""main": "model.aimodel""#))
        let resolved = try LanguageModelBundle(at: url).tensorData

        // Absent rather than present-and-missing: the static engine distinguishes
        // the two, and reports "bundle ships no PLE table" rather than failing to
        // open a file it was told to expect.
        #expect(resolved[EngineOptions.TensorDataKey.perLayerEmbeddings] == nil)
    }

    // MARK: - Config the engine reads

    @Test("Gemma-shaped config survives the trip from metadata to the engine")
    func engineFacingConfigIsCarried() throws {
        let url = try Self.bundle(
            metadata: Self.gemmaMetadata(), files: ["model_ple.safetensors"])
        let bundle = try LanguageModelBundle(at: url)

        // Every field here drives a handler the static engine attaches.
        let overrides = try #require(bundle.overrides)
        #expect(overrides.slidingWindow == 512)
        #expect(overrides.finalLogitSoftcapping == 30.0)
        #expect(overrides.rope?.slidingHeadDim == 256)
        #expect(overrides.rope?.globalHeadDim == 512)
        #expect(overrides.rope?.partialRotaryFactor == 0.25)
    }

    @Test("A model with no overrides block leaves them nil")
    func plainModelCarriesNoGemmaConfig() throws {
        let url = try Self.bundle(
            metadata: """
                {
                  "metadata_version": "0.2",
                  "kind": "llm",
                  "name": "qwen3_0_6b_static",
                  "assets": { "main": "model.aimodel" },
                  "language": {
                    "tokenizer": "Qwen/Qwen3-0.6B",
                    "vocab_size": 151936,
                    "max_context_length": 4096
                  }
                }
                """)
        let bundle = try LanguageModelBundle(at: url)

        #expect(bundle.overrides == nil)
        #expect(bundle.tensorData.isEmpty)
    }

    @Test("A non-positive soft cap is rejected", arguments: ["0.0", "-30.0"])
    func nonPositiveSoftcapIsRejected(cap: String) {
        let json = #"{"final_logit_softcapping": "# + cap + "}"
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(LanguageOverrides.self, from: Data(json.utf8))
        }
    }

    @Test("ModelConfig carries the overrides through encode and decode")
    func modelConfigRoundTripsOverrides() throws {
        // EngineFactory encodes the ModelConfig it builds and the engine decodes it, so a
        // gap here would drop the overrides between bundle load and engine construction.
        let overrides = LanguageOverrides(
            slidingWindow: 512,
            rope: RoPEConfig(
                slidingHeadDim: 256, globalHeadDim: 512, slidingRopeTheta: 10_000,
                globalRopeTheta: 1_000_000, partialRotaryFactor: 0.25),
            finalLogitSoftcapping: 30.0)
        let config = ModelConfig(
            name: "gemma_4_e2b_it_static", tokenizer: "google/gemma-4-E2B-it", vocabSize: 262144,
            maxContextLength: 131072, serializedModel: ["model.aimodel"], function: "main",
            overrides: overrides)

        let decoded = try ModelConfig(parsing: try JSONEncoder().encode(config))
        #expect(decoded.overrides == overrides)
    }

    // MARK: - Variant resolution

    @Test("A chunked-static asset resolves to the static-shape engine")
    func chunkedStaticPicksStaticShape() throws {
        let variant = try EngineFactory.resolveVariant(
            override: nil, detectedStructure: .chunkedStatic(batchSize: 8))
        #expect(variant == .staticShape)
    }

    @Test("A dynamic asset resolves to the pipelined engine")
    func dynamicPicksPipelined() throws {
        let variant = try EngineFactory.resolveVariant(
            override: nil, detectedStructure: .dynamic)
        #expect(variant == .pipelined)
    }

    @Test("`auto` and `default` mean auto-detect", arguments: ["auto", "default"])
    func autoOverridesDetect(_ override: String) throws {
        let variant = try EngineFactory.resolveVariant(
            override: override, detectedStructure: .chunkedStatic(batchSize: 8))
        #expect(variant == .staticShape)
    }

    @Test("A compatible override is honoured")
    func compatibleOverrideHonoured() throws {
        let variant = try EngineFactory.resolveVariant(
            override: "coreai-sequential", detectedStructure: .dynamic)
        #expect(variant == .sequential)
    }

    @Test("Static-shape on a dynamic asset is rejected")
    func staticShapeOnDynamicRejected() {
        #expect(throws: InferenceRuntimeError.self) {
            try EngineFactory.resolveVariant(
                override: "static-shape", detectedStructure: .dynamic)
        }
    }

    @Test("A pipelined override on a static ladder is rejected")
    func pipelinedOnStaticRejected() {
        #expect(throws: InferenceRuntimeError.self) {
            try EngineFactory.resolveVariant(
                override: "coreai-pipelined", detectedStructure: .chunkedStatic(batchSize: 8))
        }
    }

    @Test("An unknown variant name is rejected")
    func unknownVariantRejected() {
        #expect(throws: InferenceRuntimeError.self) {
            try EngineFactory.resolveVariant(
                override: "not-an-engine", detectedStructure: .dynamic)
        }
    }
}
