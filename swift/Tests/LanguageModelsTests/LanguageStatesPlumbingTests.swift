// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels
@testable import CoreAIShared

/// `language.states` must reach the engine config: through `ModelConfig(bundle:)` and the JSON
/// round trip `EngineFactory.createEngine` performs, and through the VLM config.
@Suite("language.states reaches the engine config")
struct LanguageStatesPlumbingTests {
    private static let expected: [String: StateKind] = [
        "keyCache": .kvCache, "valueCache": .kvCache,
        "convStates": .fixed, "recurrentStates": .fixed,
    ]

    private static func bundle(
        kind: String, assets: String = #""main": "model.aimodel""#, extra: String = ""
    ) throws -> LanguageModelBundle {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "LanguageStatesPlumbingTests-\(UUID().uuidString)/bundle")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let metadata = """
            {
              "metadata_version": "0.2",
              "kind": "\(kind)",
              "name": "hybrid-test",
              "assets": { \(assets) },
              "language": {
                "states": {
                  "keyCache": "kv_cache", "valueCache": "kv_cache",
                  "convStates": "fixed", "recurrentStates": "fixed"
                },
                "tokenizer": "example/hybrid-test",
                "vocab_size": 100,
                "max_context_length": 512
              }\(extra)
            }
            """
        try metadata.write(
            to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        return try LanguageModelBundle(at: dir)
    }

    @Test("states survive ModelConfig(bundle:) and the JSON round trip")
    func statesSurviveRoundTrip() throws {
        let config = ModelConfig(bundle: try Self.bundle(kind: "llm"))
        #expect(config.states == Self.expected)
        let data = try JSONEncoder().encode(config)
        #expect(try ModelConfig(parsing: data).states == Self.expected)
    }

    @Test("makeVLMConfig carries language.states into the base config")
    func vlmConfigCarriesStates() throws {
        let bundle = try Self.bundle(
            kind: "vlm",
            assets: #""main": "main.aimodel", "vision": "vision.aimodel", "embedding": "embed.aimodel""#,
            extra: """
                ,
                  "vision": { "image_size": 448, "patch_size": 16, "image_token_count": 256, "image_token_id": 1 }
                """)
        let mainURL = try bundle.modelBundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        let config = try EngineFactory.makeVLMConfig(
            bundle: bundle, languageModelURL: mainURL, options: EngineOptions())
        #expect(config.base.states == Self.expected)
    }
}
