// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels
@testable import CoreAIShared

@Suite("LanguageModelBundle")
struct LanguageModelBundleTests {
    private static func tempBundle(_ metadata: String, named name: String = "test") throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "LanguageModelBundleTests-\(UUID().uuidString)/\(name)"
        )
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try metadata.write(
            to: dir.appending(path: "metadata.json"),
            atomically: true, encoding: .utf8
        )
        return dir
    }

    @Test("0.2 LLM bundle decodes language config + assets")
    func decodes02LLM() throws {
        let url = try Self.tempBundle(
            """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "qwen3-0.6b",
              "assets": { "main": "model.aimodel" },
              "language": {
                "tokenizer": "Qwen/Qwen3-0.6B",
                "vocab_size": 151936,
                "max_context_length": 8192,
                "function_map": { "main": ["main"], "extend": ["extend_512", "extend_1024"] }
              }
            }
            """)
        let bundle = try LanguageModelBundle(at: url)
        #expect(bundle.tokenizer == "Qwen/Qwen3-0.6B")
        #expect(bundle.vocabSize == 151936)
        #expect(bundle.maxContextLength == 8192)
        #expect(bundle.modelAssetPath == "model.aimodel")
        #expect(bundle.language.embeddedTokenizer == true)  // default
        #expect(bundle.language.functionMap?.names(for: "extend") == ["extend_512", "extend_1024"])
        #expect(bundle.language.functionMap?.name(for: "main") == "main")
    }

    @Test("0.1 legacy bundle throws unsupportedVersion")
    func legacy01Throws() throws {
        let url = try Self.tempBundle(
            """
            {
              "name": "qwen3_0_6b_4bit",
              "engine": "legacy",
              "tokenizer": "Qwen/Qwen3-0.6B",
              "vocab_size": 151936,
              "max_context_length": 8192,
              "function": "main",
              "serialized_model": ["model.aimodel"],
              "embedded_tokenizer": true
            }
            """)
        #expect(throws: ModelBundle.BundleError.self) {
            _ = try LanguageModelBundle(at: url)
        }
    }

    @Test("Wrong kind throws kindMismatch")
    func wrongKindThrows() throws {
        let url = try Self.tempBundle(
            """
            {
              "metadata_version": "0.2",
              "kind": "diffusion",
              "name": "sd-1.5",
              "assets": { "vae_decoder": "vae.aimodel" }
            }
            """)
        let bundle = try ModelBundle(at: url)
        #expect(throws: ModelBundle.BundleError.self) {
            _ = try LanguageModelBundle(bundle: bundle)
        }
    }

    @Test("Extension property bundle.language returns nil for non-LLM kind")
    func extensionPropertyNilForNonLLM() throws {
        let url = try Self.tempBundle(
            """
            {
              "metadata_version": "0.2",
              "kind": "diffusion",
              "name": "sd-1.5",
              "assets": { "vae_decoder": "vae.aimodel" }
            }
            """)
        let bundle = try ModelBundle(at: url)
        #expect(bundle.language == nil)
    }

    @Test("function_map omitted decodes to nil (runtime falls back to probing)")
    func functionMapOptional() throws {
        let url = try Self.tempBundle(
            """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "minimal",
              "assets": { "main": "model.aimodel" },
              "language": {
                "tokenizer": "x/y",
                "vocab_size": 100,
                "max_context_length": 512
              }
            }
            """)
        let bundle = try LanguageModelBundle(at: url)
        #expect(bundle.language.functionMap == nil)
    }

    @Test("user_data round-trips as [String: String]")
    func userDataRoundTrip() throws {
        let url = try Self.tempBundle(
            """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "minimal",
              "assets": { "main": "model.aimodel" },
              "language": {
                "tokenizer": "x/y",
                "vocab_size": 100,
                "max_context_length": 512
              },
              "user_data": {
                "exported_by": "ci",
                "git_sha": "abc1234",
                "tags": "gold,recommended"
              }
            }
            """)
        let bundle = try LanguageModelBundle(at: url)
        #expect(bundle.modelBundle.userData?["exported_by"] == "ci")
        #expect(bundle.modelBundle.userData?["git_sha"] == "abc1234")
        #expect(bundle.modelBundle.userData?["tags"] == "gold,recommended")
    }
}
