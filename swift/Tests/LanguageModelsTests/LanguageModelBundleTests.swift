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
        #expect(bundle.maxBatchSize == 1)  // default when max_batch_size is absent
    }

    @Test("max_batch_size decodes from the language block")
    func decodesMaxBatchSize() throws {
        let url = try Self.tempBundle(
            """
            {
              "metadata_version": "0.3",
              "kind": "llm",
              "name": "qwen3-1.7b-batched",
              "assets": { "main": "model.aimodel" },
              "language": {
                "tokenizer": "Qwen/Qwen3-1.7B",
                "vocab_size": 151936,
                "max_context_length": 8192,
                "max_batch_size": 8
              }
            }
            """)
        let bundle = try LanguageModelBundle(at: url)
        #expect(bundle.maxBatchSize == 8)
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

    private static func drafterBundle(
        assets: String, auxiliary: String, overrides: String = "", speculative: String = ""
    ) throws -> URL {
        try tempBundle(
            """
            {
              "metadata_version": "0.2",
              "kind": "llm",
              "name": "gemma",
              "assets": { "main": "model.aimodel"\(assets) },
              "auxiliary_assets": { \(auxiliary) },
              \(speculative)
              "language": {
                "tokenizer": "x/y",
                "vocab_size": 100,
                "max_context_length": 512\(overrides)
              }
            }
            """)
    }

    private static let drafterAssets = ", \"drafter\": \"draft_step.aimodel\""
    private static let drafterSidecars = """
        "drafter_config": "draft_step.json", "drafter_scan": "draft_scan.bin",
        "drafter_scan_index": "draft_scan.json"
        """

    @Test("A bundle without a drafter component has no draft head and no hidden states")
    func noDrafter() throws {
        let bundle = try LanguageModelBundle(at: try Self.drafterBundle(assets: "", auxiliary: ""))
        #expect(try bundle.drafter == nil)
        #expect(!bundle.emitsHiddenStates)
    }

    @Test("The drafter component resolves the program and its sidecars within the bundle")
    func drafterComponent() throws {
        let url = try Self.drafterBundle(
            assets: Self.drafterAssets, auxiliary: Self.drafterSidecars,
            overrides: ", \"overrides\": { \"emits_hidden_states\": true }",
            speculative: "\"speculative\": { \"drafter_kind\": \"mtp\" },")
        let bundle = try LanguageModelBundle(at: url)
        let drafter = try #require(try bundle.drafter)
        #expect(drafter == DrafterComponent(directory: url))
        #expect(bundle.emitsHiddenStates)
    }

    @Test("A drafter program listed without its sidecars is rejected, naming the missing one")
    func drafterWithoutSidecars() throws {
        let url = try Self.drafterBundle(
            assets: Self.drafterAssets,
            auxiliary: "\"drafter_config\": \"draft_step.json\", \"drafter_scan\": \"draft_scan.bin\"",
            speculative: "\"speculative\": { \"drafter_kind\": \"mtp\" },")
        let bundle = try LanguageModelBundle(at: url)
        #expect {
            _ = try bundle.drafter
        } throws: { error in
            "\(error)".contains("drafter_scan_index")
        }
    }

    @Test(
        "A drafter of another kind, or of none, is not read as a hidden-state draft head",
        arguments: ["\"speculative\": { \"drafter_kind\": \"block\" },", ""])
    func drafterOfAnotherKind(speculative: String) throws {
        let url = try Self.drafterBundle(
            assets: Self.drafterAssets, auxiliary: Self.drafterSidecars, speculative: speculative)
        let bundle = try LanguageModelBundle(at: url)
        #expect {
            _ = try bundle.drafter
        } throws: { error in
            "\(error)".contains("mtp")
        }
    }

    @Test("A bundle that declares no hidden states reports none")
    func noHiddenStatesByDefault() throws {
        let bundle = try LanguageModelBundle(
            at: try Self.drafterBundle(
                assets: "", auxiliary: "", overrides: ", \"overrides\": { \"sliding_window\": 512 }"))
        #expect(!bundle.emitsHiddenStates)
    }
}
