// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIShared

@Suite("ModelBundle")
struct ModelBundleTests {
    private static func tempBundle(
        _ metadata: String,
        named name: String = "test"
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "ModelBundleTests-\(UUID().uuidString)/\(name)"
        )
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try metadata.write(
            to: dir.appending(path: "metadata.json"),
            atomically: true, encoding: .utf8
        )
        return dir
    }

    @Test("0.2 schema decodes common fields")
    func decodes02CommonFields() throws {
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
                "max_context_length": 8192
              },
              "user_data": { "git_sha": "abc1234" }
            }
            """)
        let bundle = try ModelBundle(at: url)
        #expect(bundle.metadataVersion == "0.2")
        #expect(bundle.kind == .llm)
        #expect(bundle.name == "qwen3-0.6b")
        #expect(bundle.userData == ["git_sha": "abc1234"])
    }

    @Test("0.1 legacy throws unsupportedVersion")
    func legacy01Throws() throws {
        let url = try Self.tempBundle(
            """
            {
              "name": "legacy-model",
              "engine": "legacy",
              "tokenizer": "Qwen/Qwen3-0.6B",
              "vocab_size": 151936,
              "max_context_length": 8192,
              "function": "main",
              "serialized_model": ["model.aimodel"]
            }
            """, named: "legacy")
        #expect(throws: ModelBundle.BundleError.self) {
            _ = try ModelBundle(at: url)
        }
    }

    @Test("Missing metadata.json throws")
    func missingMetadataThrows() throws {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "ModelBundleTests-empty-\(UUID().uuidString)"
        )
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        #expect(throws: ModelBundle.BundleError.self) {
            _ = try ModelBundle(at: dir)
        }
    }

    @Test("Pointing at a .aimodelc asset throws pointedAtModelAsset, not a parse error")
    func pointedAtCompiledAssetThrows() throws {
        // A compiled `.aimodelc` is a directory with its own unrelated
        // metadata.json. Pointing the tool at it must fail fast with guidance,
        // not parse that inner metadata as a bogus 0.1 bundle.
        let bundleDir = FileManager.default.temporaryDirectory.appending(
            path: "ModelBundleTests-\(UUID().uuidString)"
        )
        let asset = bundleDir.appending(path: "model.aimodelc")
        try FileManager.default.createDirectory(at: asset, withIntermediateDirectories: true)
        try """
        { "producer": "coreai-build", "assetVersion": "2.0" }
        """.write(
            to: asset.appending(path: "metadata.json"), atomically: true, encoding: .utf8)

        let error = #expect(throws: ModelBundle.BundleError.self) {
            _ = try ModelBundle(at: asset)
        }
        guard case .pointedAtModelAsset = error else {
            Issue.record("expected pointedAtModelAsset, got \(String(describing: error))")
            return
        }
        #expect(String(describing: error).contains("model.aimodelc"))
    }

    @Test("Pointing at a .aimodel asset throws pointedAtModelAsset")
    func pointedAtUncompiledAssetThrows() throws {
        let error = #expect(throws: ModelBundle.BundleError.self) {
            _ = try ModelBundle(from: "/some/where/model.aimodel")
        }
        guard case .pointedAtModelAsset = error else {
            Issue.record("expected pointedAtModelAsset, got \(String(describing: error))")
            return
        }
    }
}
