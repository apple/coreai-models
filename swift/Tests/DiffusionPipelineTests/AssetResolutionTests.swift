// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIDiffusionPipeline
import Foundation
import Testing

/// Asset lookup must honour the filenames `metadata.json` declares, so bundles whose assets
/// were compiled ahead of time to `.aimodelc` load.
@Suite("Diffusion asset resolution")
struct AssetResolutionTests {
    /// A FLUX.2 bundle directory declaring `assets`, with an empty entry on disk for each of
    /// `files`. Lookup only checks existence.
    private static func makeBundle(assets: [String: String], files: [String]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("assets_\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in files {
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(file), withIntermediateDirectories: true)
        }
        let metadata: [String: Any] = [
            "metadata_version": "0.2",
            "kind": "diffusion",
            "name": "asset-resolution-test",
            "assets": assets,
            "diffusion": ["type": "flux2", "prediction_type": "flow_matching"],
        ]
        try JSONSerialization.data(withJSONObject: metadata)
            .write(to: dir.appendingPathComponent("metadata.json"))
        return dir
    }

    @Test(
        "A declared filename is used as is, even when missing; else the exporter's default",
        arguments: [
            (declared: true, onDisk: true, expected: "Transformer.aimodelc"),
            (declared: true, onDisk: false, expected: "Transformer.aimodelc"),
            (declared: false, onDisk: false, expected: "Transformer.aimodel"),
        ])
    func declaredFilenameWithFallback(declared: Bool, onDisk: Bool, expected: String) throws {
        let dir = try Self.makeBundle(
            assets: declared ? ["transformer": "Transformer.aimodelc"] : [:],
            files: onDisk ? ["Transformer.aimodelc"] : [])
        defer { try? FileManager.default.removeItem(at: dir) }

        let bundle = try DiffusionBundle(at: dir)
        #expect(
            bundle.assetFilename(for: DiffusionComponentKey.transformer, fallback: "Transformer.aimodel")
                == expected)
    }

    @Test(
        "FLUX.2 refuses a declared asset that is missing instead of falling back to .aimodel",
        arguments: [
            // Required: the default-named file is present but must not be used.
            (key: "transformer", declared: "Transformer.aimodelc", onDisk: ["Transformer.aimodel"]),
            // Optional variant: declared but absent is an error, not "variant not exported".
            (key: "transformer_512", declared: "Transformer_512.aimodelc", onDisk: [String]()),
        ])
    func declaredMissingAssetThrows(key: String, declared: String, onDisk: [String]) async throws {
        let dir = try Self.makeBundle(assets: [key: declared], files: onDisk)
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try await FlowTransformerPipeline(from: dir)
            Issue.record("loaded a bundle whose declared \(declared) is missing")
        } catch PipelineLoadError.missingComponent(let detail) {
            #expect(detail == "\(key) (expected \(declared))")
        }
    }
}
