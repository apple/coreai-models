// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIDiffusionPipeline
import Foundation
import Testing

/// `metadata.json` is the only source of asset filenames: a declared asset must exist, and an
/// undeclared one is never picked up by its default name.
@Suite("Diffusion asset resolution")
struct AssetResolutionTests {
    /// A FLUX.2 bundle directory declaring `assets`, with an empty entry on disk for each of
    /// `files`.
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
        "An undeclared or missing asset is an error, never its default filename",
        arguments: [
            // Undeclared: the default-named file on disk is not picked up.
            (assets: [String: String](), detail: "transformer (not declared in metadata.json)"),
            // Declared but missing: no substitute.
            (assets: ["transformer": "Transformer.aimodelc"], detail: "transformer (expected Transformer.aimodelc)"),
        ])
    func declaredAssetsOnly(assets: [String: String], detail: String) throws {
        let dir = try Self.makeBundle(assets: assets, files: ["Transformer.aimodel"])
        defer { try? FileManager.default.removeItem(at: dir) }

        let bundle = try DiffusionBundle(at: dir)
        do {
            _ = try resolveDeclaredPipelineAsset(DiffusionComponentKey.transformer, in: bundle)
            Issue.record("resolved an asset that should have been rejected")
        } catch PipelineLoadError.missingComponent(let message) {
            #expect(message == detail)
        }
    }

    @Test(
        "FLUX.2 loads only the assets metadata.json declares",
        arguments: [
            // A declared required asset that is missing; the default-named file is ignored.
            (
                assets: ["transformer": "Transformer.aimodelc"], onDisk: ["Transformer.aimodel"],
                detail: "transformer (expected Transformer.aimodelc)"
            ),
            // A declared optional variant that is missing is an error, not "not exported".
            (
                assets: ["transformer_512": "Transformer_512.aimodelc"], onDisk: [String](),
                detail: "transformer_512 (expected Transformer_512.aimodelc)"
            ),
            // Default-named files that metadata.json doesn't declare don't count.
            (
                assets: [String: String](), onDisk: ["Transformer.aimodel", "VAEDecoder.aimodel"],
                detail: "No valid component combination declared in metadata.json. Need "
                    + "transformer+vae_decoder, transformer+vae_decoder_half, or transformer_512+vae_decoder_half."
            ),
        ])
    func fluxDeclaredAssetsOnly(assets: [String: String], onDisk: [String], detail: String) async throws {
        let dir = try Self.makeBundle(assets: assets, files: onDisk)
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try await FlowTransformerPipeline(from: dir)
            Issue.record("loaded a bundle that should have been rejected")
        } catch PipelineLoadError.missingComponent(let message) {
            #expect(message == detail)
        }
    }
}
