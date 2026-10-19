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
    @Test(
        "A declared filename is used when it exists, else the exporter's default",
        arguments: [
            (declared: true, onDisk: true, expected: "Transformer.aimodelc"),
            (declared: true, onDisk: false, expected: "Transformer.aimodel"),
            (declared: false, onDisk: false, expected: "Transformer.aimodel"),
        ])
    func declaredFilenameWithFallback(declared: Bool, onDisk: Bool, expected: String) throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("assets_\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        if onDisk {
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent("Transformer.aimodelc"), withIntermediateDirectories: true)
        }
        let metadata: [String: Any] = [
            "metadata_version": "0.2",
            "kind": "diffusion",
            "name": "asset-resolution-test",
            "assets": declared ? ["transformer": "Transformer.aimodelc"] : [:],
            "diffusion": ["type": "flux2", "prediction_type": "flow_matching"],
        ]
        try JSONSerialization.data(withJSONObject: metadata)
            .write(to: dir.appendingPathComponent("metadata.json"))

        let bundle = try DiffusionBundle(at: dir)
        #expect(
            bundle.assetFilename(for: DiffusionComponentKey.transformer, fallback: "Transformer.aimodel")
                == expected)
    }
}
