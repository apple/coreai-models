// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Testing

@testable import CoreAIImageSegmenter

@Suite("ImageSegmentationBundle")
struct ImageSegmentationBundleTests {
    /// A bundle directory with `metadata.json` declaring `kind` and `assets.main`. The asset
    /// file itself is only written when `writeAsset` is set.
    private static func tempBundle(kind: String = "segmenter", writeAsset: Bool = true) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "ImageSegmentationBundleTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try """
        {
          "metadata_version": "0.2",
          "kind": "\(kind)",
          "name": "sam3",
          "assets": { "main": "sam3.aimodel" }
        }
        """.write(to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        if writeAsset {
            let assetDir = dir.appending(path: "sam3.aimodel")
            try FileManager.default.createDirectory(at: assetDir, withIntermediateDirectories: true)
            // `AIModelAsset.isValid` requires a source program file to recognize a .aimodel as valid.
            FileManager.default.createFile(atPath: assetDir.appending(path: "main.mlirb").path, contents: nil)
        }
        return dir
    }

    @Test("Resolves the main asset and tokenizer folder")
    func resolvesPaths() throws {
        let dir = try Self.tempBundle()
        let bundle = try ImageSegmentationBundle(from: dir.path)
        #expect(bundle.modelURL.lastPathComponent == "sam3.aimodel")
        #expect(bundle.tokenizerFolder.lastPathComponent == "tokenizer")
        #expect(bundle.modelBundle.name == "sam3")
    }

    @Test("A non-segmenter bundle throws kindMismatch")
    func rejectsWrongKind() throws {
        let dir = try Self.tempBundle(kind: "video_segmenter")
        let error = #expect(throws: ModelBundle.BundleError.self) {
            _ = try ImageSegmentationBundle(from: dir.path)
        }
        guard case .kindMismatch(expected: .segmenter, got: .videoSegmenter) = error else {
            Issue.record("expected kindMismatch, got \(String(describing: error))")
            return
        }
    }

    @Test("A declared asset missing on disk throws missingAsset")
    func rejectsMissingAsset() throws {
        let dir = try Self.tempBundle(writeAsset: false)
        let error = #expect(throws: ModelBundle.BundleError.self) {
            _ = try ImageSegmentationBundle(from: dir.path)
        }
        guard case .missingAsset(key: "main", _) = error else {
            Issue.record("expected missingAsset, got \(String(describing: error))")
            return
        }
    }
}
