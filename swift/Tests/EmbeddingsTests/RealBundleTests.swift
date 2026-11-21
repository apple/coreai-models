// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Testing

@testable import CoreAIEmbeddings

/// Opt-in: runs exported bundles. `COREAI_TEST_EMBEDDING_BUNDLES=<dir>` holds bundle directories
/// named as the exporter names them (`clip-vit-b32`, `embeddinggemma-2-ios`, …); missing ones are
/// skipped.
@Suite(
    "Real embedding bundles (opt-in)",
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_TEST_EMBEDDING_BUNDLES"] != nil),
    .serialized
)
struct RealBundleTests {
    static let root = ProcessInfo.processInfo.environment["COREAI_TEST_EMBEDDING_BUNDLES"].map {
        URL(fileURLWithPath: $0)
    }

    static let captions = ["a plain red square", "a black and white checkerboard", "a green circle on white"]

    static func images() throws -> [CGImage] {
        [
            try Fixtures.image(width: 320, height: 240) { _, _ in (220, 20, 20) },
            try Fixtures.image(width: 256, height: 256) { x, y in
                (x / 32 + y / 32) % 2 == 0 ? (0, 0, 0) : (255, 255, 255)
            },
            try Fixtures.image(width: 256, height: 256) { x, y in
                (x - 128) * (x - 128) + (y - 128) * (y - 128) < 80 * 80 ? (30, 160, 40) : (255, 255, 255)
            },
        ]
    }

    static func model(_ name: String) async throws -> CoreAIEmbeddingModel? {
        guard let url = root?.appending(path: name), FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return try await CoreAIEmbeddingModel(resourcesAt: url)
    }

    /// Each caption scores its own image highest; text-only bundles embed each caption differently.
    @Test(arguments: [
        "clip-vit-b32", "clip-vit-b32-fused", "clip-vit-b32-ios", "clip-vit-b32-text", "clap-htsat",
        "clap-htsat-text", "embeddinggemma-2", "embeddinggemma-2-ios", "qwen3-embedding-0.6b",
    ])
    func captionsMatchImages(_ name: String) async throws {
        guard var model = try await Self.model(name) else { return }
        let texts = try await model.embed(texts: Self.captions)
        guard model.modalities.contains(.image) else {
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                let similarity = try texts[i].cosine(texts[j])
                #expect(similarity < 0.99, "\(name) captions \(i), \(j): \(similarity)")
            }
            return
        }
        let images = try await model.embed(images: Self.images())
        for (i, text) in texts.enumerated() {
            let scores = try images.map(text.cosine)
            #expect(scores.firstIndex(of: scores.max()!) == i, "\(name) caption \(i): \(scores)")
        }
    }

    @Test func gpuAndNeuralEngineBundlesAgree() async throws {
        guard var gpu = try await Self.model("embeddinggemma-2"),
            var neuralEngine = try await Self.model("embeddinggemma-2-ios")
        else { return }
        let images = try Self.images()
        let a = try await gpu.embed(texts: Self.captions) + gpu.embed(images: images)
        let b = try await neuralEngine.embed(texts: Self.captions) + neuralEngine.embed(images: images)
        for (x, y) in zip(a, b) {
            let similarity = try x.cosine(y)
            #expect(similarity > 0.999, "\(similarity)")
        }
    }
}
