// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

@Suite("Multi-output model function")
struct MultiOutputModelFunctionTests {
    // MARK: - Error surface (no asset needed)

    @Test("expectedSingleOutput description lists all output names and points at predictAllOutputs")
    func expectedSingleOutputDescription() {
        let err = CoreAIDiffusionError.expectedSingleOutput(got: ["hidden_embeds", "pooled_outputs"])
        let msg = err.errorDescription ?? ""
        #expect(msg.contains("2 outputs"))
        #expect(msg.contains("hidden_embeds"))
        #expect(msg.contains("pooled_outputs"))
        #expect(msg.contains("predictAllOutputs"))
    }

    // MARK: - Real-asset integration (skipped when exports aren't present)

    @Test(
        "CLIP-L asset reports 2 outputs (hidden + pooled) via predictAllOutputs",
        .enabled(if: Self.clipLAssetURL() != nil))
    func clipLMultiOutput() async throws {
        guard let url = Self.clipLAssetURL() else { return }
        let fn = CoreAIDiffusionModelFunction(modelURL: url)
        try await fn.loadResources()

        let descs = try await fn.outputDescriptors
        #expect(descs.count == 2, "CLIP-L should emit (last_hidden_state, pooled_output)")

        // Build a dummy int32 input_ids tensor
        var ids = NDArray(shape: [1, 77], scalarType: .int32)
        var view = ids.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<77 { ptr[i] = 0 }
        }
        let inputDescs = try await fn.inputDescriptors
        guard let inputName = inputDescs.keys.first else {
            Issue.record("CLIP-L had no input descriptors")
            return
        }

        let outputs = try await fn.predictAllOutputs(inputs: [inputName: ids])
        #expect(outputs.count == 2)

        // One output is rank-3 (hidden), one is rank-2 (pooled).
        var ranks = [Int]()
        for (name, _) in outputs {
            if let r = descs[name]?.shape.count { ranks.append(r) }
        }
        #expect(ranks.sorted() == [2, 3])
    }

    @Test(
        "predict(inputs:) throws expectedSingleOutput for a CLIP-L asset",
        .enabled(if: Self.clipLAssetURL() != nil))
    func predictRejectsMultiOutputAsset() async throws {
        guard let url = Self.clipLAssetURL() else { return }
        let fn = CoreAIDiffusionModelFunction(modelURL: url)
        try await fn.loadResources()

        var ids = NDArray(shape: [1, 77], scalarType: .int32)
        var view = ids.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<77 { ptr[i] = 0 }
        }
        let inputDescs = try await fn.inputDescriptors
        guard let inputName = inputDescs.keys.first else {
            Issue.record("CLIP-L had no input descriptors")
            return
        }

        await #expect(throws: CoreAIDiffusionError.self) {
            _ = try await fn.predict(inputs: [inputName: ids])
        }
    }

    @Test(
        "CoreAITextEncoder populates pooledOutput for a CLIP-L asset",
        .enabled(if: Self.clipLAssetURL() != nil))
    func textEncoderPooledOutputPresent() async throws {
        guard let url = Self.clipLAssetURL() else { return }
        let fn = CoreAIDiffusionModelFunction(modelURL: url)
        try await fn.loadResources()

        let encoder = CoreAITextEncoder(
            function: fn,
            tokenize: { _ in [Int32](repeating: 0, count: 77) },
            maxLength: 77)

        let output = try await encoder.encode("anything")
        #expect(output.pooledOutput != nil, "CLIP-L wrapper exports a pooled output")
        if let pooled = output.pooledOutput {
            #expect(pooled.shape.last == 768)
        }
        #expect(output.hiddenStates.shape == [1, 77, 768])
    }

    // MARK: - Asset location

    private static func clipLAssetURL() -> URL? {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // DiffusionPipelineTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // swift/
            .deletingLastPathComponent()  // repo root
        let candidate =
            repoRoot
            .appendingPathComponent("exports/FLUX.2-klein-4B/TextEncoder.aimodel")
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }
}
