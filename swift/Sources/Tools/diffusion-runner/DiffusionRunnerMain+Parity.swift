// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAI
import CoreAIDiffusionPipeline
import CoreAIShared
import Foundation

// Component-parity tracing for the hidden `--parity-test` flag. Not part of image generation and
// not model-specific beyond the FLUX.2 fixture layout — kept out of the main command for clarity.
extension DiffusionRunner {
    // MARK: - Parity Test

    func runParityTest(modelURL: URL, dataDir: URL) async throws {
        print("Running parity test with data from: \(dataDir.path)")

        let isFlux2 = FileManager.default.fileExists(
            atPath: dataDir.appendingPathComponent("text_encoder_attention_mask.npy").path)

        guard isFlux2 else {
            print("Error: parity test data does not match a supported (FLUX.2) pipeline")
            throw ExitCode.failure
        }
        try await runFlux2ComponentParity(modelURL: modelURL, dataDir: dataDir)
    }

    // MARK: - Flux2 Component Parity

    private func runFlux2ComponentParity(modelURL: URL, dataDir: URL) async throws {
        let configSource: PipelineDescriptor.ConfigSource = configPath.map { .file(URL(fileURLWithPath: $0)) } ?? .auto
        let descriptor = try PipelineDescriptor.resolve(at: modelURL, config: configSource)

        guard let teAsset = descriptor.components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }
        guard let vaeAsset = descriptor.components.vaeDecoder else {
            throw PipelineLoadError.missingComponent("vae_decoder")
        }

        let textEncoder = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(teAsset))
        let vaeDecoder = CoreAIDiffusionModelFunction(modelURL: modelURL.appendingPathComponent(vaeAsset))

        // --- Text Encoder ---
        print("\n=== Text Encoder ===")
        let teInputIds = try loadNpy(dataDir.appendingPathComponent("text_encoder_input_ids.npy"))
        let teAttMask = try loadNpy(dataDir.appendingPathComponent("text_encoder_attention_mask.npy"))
        let expectedTE = try loadNpy(dataDir.appendingPathComponent("text_encoder_output.npy"))

        try await textEncoder.loadResources()
        let teInputDescs = try await textEncoder.inputDescriptors
        let idsType = teInputDescs["input_ids"]?.scalarType ?? .int32
        let maskType = teInputDescs["attention_mask"]?.scalarType ?? .int32

        let teOutputs = try await textEncoder.predictAllOutputs(inputs: [
            "input_ids": floatsToNDArray(teInputIds.data, asInt32: true, shape: teInputIds.shape, scalarType: idsType),
            "attention_mask": floatsToNDArray(
                teAttMask.data, asInt32: true, shape: teAttMask.shape, scalarType: maskType),
        ])
        if let hiddenKey = teOutputs.keys.first(where: { $0.contains("hidden") }),
            let actual = teOutputs[hiddenKey]
        {
            let cosine = cosineSimilarity(actual, expectedTE.data)
            print("  Cosine similarity: \(cosine)")
        } else if let actual = teOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedTE.data)
            print("  Cosine similarity (first output): \(cosine)")
        }

        // --- VAE Decoder ---
        print("\n=== VAE Decoder ===")
        let vaeInput = try loadNpy(dataDir.appendingPathComponent("vae_decoder_input.npy"))
        let expectedVAE = try loadNpy(dataDir.appendingPathComponent("vae_decoder_output.npy"))

        try await vaeDecoder.loadResources()
        let vaeInputDescs = try await vaeDecoder.inputDescriptors
        let vaeType = vaeInputDescs.values.first?.scalarType ?? .float32
        let vaeND = floatsToNDArray(vaeInput.data, asInt32: false, shape: vaeInput.shape, scalarType: vaeType)
        let vaeOutputs = try await vaeDecoder.predictAutoNamed(inputs: [vaeND])
        if let actual = vaeOutputs.values.first {
            let cosine = cosineSimilarity(actual, expectedVAE.data)
            print("  Cosine similarity: \(cosine)")
        }

        print("\nDone.")
    }

    // MARK: - Numpy Loader

    /// Shape plus every element widened to `Float`, which is all the comparisons below need.
    struct NpyData {
        let shape: [Int]
        let data: [Float]
    }

    /// Thin adapter over ``CoreAIShared/NpyArray``.
    private func loadNpy(_ url: URL) throws -> NpyData {
        let array = try NpyArray.load(url)
        return NpyData(shape: array.shape, data: array.asFloat())
    }

    // MARK: - Helpers

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }

    /// Widen a `[Float]` buffer into an `NDArray` of the requested scalar type.
    ///
    /// Not diffusion- or model-specific. Float targets defer to
    /// `CoreAIShared.fillFloatNDArray` (the codebase's f16-safe fill); only the int32 case —
    /// token-id inputs the `.npy` fixtures store as float — is handled locally.
    private func floatsToNDArray(_ floats: [Float], asInt32: Bool, shape: [Int], scalarType: NDArray.ScalarType? = nil)
        -> NDArray
    {
        if asInt32 {
            var array = NDArray(shape: shape, scalarType: .int32)
            let view = array.mutableView(as: Int32.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for i in 0..<floats.count { ptr[i] = Int32(floats[i]) }
            }
            return array
        }
        var array = NDArray(shape: shape, scalarType: scalarType ?? .float32)
        fillFloatNDArray(&array, with: floats)
        return array
    }
}
