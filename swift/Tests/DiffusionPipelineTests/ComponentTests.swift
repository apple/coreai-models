// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreGraphics
import Foundation
import Synchronization
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAIDiffusionPipeline

@Suite("Diffusion Components")
struct ComponentTests {
    // MARK: - CoreAITextEncoder

    @Test("TextEncoder requires function to be loaded")
    func textEncoderRequiresLoad() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        let encoder = CoreAITextEncoder(
            function: fn,
            tokenize: { _ in [Int32](repeating: 0, count: 77) },
            maxLength: 77)

        await #expect(throws: (any Error).self) {
            try await encoder.encode("test prompt")
        }
    }

    @Test("TextEncoder truncates long token sequences")
    func textEncoderTruncates() async {
        let capturedIds = Mutex<[Int32]?>(nil)
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        let encoder = CoreAITextEncoder(
            function: fn,
            tokenize: { _ in
                let ids = [Int32](repeating: 1, count: 100)
                capturedIds.withLock({ $0 = ids })
                return ids
            },
            maxLength: 77)

        // Will fail at predict (no model), but tokenize runs first
        _ = try? await encoder.encode("long text")
        let count = capturedIds.withLock({ $0?.count })
        #expect(count == 100)
    }

    // MARK: - CoreAIDiffusionModelFunction

    @Test("loadResources throws promptly on a missing file rather than hanging")
    func loadResourcesMissingFileThrows() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        await #expect(throws: CoreAIDiffusionError.self) {
            try await fn.loadResources()
        }
    }

    @Test("hasFunction throws promptly on a missing file rather than hanging")
    func hasFunctionMissingFileThrows() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        await #expect(throws: CoreAIDiffusionError.self) {
            _ = try await fn.hasFunction(named: "main")
        }
    }

    // MARK: - CoreAIDenoiser

    @Test("Denoiser requires function to be loaded")
    func denoiserRequiresLoad() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        let denoiser = CoreAIDenoiser(function: fn)

        await #expect(throws: (any Error).self) {
            try await denoiser.predictNoise(
                latents: NDArray(shape: [1, 4, 64, 64], scalarType: .float32),
                timestep: 999.0,
                textEmbeddings: NDArray(shape: [1, 77, 768], scalarType: .float32),
                additionalInputs: [:])
        }
    }

    // MARK: - CoreAILatentDecoder

    @Test("LatentDecoder requires function to be loaded")
    func decoderRequiresLoad() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        let decoder = CoreAILatentDecoder(function: fn)

        await #expect(throws: (any Error).self) {
            try await decoder.decode(
                NDArray(shape: [1, 4, 64, 64], scalarType: .float32),
                scaleFactor: 0.18215,
                shiftFactor: 0.0)
        }
    }

    @Test("pixelsToCGImage rejects wrong pixel count")
    func pixelsToCGImageWrongCount() {
        let badPixels = [Float](repeating: 0, count: 4 * 8 * 8)
        #expect(throws: CoreAIComponentError.self) {
            _ = try DiffusionUtilities.pixelsToCGImage(badPixels, height: 8, width: 8)
        }
    }

    @Test("pixelsToCGImage produces valid image")
    func pixelsToCGImageValid() throws {
        let pixels = [Float](repeating: 0.5, count: 3 * 8 * 8)
        let image = try DiffusionUtilities.pixelsToCGImage(pixels, height: 8, width: 8)
        #expect(image.width == 8)
        #expect(image.height == 8)
    }

    // MARK: - CoreAILatentEncoder

    @Test("LatentEncoder requires function to be loaded")
    func encoderRequiresLoad() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        let encoder = CoreAILatentEncoder(function: fn)

        // Create a minimal 1x1 CGImage for testing
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard
            let context = CGContext(
                data: nil, width: 1, height: 1,
                bitsPerComponent: 8, bytesPerRow: 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
            let image = context.makeImage()
        else {
            Issue.record("Failed to create test CGImage")
            return
        }

        await #expect(throws: (any Error).self) {
            try await encoder.encode(image, scaleFactor: 0.18215)
        }
    }
}

// MARK: - Flux2 Pipeline Utilities

private func makeFlux2Pipeline(bnMean: [Float]? = nil, bnVar: [Float]? = nil) -> FlowTransformerPipeline {
    let stub = CoreAIDiffusionModelFunction(modelURL: URL(filePath: "/nonexistent.aimodel"))
    return FlowTransformerPipeline(
        descriptor: PipelineDescriptor(),
        mode: .full,
        transformer: stub,
        textEncoder: stub,
        decoder: stub,
        encoder: nil,
        tokenizer: MockTokenizer(),
        batchNormMean: bnMean,
        batchNormVar: bnVar,
        batchNormEps: 1e-5
    )
}

@Suite("Flux2 Pipeline Utilities")
struct Flux2UtilityTests {
    // MARK: - patchify / unpatchify round-trip

    @Test("patchifyLatents is inverse of unpatchifyLatents")
    func patchifyRoundTrip() {
        guard #available(macOS 27, iOS 27, *) else { return }
        let inCh = 32
        let h = 2
        let w = 2
        let inChannels = inCh * 4  // 128
        let original = (0..<(inCh * (h * 2) * (w * 2))).map { Float($0) }

        let patchified = FlowTransformerPipeline.patchifyLatents(original, inChannels: inChannels, height: h, width: w)
        let recovered = FlowTransformerPipeline.unpatchifyLatents(patchified, channels: inChannels, height: h, width: w)

        #expect(original.count == recovered.count)
        for (a, b) in zip(original, recovered) {
            #expect(abs(a - b) < 1e-5)
        }
    }

    // MARK: - applyBatchNormNorm / applyBatchNormDenorm round-trip

    @Test("applyBatchNormNorm is inverse of applyBatchNormDenorm")
    func batchNormRoundTrip() {
        guard #available(macOS 27, iOS 27, *) else { return }
        let channels = 4
        let h = 2
        let w = 2
        let pipeline = makeFlux2Pipeline(
            bnMean: [0.1, -0.2, 0.3, -0.4],
            bnVar: [0.5, 1.0, 2.0, 0.25]
        )
        let original = (0..<(channels * h * w)).map { Float($0) * 0.1 }
        let normed = pipeline.applyBatchNormNorm(original, channels: channels, height: h, width: w)
        let recovered = pipeline.applyBatchNormDenorm(normed, channels: channels, height: h, width: w)

        #expect(original.count == recovered.count)
        for (a, b) in zip(original, recovered) {
            #expect(abs(a - b) < 1e-4)
        }
    }

    @Test("applyBatchNormNorm passes through when BN stats are nil")
    func batchNormNormPassthrough() {
        guard #available(macOS 27, iOS 27, *) else { return }
        let pipeline = makeFlux2Pipeline(bnMean: nil, bnVar: nil)
        let input: [Float] = [1, 2, 3, 4]
        let result = pipeline.applyBatchNormNorm(input, channels: 4, height: 1, width: 1)
        #expect(result == input)
    }
}
