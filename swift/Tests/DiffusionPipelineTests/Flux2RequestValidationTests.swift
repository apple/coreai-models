// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAIDiffusionPipeline

/// FLUX.2 request validation against a bundle's exported shapes.
///
/// A request a bundle can't serve must be rejected up front: past a static trace or a
/// dynamic `Dim` bound, the graph fails deep inside the runtime instead.
@Suite("FLUX.2 request validation")
struct Flux2RequestValidationTests {
    /// The bounds `--dynamic` writes to metadata.json.
    private let dynamic = DiffusionConfig(
        type: .flux2, imageSize: 1024, dynamicShapes: true, minImageSize: 256, maxImageSize: 1024,
        imageSizeMultiple: 16, maxImageTokens: 16384, minReferenceSize: 128)
    private let fixed = DiffusionConfig(type: .flux2, imageSize: 1024)

    private typealias Pipeline = FlowTransformerPipeline

    // MARK: Output size

    @Test("A static bundle only produces the size it was loaded for")
    func staticImageSize() throws {
        #expect(try Pipeline.resolveImageSize(nil, loadedSize: 512, config: fixed) == 512)
        #expect(try Pipeline.resolveImageSize(512, loadedSize: 512, config: fixed) == 512)
        #expect(throws: (any Error).self) { try Pipeline.resolveImageSize(768, loadedSize: 1024, config: fixed) }
        #expect(throws: (any Error).self) { try Pipeline.resolveImageSize(1024, loadedSize: 512, config: fixed) }
    }

    @Test("A dynamic bundle takes any multiple of 16 within its bounds")
    func dynamicImageSize() throws {
        for size in [256, 400, 768, 1024] {
            #expect(try Pipeline.resolveImageSize(size, loadedSize: 1024, config: dynamic) == size)
        }
        // Below the range, above it, and not on the token grid.
        for size in [240, 1040, 500] {
            #expect(throws: (any Error).self, "\(size)") {
                try Pipeline.resolveImageSize(size, loadedSize: 1024, config: dynamic)
            }
        }
    }

    // MARK: References

    @Test("Only a dynamic bundle takes more than one reference")
    func referenceCount() throws {
        try Pipeline.checkReferenceCount(1, config: fixed)
        try Pipeline.checkReferenceCount(3, config: dynamic)
        #expect(throws: (any Error).self) { try Pipeline.checkReferenceCount(2, config: fixed) }
    }

    /// A static bundle is traced per preset, so a pixel size must name one; the preset picks
    /// the img2img graph that serves it.
    @Test("A static bundle serves only its presets, by name or by pixel size")
    func staticReferenceSize() throws {
        let outputSide = 32  // 512 px output
        let cases: [(ReferenceSize, side: Int, preset: ReferenceGrid)] = [
            (.half, 16, .half), (.pixels(512), 32, .full), (.pixels(256), 16, .half),
            (.pixels(128), 8, .quarter),
        ]
        for (size, side, preset) in cases {
            let resolved = try Pipeline.resolveReferenceSide(
                size, spatialSide: outputSide, loadedSize: 512, config: fixed)
            #expect(resolved.side == side && resolved.preset == preset, "\(size)")
        }
        for size: ReferenceSize in [.pixels(384), .pixels(250)] {
            #expect(throws: (any Error).self, "\(size)") {
                try Pipeline.resolveReferenceSide(size, spatialSide: outputSide, loadedSize: 512, config: fixed)
            }
        }
    }

    @Test("A dynamic bundle takes any aligned reference size in its encoder range")
    func dynamicReferenceSize() throws {
        let atSide48 = try Pipeline.resolveReferenceSide(
            .pixels(384), spatialSide: 48, loadedSize: 1024, config: dynamic)
        #expect(atSide48.side == 24 && atSide48.preset == nil)
        let half = try Pipeline.resolveReferenceSide(.half, spatialSide: 64, loadedSize: 1024, config: dynamic)
        #expect(half.side == 32)

        // Not on the token grid, below the encoder minimum, above the maximum, and a preset
        // that resolves below the minimum at a small output (quarter of 256 px is 64 px).
        let rejected: [(ReferenceSize, spatialSide: Int)] = [
            (.pixels(100), 64), (.pixels(112), 64), (.pixels(1040), 64), (.quarter, 16),
        ]
        for (size, spatialSide) in rejected {
            #expect(throws: (any Error).self, "\(size) at \(spatialSide)") {
                try Pipeline.resolveReferenceSide(
                    size, spatialSide: spatialSide, loadedSize: 1024, config: dynamic)
            }
        }
    }

    @Test("Noise plus references must fit the exported token limit")
    func tokenBudget() throws {
        let full = 64  // 1024 px: 4096 tokens per grid
        try Pipeline.checkImageTokenBudget(noiseTokens: full * full, refSides: [full, full, full], config: dynamic)
        #expect(throws: (any Error).self) {
            try Pipeline.checkImageTokenBudget(
                noiseTokens: full * full, refSides: [full, full, full, full], config: dynamic)
        }
        // A static bundle's length is fixed by its trace; there is no budget to check.
        try Pipeline.checkImageTokenBudget(noiseTokens: full * full, refSides: [full, full, full, full], config: fixed)
    }

    // MARK: Encoder

    @Test("Input images without a VAE encoder are refused, not dropped")
    func missingEncoder() throws {
        try Pipeline.checkEncoder(hasReferences: false, hasEncoder: false, mode: .full, config: fixed)
        try Pipeline.checkEncoder(hasReferences: true, hasEncoder: true, mode: .full, config: fixed)
        for (mode, config, asset) in [
            (DecodeResolution.full, fixed, "vae_encoder"), (.half, fixed, "vae_encoder_half"),
            (.half, dynamic, "vae_encoder"),
        ] {
            do {
                try Pipeline.checkEncoder(hasReferences: true, hasEncoder: false, mode: mode, config: config)
                Issue.record("accepted input images without an encoder (\(mode))")
            } catch PipelineLoadError.unsupportedConfiguration(let message) {
                #expect(message.contains("the \(asset) component"))
            }
        }
    }

    // MARK: Tiled decode

    @Test("Tiled decode needs tiles the decoder accepts, and static img2img can't tile")
    func tiledDecode() throws {
        try Pipeline.checkTiledDecode(imageSize: 512, isImageToImage: true, config: dynamic)
        #expect(throws: (any Error).self) {
            try Pipeline.checkTiledDecode(imageSize: 400, isImageToImage: false, config: dynamic)
        }
        try Pipeline.checkTiledDecode(imageSize: 1024, isImageToImage: false, config: fixed)
        #expect(throws: (any Error).self) {
            try Pipeline.checkTiledDecode(imageSize: 1024, isImageToImage: true, config: fixed)
        }
    }
}
