// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

@Suite("DiffusionPipeline")
struct DiffusionPipelineTests {
    @Test("Configuration defaults")
    func configurationDefaults() {
        let config = PipelineConfiguration(prompt: "a cat")
        #expect(config.prompt == "a cat")
        #expect(config.negativePrompt == "")
        #expect(config.stepCount == 50)
        #expect(config.guidanceScale == 7.5)
        #expect(config.schedulerType == .discreteFlow)
        #expect(!config.isImageToImage)
    }

    @Test("startingImage is reference 1, followed by referenceImages in order")
    func referenceImageOrder() throws {
        func image(width: Int) throws -> CGImage {
            try DiffusionUtilities.pixelsToCGImage(
                [Float](repeating: 0.5, count: 3 * width), height: 1, width: width)
        }
        let config = PipelineConfiguration(
            prompt: "a cat", startingImage: try image(width: 1),
            referenceImages: [try image(width: 2), try image(width: 3)])
        #expect(config.allReferenceImages.map(\.width) == [1, 2, 3])
        #expect(config.isImageToImage)

        // A reference list without startingImage is still img2img.
        let listOnly = PipelineConfiguration(prompt: "a cat", referenceImages: [try image(width: 2)])
        #expect(listOnly.allReferenceImages.map(\.width) == [2])
        #expect(listOnly.isImageToImage)

        // Equality counts every reference (images themselves aren't Hashable): img2img
        // differs from txt2img, and both spellings of a single reference agree.
        let startingOnly = PipelineConfiguration(prompt: "a cat", startingImage: try image(width: 2))
        #expect(startingOnly == listOnly)
        #expect(startingOnly != PipelineConfiguration(prompt: "a cat"))
    }

    @Test("ReferenceSize parses presets and pixel sizes, and round-trips")
    func referenceSizeParsing() {
        #expect(ReferenceSize("half") == .relative(.half))
        #expect(ReferenceSize("384") == .pixels(384))
        for bad in ["0", "-16", "Half", "384px", ""] {
            #expect(ReferenceSize(bad) == nil, "\(bad) should not parse")
        }
        for size: ReferenceSize in [.relative(.full), .relative(.quarter), .pixels(128)] {
            #expect(ReferenceSize(size.description) == size)
        }
    }

    /// A static bundle is traced per preset, so a pixel size is only servable when it
    /// equals one of them at the current output grid.
    @Test("Pixel reference sizes map to the preset they equal at the output grid")
    func referenceSizePresetMapping() {
        let outputSide = 32  // 512 px output, 16 px per token
        #expect(ReferenceSize.pixels(512).preset(outputSide: outputSide, patchSize: 16) == .full)
        #expect(ReferenceSize.pixels(256).preset(outputSide: outputSide, patchSize: 16) == .half)
        #expect(ReferenceSize.pixels(128).preset(outputSide: outputSide, patchSize: 16) == .quarter)
        #expect(ReferenceSize.pixels(384).preset(outputSide: outputSide, patchSize: 16) == nil)
        #expect(ReferenceSize.pixels(250).preset(outputSide: outputSide, patchSize: 16) == nil)
        #expect(ReferenceSize.relative(.half).preset(outputSide: outputSide, patchSize: 16) == .half)
    }

    @Test("Reference sizes resolve to a token-grid side")
    func referenceSizeSide() {
        #expect(ReferenceSize.pixels(384).side(outputSide: 64, patchSize: 16) == 24)
        #expect(ReferenceSize.relative(.half).side(outputSide: 48, patchSize: 16) == 24)
        #expect(ReferenceSize.relative(.quarter).side(outputSide: 64, patchSize: 16) == 16)
    }

    /// Presets round down to whole 16-pixel tokens, so the reported size is what is encoded.
    /// A pixel size comes back unchanged, so an invalid one is reported as given.
    @Test("Reference sizes resolve to pixels at an output size")
    func referenceSizePixels() {
        #expect(ReferenceSize.full.pixelSize(outputSize: 1024) == 1024)
        #expect(ReferenceSize.half.pixelSize(outputSize: 400) == 192)  // 25 tokens → 12
        #expect(ReferenceSize.pixels(384).pixelSize(outputSize: 1024) == 384)
        #expect(ReferenceSize.pixels(100).pixelSize(outputSize: 1024) == 100)
        #expect(ReferenceSize.half == .relative(.half))
    }

    /// Callers written against `referenceGrid` must keep compiling and behaving the same.
    @available(*, deprecated)
    @Test("Pre-referenceSize API still works")
    func legacyReferenceGrid() {
        var config = PipelineConfiguration(prompt: "a cat", referenceGrid: .quarter)
        #expect(config.referenceSize == .quarter)
        #expect(config.referenceGrid == .quarter)
        config.referenceGrid = .half
        #expect(config.referenceSize == .half)
        config.referenceSize = .pixels(384)
        #expect(config.referenceGrid == .full)
    }

    @Test("Scheduler type raw values match the metadata.json diffusion block")
    func schedulerTypeRawValues() {
        #expect(SchedulerType(rawValue: "flow_match_euler") == .discreteFlow)
    }

    @Test("Descriptors auto-load and throw on bad path")
    func descriptorsThrowBeforeLoad() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        await #expect(throws: (any Error).self) { try await fn.inputDescriptors }
        await #expect(throws: (any Error).self) { try await fn.outputDescriptors }
    }

    @Test("Load with bad path throws")
    func loadBadPathThrows() async {
        let fn = CoreAIDiffusionModelFunction(
            modelURL: URL(filePath: "/nonexistent.aimodel"))
        await #expect(throws: (any Error).self) {
            try await fn.loadResources()
        }
    }
}

@Suite("TextEncoder tokenize padding")
struct TextEncoderTokenizeTests {
    @Test("Tokenize closure pads short input to maxLength")
    func padsShortInput() {
        let maxLength = 77
        let tokenize: @Sendable (String) -> [Int32] = { _ in
            var ids = [Int32](1...10)
            if ids.count < maxLength {
                ids += [Int32](repeating: 0, count: maxLength - ids.count)
            }
            return ids
        }
        let ids = tokenize("short prompt")
        #expect(ids.count == 77)
        #expect(ids.last == 0)
    }

    @Test("Tokenize closure truncates long input to maxLength")
    func truncatesLongInput() {
        let maxLength = 77
        let tokenize: @Sendable (String) -> [Int32] = { _ in
            var ids = [Int32](1...100)
            if ids.count > maxLength {
                ids = Array(ids.prefix(maxLength))
            }
            return ids
        }
        let ids = tokenize("very long prompt that produces many tokens")
        #expect(ids.count == 77)
        #expect(ids[76] == 77)
    }
}
