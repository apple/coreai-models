// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Covers the host-side patchify of pre-patchified vision encoders and how `VisionFrontEnd` binds
/// its pixel input by rank.
///
/// The small geometry: an 8x8 image at patch 2 is a 4x4 patch grid, merged 2x2 into 4 tokens, so
/// the pre-patchified input is `[1, 16, patch_dim]`; a `patch_dim` of 24 is 2 temporal copies of
/// `3 * 2 * 2`.
@Suite("Vision patchify")
struct VisionPatchifyTests {
    private let imageSize = 8
    private let numPatches = 16
    private let patchDim = 24

    private var visionConfig: VisionConfig {
        VisionConfig(imageSize: imageSize, patchSize: 2, imageTokenCount: 4, imageTokenId: 7)
    }

    /// CHW pixels whose value is their own flat index.
    private var chw: [Float] { (0..<(3 * imageSize * imageSize)).map(Float.init) }

    /// The exporter's `StaticVisionEncoder._patchify` for one image: duplicate the frame along
    /// the temporal axis, reshape `[T, C, H, W]` to
    /// `(1, T, C, grid / merge, merge, patch, grid / merge, merge, patch)`, permute it by
    /// `(0, 3, 6, 4, 7, 2, 1, 5, 8)`, and flatten to `[num_patches, patch_dim]`.
    private func exporterPatchify(
        chw: [Float], imageSize: Int, patch: Int, merge: Int, temporal: Int
    ) -> [Float] {
        let source = Array([[Float]](repeating: chw, count: temporal).joined())
        let grid = imageSize / patch
        let dims = [1, temporal, 3, grid / merge, merge, patch, grid / merge, merge, patch]
        let permutation = [0, 3, 6, 4, 7, 2, 1, 5, 8]
        var strides = [Int](repeating: 1, count: dims.count)
        for axis in stride(from: dims.count - 2, through: 0, by: -1) {
            strides[axis] = strides[axis + 1] * dims[axis + 1]
        }
        let permutedDims = permutation.map { dims[$0] }

        var result: [Float] = []
        result.reserveCapacity(source.count)
        var index = [Int](repeating: 0, count: dims.count)
        for _ in 0..<source.count {
            var offset = 0
            for (axis, sourceAxis) in permutation.enumerated() {
                offset += index[axis] * strides[sourceAxis]
            }
            result.append(source[offset])
            var axis = dims.count - 1
            while axis >= 0 {
                index[axis] += 1
                if index[axis] < permutedDims[axis] { break }
                index[axis] = 0
                axis -= 1
            }
        }
        return result
    }

    private func row(_ patches: [Float], _ index: Int) -> [Int] {
        patches[(index * patchDim)..<((index + 1) * patchDim)].map { Int($0) }
    }

    // MARK: - VisionPatchify

    @Test("Geometry is derived from the vision config and the declared input shape")
    func derivesGeometry() throws {
        let patchify = try VisionPatchify(visionConfig: visionConfig, numPatches: numPatches, patchDim: patchDim)
        #expect(patchify.patchSize == 2)
        #expect(patchify.mergeSize == 2)
        #expect(patchify.temporalSize == 2)

        let qwen = VisionConfig(imageSize: 448, patchSize: 16, imageTokenCount: 196, imageTokenId: 7)
        let qwenPatchify = try VisionPatchify(visionConfig: qwen, numPatches: 784, patchDim: 1536)
        #expect(qwenPatchify.mergeSize == 2)
        #expect(qwenPatchify.temporalSize == 2)
    }

    @Test("Patch rows follow the exporter's patch order")
    func matchesExporterOrder() throws {
        let patchify = try VisionPatchify(visionConfig: visionConfig, numPatches: numPatches, patchDim: patchDim)
        let patches = try patchify(chw: chw)

        #expect(patches == exporterPatchify(chw: chw, imageSize: imageSize, patch: 2, merge: 2, temporal: 2))

        // The patch whose top-left pixel sits at `topLeft` in channel 0: for each channel, both
        // temporal copies of its 2x2 pixels.
        func patchRow(topLeft: Int) -> [Int] {
            [0, 64, 128].flatMap { channel in
                let pixels = [0, 1, 8, 9].map { topLeft + channel + $0 }
                return pixels + pixels
            }
        }
        // Rows are merge-window-major: row 2 is the lower-left patch of the first window, row 5
        // the upper-right patch of the second, row 15 the last patch of the image.
        #expect(row(patches, 0) == patchRow(topLeft: 0))
        #expect(row(patches, 2) == patchRow(topLeft: 16))
        #expect(row(patches, 5) == patchRow(topLeft: 6))
        #expect(row(patches, 15) == patchRow(topLeft: 54))
    }

    @Test("A single temporal copy and a 1x1 merge window follow the exporter's order too")
    func matchesExporterOrderWithoutMergeOrTemporal() throws {
        let config = VisionConfig(imageSize: imageSize, patchSize: 2, imageTokenCount: 16, imageTokenId: 7)
        let patchify = try VisionPatchify(visionConfig: config, numPatches: 16, patchDim: 12)
        #expect(patchify.mergeSize == 1)
        #expect(patchify.temporalSize == 1)
        #expect(
            try patchify(chw: chw) == exporterPatchify(chw: chw, imageSize: imageSize, patch: 2, merge: 1, temporal: 1))
    }

    @Test("A declared shape that does not fit the vision config is rejected")
    func rejectsInconsistentGeometry() {
        // 8x8 at patch 2 has 16 patches, not 9.
        #expect(throws: InferenceRuntimeError.self) {
            _ = try VisionPatchify(visionConfig: visionConfig, numPatches: 9, patchDim: patchDim)
        }
        // patch_dim is not a multiple of 3 * 2 * 2.
        #expect(throws: InferenceRuntimeError.self) {
            _ = try VisionPatchify(visionConfig: visionConfig, numPatches: numPatches, patchDim: 20)
        }
        // 16 patches do not merge into 3 tokens.
        let threeTokens = VisionConfig(imageSize: imageSize, patchSize: 2, imageTokenCount: 3, imageTokenId: 7)
        #expect(throws: InferenceRuntimeError.self) {
            _ = try VisionPatchify(visionConfig: threeTokens, numPatches: numPatches, patchDim: patchDim)
        }
        // image_size is not a multiple of patch_size.
        let ragged = VisionConfig(imageSize: 9, patchSize: 2, imageTokenCount: 4, imageTokenId: 7)
        #expect(throws: InferenceRuntimeError.self) {
            _ = try VisionPatchify(visionConfig: ragged, numPatches: numPatches, patchDim: patchDim)
        }
    }

    @Test("CHW pixels of another image size are rejected")
    func rejectsWrongPixelCount() throws {
        let patchify = try VisionPatchify(visionConfig: visionConfig, numPatches: numPatches, patchDim: patchDim)
        #expect(throws: InferenceRuntimeError.self) { _ = try patchify(chw: Array(chw.dropLast())) }
    }

    // MARK: - VisionFrontEnd.PixelInput

    @Test("A rank-3 pixel input is bound as declared and filled with patch rows")
    func rank3Patchifies() throws {
        let input = try VisionFrontEnd.PixelInput(
            declaredShape: [1, numPatches, patchDim], visionConfig: visionConfig)
        #expect(input.patchify != nil)
        #expect(input.shape == [1, numPatches, patchDim])

        let buffer = try input.buffer(fromCHW: chw)
        #expect(buffer == exporterPatchify(chw: chw, imageSize: imageSize, patch: 2, merge: 2, temporal: 2))
        try input.validate(buffer, name: "pixel_values")
        // The CHW pixels themselves hold half the floats of two temporal copies.
        #expect(throws: InferenceRuntimeError.self) { try input.validate(chw, name: "pixel_values") }
    }

    @Test("A rank-4 pixel input takes the CHW pixels unchanged")
    func rank4PassesThrough() throws {
        for declared in [[1, 3, -1, -1], [1, 3, imageSize, imageSize]] {
            let input = try VisionFrontEnd.PixelInput(declaredShape: declared, visionConfig: visionConfig)
            #expect(input.patchify == nil)
            #expect(input.shape == [1, 3, imageSize, imageSize])

            let buffer = try input.buffer(fromCHW: chw)
            #expect(buffer == chw)
            try input.validate(buffer, name: "pixel_values")
        }
    }

    @Test("A static input is bound at its declared shape and rejects a buffer of another size")
    func staticShapeIsDeclared() throws {
        let input = try VisionFrontEnd.PixelInput(declaredShape: [1, 3, 4, 4], visionConfig: visionConfig)
        #expect(input.shape == [1, 3, 4, 4])
        #expect(input.elementCount == 48)
        #expect(throws: InferenceRuntimeError.self) { try input.validate(chw, name: "pixel_values") }
    }

    @Test("A rank-3 pixel input with a dynamic dimension is rejected")
    func rank3DynamicRejected() {
        #expect(throws: InferenceRuntimeError.self) {
            _ = try VisionFrontEnd.PixelInput(declaredShape: [1, -1, patchDim], visionConfig: visionConfig)
        }
    }
}
