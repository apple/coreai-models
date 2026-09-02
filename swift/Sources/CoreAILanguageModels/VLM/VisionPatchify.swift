// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Host-side patchify for pre-patchified vision encoders.
///
/// ANE-mappable exports declare `pixel_values` as `[1, num_patches, patch_dim]`
/// instead of `[1, 3, H, W]`: the rank-9 rearrangement that turns pixels into
/// Qwen patches exceeds the ANE compiler's rank-5 limit, so the host produces the
/// patch buffer the encoder consumes.
///
/// Layout, matching the exporter's `StaticVisionEncoder._patchify`: patches run
/// merge-window-major over `(grid_h, grid_w, merge_h, merge_w)`, and each patch
/// holds `[channel][temporal][patch_h][patch_w]` with the temporal axis carrying
/// duplicates of the single frame.
public struct VisionPatchify: Sendable {
    /// Square input edge in pixels.
    public let imageSize: Int
    /// Vision transformer patch edge in pixels.
    public let patchSize: Int
    /// Spatial merge window edge, in patches.
    public let mergeSize: Int
    /// Patches stacked along the temporal axis (a single image is duplicated).
    public let temporalSize: Int
    /// Rows of the vision input: `grid_h * grid_w`.
    public let numPatches: Int
    /// Columns of the vision input: `temporal * channels * patch * patch`.
    public let patchDim: Int

    private static let channels = 3

    /// Derive the patch geometry for a `[1, num_patches, patch_dim]` vision input.
    ///
    /// Bundle metadata carries `image_size`, `patch_size` and `image_token_count`;
    /// the temporal and merge factors are not written for single-image VLMs, and
    /// both are pinned by the declared shapes:
    /// `patch_dim = temporal * 3 * patch²` and `merge² = num_patches / tokens`.
    public init(visionConfig: VisionConfig, numPatches: Int, patchDim: Int) throws {
        let imageSize = visionConfig.imageSize
        let patchSize = visionConfig.patchSize
        guard patchSize > 0, imageSize > 0, imageSize % patchSize == 0 else {
            throw InferenceRuntimeError.invalidArgument(
                "VisionPatchify: image_size \(imageSize) is not a multiple of patch_size \(patchSize)")
        }
        let gridSide = imageSize / patchSize
        guard gridSide * gridSide == numPatches else {
            throw InferenceRuntimeError.invalidArgument(
                "VisionPatchify: vision input declares \(numPatches) patches, "
                    + "but a \(imageSize)x\(imageSize) image at patch \(patchSize) has \(gridSide * gridSide)")
        }

        let patchArea = patchSize * patchSize
        let spatialDim = Self.channels * patchArea
        guard patchDim > 0, patchDim % spatialDim == 0 else {
            throw InferenceRuntimeError.invalidArgument(
                "VisionPatchify: patch_dim \(patchDim) is not a multiple of "
                    + "channels*patch² (\(spatialDim))")
        }

        let tokenCount = visionConfig.tokensPerFrame ?? visionConfig.imageTokenCount
        guard tokenCount > 0, numPatches % tokenCount == 0 else {
            throw InferenceRuntimeError.invalidArgument(
                "VisionPatchify: \(numPatches) patches do not merge into \(tokenCount) visual tokens")
        }
        let mergeArea = numPatches / tokenCount
        let mergeSize = Int(Double(mergeArea).squareRoot().rounded())
        guard mergeSize * mergeSize == mergeArea, gridSide % mergeSize == 0 else {
            throw InferenceRuntimeError.invalidArgument(
                "VisionPatchify: patch grid \(gridSide) does not tile with merge window "
                    + "area \(mergeArea)")
        }

        self.imageSize = imageSize
        self.patchSize = patchSize
        self.mergeSize = mergeSize
        self.temporalSize = patchDim / spatialDim
        self.numPatches = numPatches
        self.patchDim = patchDim
    }

    /// Rearrange normalized CHW pixels `[3, imageSize, imageSize]` into
    /// `[num_patches, patch_dim]`.
    public func callAsFunction(chw pixels: [Float]) throws -> [Float] {
        let plane = imageSize * imageSize
        guard pixels.count == Self.channels * plane else {
            throw InferenceRuntimeError.invalidArgument(
                "VisionPatchify: expected \(Self.channels * plane) CHW floats, got \(pixels.count)")
        }

        let mergedGrid = imageSize / patchSize / mergeSize
        var patches = [Float](repeating: 0, count: numPatches * patchDim)
        patches.withUnsafeMutableBufferPointer { dst in
            pixels.withUnsafeBufferPointer { src in
                guard let dstBase = dst.baseAddress, let srcBase = src.baseAddress else { return }
                var index = 0
                for gh in 0..<mergedGrid {
                    for gw in 0..<mergedGrid {
                        for mh in 0..<mergeSize {
                            for mw in 0..<mergeSize {
                                let row = (gh * mergeSize + mh) * patchSize
                                let col = (gw * mergeSize + mw) * patchSize
                                for c in 0..<Self.channels {
                                    let channelBase = c * plane
                                    for _ in 0..<temporalSize {
                                        for ph in 0..<patchSize {
                                            let source = channelBase + (row + ph) * imageSize + col
                                            (dstBase + index).update(
                                                from: srcBase + source, count: patchSize)
                                            index += patchSize
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return patches
    }
}
