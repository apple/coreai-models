// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreGraphics
import Foundation

/// Image preprocessing (R7): RGB decode, antialiased bicubic resize, `/255`, then `(x - mean) / std`.
/// The resize matches torchvision's uint8 path (HF's fast processors) exactly and PIL's within 2 levels.
enum ImagePreparer {
    /// `centerCrop`: resizes the shortest edge to `size`, crops the center square and returns
    /// `[3, size, size]` CHW values.
    static func centerCrop(_ image: RGB8Image, _ config: ImageConfig) -> [Float] {
        let size = config.size!
        let (width, height) = shortestEdgeSize(width: image.width, height: image.height, to: size)
        let resized = AntialiasedBicubicResize.resize(image, toWidth: width, toHeight: height)
        let top = (height - size) / 2
        let left = (width - size) / 2
        let plane = size * size
        var out = [Float](repeating: 0, count: 3 * plane)
        resized.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for c in 0..<3 {
                    for y in 0..<size {
                        let row = src.baseAddress! + ((top + y) * width + left) * 3 + c
                        vDSP_vfltu8(row, 3, dst.baseAddress! + c * plane + y * size, 1, vDSP_Length(size))
                    }
                    normalize(dst.baseAddress! + c * plane, stride: 1, count: plane, channel: c, config)
                }
            }
        }
        return out
    }

    /// `/255` then `(x - mean) / std` of `channel`, in place, as `x * scale + offset`. A multiply then
    /// an add, not `vDSP_vsmsa`, which fuses them and rounds differently.
    private static func normalize(
        _ values: UnsafeMutablePointer<Float>, stride: Int, count: Int, channel c: Int, _ config: ImageConfig
    ) {
        var scale = 1 / (255 * config.std[c])
        var offset = -config.mean[c] / config.std[c]
        vDSP_vsmul(values, stride, &scale, values, stride, vDSP_Length(count))
        vDSP_vsadd(values, stride, &offset, values, stride, vDSP_Length(count))
    }

    /// The size with the shortest edge at `size` and the other edge scaled and truncated, as
    /// Hugging Face's `get_resize_output_image_size(default_to_square=False)` computes it.
    static func shortestEdgeSize(width: Int, height: Int, to size: Int) -> (width: Int, height: Int) {
        let (short, long) = width <= height ? (width, height) : (height, width)
        let scaled = Int(Double(size) * Double(long) / Double(short))
        return width <= height ? (size, scaled) : (scaled, size)
    }

    /// Patches of one image for `aspectRatioPreserving`.
    struct Patches: Sendable, Equatable {
        /// `[maxPatches, patchDimension]`, zero rows for padding.
        var pixelValues: [Float]
        /// `[maxPatches, 2]` `(x, y)` patch coordinates, `(-1, -1)` for padding.
        var positionIds: [Int32]
        /// `realPatches / poolingKernelSize²`.
        var softTokens: Int
    }

    /// `aspectRatioPreserving`: resizes to `targetSize`, cuts row-major patches with pixels in
    /// `(y, x, channel)` order and pads to the budget.
    static func patchify(_ image: RGB8Image, _ config: ImageConfig) throws -> Patches {
        let geometry = config.patches!
        let (h, w) = try targetSize(height: image.height, width: image.width, geometry)
        let resized = AntialiasedBicubicResize.resize(image, toWidth: w, toHeight: h)
        let p = geometry.patchSize
        let (ph, pw) = (h / p, w / p)
        let dimension = geometry.patchDimension
        let row = w * 3
        var normalized = [Float](repeating: 0, count: h * row)
        resized.pixels.withUnsafeBufferPointer { src in
            normalized.withUnsafeMutableBufferPointer { dst in
                vDSP_vfltu8(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(h * row))
                for c in 0..<3 { normalize(dst.baseAddress! + c, stride: 3, count: h * w, channel: c, config) }
            }
        }
        var pixels = [Float](repeating: 0, count: geometry.maxPatches * dimension)
        var positions = [Int32](repeating: -1, count: geometry.maxPatches * 2)
        normalized.withUnsafeBufferPointer { src in
            pixels.withUnsafeMutableBufferPointer { dst in
                for py in 0..<ph {
                    for px in 0..<pw {
                        let patch = py * pw + px
                        positions[patch * 2] = Int32(px)
                        positions[patch * 2 + 1] = Int32(py)
                        for y in 0..<p {
                            (dst.baseAddress! + patch * dimension + y * p * 3).update(
                                from: src.baseAddress! + (py * p + y) * row + px * p * 3, count: p * 3)
                        }
                    }
                }
            }
        }
        let k = geometry.poolingKernelSize
        return Patches(pixelValues: pixels, positionIds: positions, softTokens: ph * pw / (k * k))
    }

    /// The largest `(height, width)` with sides divisible by `patchSize * poolingKernelSize` and at
    /// most `maxPatches` patches, as `embedding_gemma2_vision.aspect_ratio_preserving_size` computes it.
    static func targetSize(height: Int, width: Int, _ g: ImageConfig.Patches) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else { throw EmbeddingError.imageConversionFailed }
        let targetPixels = Double(g.maxPatches * g.patchSize * g.patchSize)
        let factor = (targetPixels / Double(height * width)).squareRoot()
        let side = g.poolingKernelSize * g.patchSize
        var th = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var tw = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        guard th > 0 || tw > 0 else { throw EmbeddingError.imageConversionFailed }
        let maxSide = g.maxSoftTokens * side
        if th == 0 {
            th = side
            tw = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maxSide)
        } else if tw == 0 {
            tw = side
            th = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maxSide)
        }
        guard Double(th * tw) <= targetPixels else { throw EmbeddingError.imageConversionFailed }
        return (th, tw)
    }
}
