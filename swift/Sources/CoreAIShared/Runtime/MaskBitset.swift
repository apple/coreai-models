// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// A binary mask packed one pixel per bit, row-major.
///
/// Tracking is dominated by set arithmetic over masks: association, NMS and occlusion
/// suppression are all pairwise IoU matrices, and each runs once per frame. Packed, both
/// intersection and union fall out of `nonzeroBitCount` without needing Accelerate.
public struct MaskBitset: Sendable, Equatable {
    public let width: Int
    public let height: Int
    private(set) var words: [UInt64]

    public init(width: Int, height: Int) {
        precondition(width >= 0 && height >= 0, "MaskBitset dimensions must be non-negative")
        self.width = width
        self.height = height
        self.words = [UInt64](repeating: 0, count: (width * height + 63) / 64)
    }

    /// Threshold a row-major float buffer: `value > threshold` becomes a set bit.
    ///
    /// The comparison is strictly greater to match HF, which binarizes mask logits with
    /// `mask > 0` throughout; `>=` would flip every exactly-zero pixel.
    public init(thresholding values: [Float], width: Int, height: Int, above threshold: Float = 0) {
        self.init(width: width, height: height)
        precondition(
            values.count >= width * height,
            "MaskBitset needs \(width * height) values, got \(values.count)")
        let count = width * height
        words.withUnsafeMutableBufferPointer { output in
            values.withUnsafeBufferPointer { input in
                var index = 0
                var wordIndex = 0
                while index < count {
                    let end = min(index + 64, count)
                    var word: UInt64 = 0
                    var bit: UInt64 = 1
                    for i in index..<end {
                        if input[i] > threshold { word |= bit }
                        bit <<= 1
                    }
                    output[wordIndex] = word
                    wordIndex += 1
                    index = end
                }
            }
        }
    }

    /// Threshold a sub-range of a larger row-major buffer, one mask out of a stacked
    /// `[N, H, W]` tensor.
    public init(
        thresholding values: ArraySlice<Float>, width: Int, height: Int, above threshold: Float = 0
    ) {
        self.init(thresholding: Array(values), width: width, height: height, above: threshold)
    }

    public subscript(x: Int, y: Int) -> Bool {
        get {
            let index = y * width + x
            return words[index >> 6] & (1 << UInt64(index & 63)) != 0
        }
        set {
            let index = y * width + x
            if newValue {
                words[index >> 6] |= 1 << UInt64(index & 63)
            } else {
                words[index >> 6] &= ~(1 << UInt64(index & 63))
            }
        }
    }

    /// Number of set pixels.
    public var area: Int {
        var total = 0
        for word in words { total += word.nonzeroBitCount }
        return total
    }

    public var isEmpty: Bool {
        for word in words where word != 0 { return false }
        return true
    }

    /// Intersection over union, matching `modeling_sam3_video.mask_iou`.
    ///
    /// HF clamps the union to a minimum of 1, so empty-vs-empty scores 0 rather than
    /// dividing by zero. Association therefore treats two empty masks as unrelated.
    public func iou(_ other: MaskBitset) -> Float {
        precondition(
            width == other.width && height == other.height,
            "MaskBitset.iou requires matching dimensions")
        var intersection = 0
        var union = 0
        for index in words.indices {
            let a = words[index]
            let b = other.words[index]
            intersection += (a & b).nonzeroBitCount
            union += (a | b).nonzeroBitCount
        }
        return Float(intersection) / Float(max(union, 1))
    }

    /// Tight bounding box of the set pixels, top-left origin. Empty masks give `.zero`.
    ///
    /// `torchvision.ops.masks_to_boxes` reports inclusive extremes, so a single set pixel is
    /// a zero-sized rect. Kept that way so the values compare directly against a reference.
    public var boundingBox: CGRect {
        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1
        forEachSetIndex { index in
            let y = index / width
            let x = index - y * width
            if x < minX { minX = x }
            if x > maxX { maxX = x }
            if y < minY { minY = y }
            if y > maxY { maxY = y }
        }
        guard maxX >= 0 else { return .zero }
        return CGRect(
            x: CGFloat(minX), y: CGFloat(minY),
            width: CGFloat(maxX - minX), height: CGFloat(maxY - minY))
    }

    /// Visit every set pixel's row-major index in ascending order.
    @inline(__always)
    public func forEachSetIndex(_ body: (Int) -> Void) {
        let count = width * height
        for wordIndex in words.indices {
            var word = words[wordIndex]
            let base = wordIndex << 6
            while word != 0 {
                let index = base + Int(word.trailingZeroBitCount)
                if index >= count { return }
                body(index)
                word &= word - 1
            }
        }
    }

    /// Row-major bytes, one per pixel, 1 for foreground.
    public func toBytes() -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        forEachSetIndex { out[$0] = 1 }
        return out
    }

    /// Rebuild from `numpy.packbits` output: MSB-first within each byte, row-major.
    public init(packedBits: [UInt8], width: Int, height: Int) {
        self.init(width: width, height: height)
        let count = width * height
        for index in 0..<count {
            let byte = packedBits[index >> 3]
            if byte & (0x80 >> UInt8(index & 7)) != 0 {
                words[index >> 6] |= 1 << UInt64(index & 63)
            }
        }
    }
}
