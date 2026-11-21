// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Antialiased bicubic resize of 8-bit RGB, bit-identical to PyTorch's CPU uint8 path
/// (`interpolate(uint8, mode="bicubic", antialias=True, align_corners=False)`, which torchvision's
/// `resize(..., BICUBIC, antialias=True)` reaches): width pass, then height pass, Keys cubic
/// (`a = -0.5`) stretched when downscaling, weights quantized to int16, uint8 between the passes.
/// A width-1 intermediate gets the intended result; PyTorch's height pass repeats its first row there.
enum AntialiasedBicubicResize {
    static func resize(_ image: RGB8Image, toWidth width: Int, toHeight height: Int) -> RGB8Image {
        var pixels = image.pixels
        if width != image.width { pixels = pass(pixels, lines: image.height, from: image.width, to: width, run: 3) }
        if height != image.height { pixels = pass(pixels, lines: 1, from: image.height, to: height, run: width * 3) }
        return RGB8Image(width: width, height: height, pixels: pixels)
    }

    /// Resizes each of `lines` lines of `from` elements of `run` bytes to `to` elements.
    private static func pass(_ src: [UInt8], lines: Int, from n: Int, to m: Int, run: Int) -> [UInt8] {
        let taps = AntialiasTaps.taps(sourceSize: n, destinationSize: m, filter: .keysCubic).map { tap in
            let total = tap.weights.reduce(0, +)
            return (start: tap.start, weights: total == 0 ? tap.weights : tap.weights.map { $0 / total })
        }
        let maxWeight = taps.lazy.flatMap(\.weights).max() ?? 0
        var precision = 0
        while precision < 22, Int(0.5 + maxWeight * Double(1 << (precision + 1))) < 1 << 15 { precision += 1 }
        let one = Double(1 << precision)
        let quantized = taps.map { tap in
            tap.weights.map { w in
                let v = w * one
                return Int32(Int16(truncatingIfNeeded: v < 0 ? Int(-0.5 + v) : Int(0.5 + v)))
            }
        }
        var out = [UInt8](repeating: 0, count: lines * m * run)
        var acc = [Int32](repeating: 0, count: run)
        src.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { d in
                acc.withUnsafeMutableBufferPointer { a in
                    for line in 0..<lines {
                        for i in 0..<m {
                            for k in 0..<run { a[k] = 1 << Int32(precision - 1) }
                            quantized[i].withUnsafeBufferPointer { q in
                                for (j, w) in q.enumerated() {
                                    let p = (line * n + taps[i].start + j) * run
                                    for k in 0..<run { a[k] &+= Int32(s[p + k]) &* w }
                                }
                            }
                            let o = (line * m + i) * run
                            for k in 0..<run { d[o + k] = UInt8(clamping: a[k] >> Int32(precision)) }
                        }
                    }
                }
            }
        }
        return out
    }
}
