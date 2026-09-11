// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation

/// Visualization utilities for segmentation outputs.
public enum SegmentationVisualization {
    /// Composites a heat-map of `SemanticSegmentationMap` probabilities over `baseImage`.
    ///
    /// The heat map uses a blue→green→red gradient (low→high probability) at up to 78% opacity,
    /// so the original image remains visible underneath. The returned image matches
    /// `baseImage`'s dimensions; the probability grid is upscaled by Core Graphics
    /// when composited.
    public static func renderSemanticOverlay(onto baseImage: CGImage, map: SemanticSegmentationMap)
        -> CGImage?
    {
        let mapWidth = map.width
        let mapHeight = map.height
        guard mapWidth > 0, mapHeight > 0,
            map.probabilities.count == mapWidth * mapHeight
        else { return nil }

        let outWidth = baseImage.width
        let outHeight = baseImage.height

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard
            let context = CGContext(
                data: nil,
                width: outWidth, height: outHeight,
                bitsPerComponent: 8, bytesPerRow: outWidth * 4,
                space: colorSpace, bitmapInfo: bitmapInfo.rawValue
            )
        else { return nil }

        // Draw the base image. CGContext.draw handles the macOS y-flip internally.
        context.draw(baseImage, in: CGRect(x: 0, y: 0, width: CGFloat(outWidth), height: CGFloat(outHeight)))

        // Build a premultiplied-RGBA overlay from the probability grid.
        // CGImage pixel data is stored top-to-bottom (row 0 = top), matching the segmentation map,
        // so no vertical flip is needed when creating the overlay CGImage.
        var pixels = [UInt8](repeating: 0, count: mapWidth * mapHeight * 4)
        for row in 0..<mapHeight {
            for col in 0..<mapWidth {
                let prob = map.probabilities[row * mapWidth + col]
                let (r, g, b) = heatmapRGB(prob)
                let alpha = UInt8(max(0, min(255, Int(prob * 200))))  // max ~78% opacity
                let a = Float(alpha) / 255.0
                let i = (row * mapWidth + col) * 4
                pixels[i + 0] = UInt8(Float(r) * a)
                pixels[i + 1] = UInt8(Float(g) * a)
                pixels[i + 2] = UInt8(Float(b) * a)
                pixels[i + 3] = alpha
            }
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
            let overlayImage = CGImage(
                width: mapWidth, height: mapHeight,
                bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: mapWidth * 4,
                space: colorSpace, bitmapInfo: bitmapInfo,
                provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent
            )
        else { return nil }

        context.draw(overlayImage, in: CGRect(x: 0, y: 0, width: CGFloat(outWidth), height: CGFloat(outHeight)))
        return context.makeImage()
    }

    /// Composites colored instance masks from `segments` over `baseImage`.
    ///
    /// Each segment receives a distinct hue at ~60% opacity, so the original image
    /// remains visible underneath. Segments are drawn in array order (index 0 on top).
    ///
    /// The returned image matches `baseImage`'s dimensions; the small mask grid is
    /// upscaled by Core Graphics when composited.
    ///
    /// - Parameters:
    ///   - baseImage: The original input image.
    ///   - segments: Instance segments, typically sorted by score descending.
    /// - Returns: A composited `CGImage` at `baseImage` resolution, or `nil` if the context could not be created.
    public static func renderInstanceMasks(onto baseImage: CGImage, segments: [Segment]) -> CGImage? {
        guard !segments.isEmpty,
            let first = segments.first,
            first.maskWidth > 0, first.maskHeight > 0
        else { return nil }

        let maskWidth = first.maskWidth
        let maskHeight = first.maskHeight
        let outWidth = baseImage.width
        let outHeight = baseImage.height

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard
            let context = CGContext(
                data: nil,
                width: outWidth, height: outHeight,
                bitsPerComponent: 8, bytesPerRow: outWidth * 4,
                space: colorSpace, bitmapInfo: bitmapInfo.rawValue
            )
        else { return nil }

        context.draw(baseImage, in: CGRect(x: 0, y: 0, width: CGFloat(outWidth), height: CGFloat(outHeight)))

        // Overlay buffer is premultipliedLast: store premultiplied RGB so the context's
        // blend treats the values correctly. Each segment over-composites onto the buffer:
        //     C_out = C_new + C_old * (1 - a_new)
        //     a_out = a_new + a_old * (1 - a_new)
        var pixels = [UInt8](repeating: 0, count: maskWidth * maskHeight * 4)
        for (segIdx, segment) in segments.enumerated() {
            guard segment.mask.count == maskWidth * maskHeight else { continue }
            let (r, g, b) = instanceColor(index: segIdx, total: segments.count)
            let alphaByte: UInt8 = 153  // ~60% opacity
            let a = Float(alphaByte) / 255.0
            let inv = 1 - a
            let segR = Float(r) * a
            let segG = Float(g) * a
            let segB = Float(b) * a
            pixels.withUnsafeMutableBufferPointer { buf in
                let ptr = buf.baseAddress!
                for i in 0..<(maskWidth * maskHeight) where segment.mask[i] {
                    let base = i * 4
                    ptr[base + 0] = UInt8(min(255, segR + Float(ptr[base + 0]) * inv))
                    ptr[base + 1] = UInt8(min(255, segG + Float(ptr[base + 1]) * inv))
                    ptr[base + 2] = UInt8(min(255, segB + Float(ptr[base + 2]) * inv))
                    ptr[base + 3] = UInt8(min(255, (a + Float(ptr[base + 3]) / 255 * inv) * 255))
                }
            }
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
            let overlayImage = CGImage(
                width: maskWidth, height: maskHeight,
                bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: maskWidth * 4,
                space: colorSpace, bitmapInfo: bitmapInfo,
                provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent
            )
        else { return nil }

        context.draw(overlayImage, in: CGRect(x: 0, y: 0, width: CGFloat(outWidth), height: CGFloat(outHeight)))
        return context.makeImage()
    }

    /// Strokes input prompt boxes onto `baseImage` so callers can see what they asked for.
    ///
    /// - Parameters:
    ///   - baseImage: Image to draw onto.
    ///   - boxes: Boxes in input-image pixel coordinates with **top-left origin**, regardless of platform.
    ///   - color: Stroke color RGB in `[0, 255]`. Defaults to red.
    ///   - lineWidth: Stroke width in pixels.
    /// - Returns: A new `CGImage` with the boxes stroked, or `nil` if the context could not be created.
    public static func renderPromptBoxes(
        onto baseImage: CGImage,
        boxes: [CGRect],
        color: (r: UInt8, g: UInt8, b: UInt8) = (255, 0, 0),
        lineWidth: CGFloat = 3
    ) -> CGImage? {
        guard !boxes.isEmpty else { return baseImage }

        let width = baseImage.width
        let height = baseImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard
            let context = CGContext(
                data: nil,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: colorSpace, bitmapInfo: bitmapInfo.rawValue
            )
        else { return nil }

        context.draw(baseImage, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))

        // CGContext y-axis points up; box coords are top-left origin → flip Y.
        context.setStrokeColor(
            red: CGFloat(color.r) / 255.0,
            green: CGFloat(color.g) / 255.0,
            blue: CGFloat(color.b) / 255.0,
            alpha: 1.0
        )
        context.setLineWidth(lineWidth)
        for box in boxes {
            let flipped = CGRect(
                x: box.origin.x,
                y: CGFloat(height) - box.origin.y - box.size.height,
                width: box.size.width,
                height: box.size.height
            )
            context.stroke(flipped)
        }
        return context.makeImage()
    }

    /// Blue (0.0) → green (0.5) → red (1.0) heat-map color.
    static func heatmapRGB(_ prob: Float) -> (UInt8, UInt8, UInt8) {
        OverlayPalette.heatmapRGB(prob)
    }

    /// Evenly-spaced hue wheel color for a given segment index, premultiplied-ready (not premultiplied).
    static func instanceColor(index: Int, total: Int) -> (UInt8, UInt8, UInt8) {
        OverlayPalette.color(forIndex: index, total: total)
    }

    /// HSV → RGB, all components in [0, 1]. Returns UInt8 tuple.
    static func hsvToRGB(h: Float, s: Float, v: Float) -> (UInt8, UInt8, UInt8) {
        OverlayPalette.hsvToRGB(h: h, s: s, v: v)
    }
}
