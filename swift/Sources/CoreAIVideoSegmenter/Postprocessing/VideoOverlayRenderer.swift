// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import CoreText
import Foundation

/// Draws a frame's tracked objects over the source image.
///
/// Colours come from ``OverlayPalette/color(forID:)``, a pure function of the object id
/// rather than of its position in the array, so a track that survives an occlusion keeps its
/// colour.
struct VideoOverlayRenderer {
    private let parameters: VideoSegmentationParameters

    init(parameters: VideoSegmentationParameters) {
        self.parameters = parameters
    }

    /// Composite `objects` onto `frame`, returning a new image at the same size. Returns
    /// `frame` unchanged when there is nothing to draw.
    func render(_ objects: [TrackedObject], onto frame: CGImage) -> CGImage {
        guard !objects.isEmpty else { return frame }
        let width = frame.width
        let height = frame.height
        guard
            let context = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return frame }

        context.draw(frame, in: CGRect(x: 0, y: 0, width: width, height: height))

        if let overlay = maskOverlay(objects, width: width, height: height) {
            context.draw(overlay, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        // Core Graphics puts the origin at the bottom left; masks and boxes are top-left.
        // Flipping the whole context once keeps every draw below in image coordinates,
        // including the text, which would otherwise render upside down.
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)

        if parameters.drawBoxes {
            context.setLineWidth(parameters.strokeWidth)
            for object in objects where !object.box.isEmpty {
                let (r, g, b) = OverlayPalette.color(forID: object.id)
                context.setStrokeColor(
                    red: CGFloat(r) / 255, green: CGFloat(g) / 255, blue: CGFloat(b) / 255,
                    alpha: 1)
                context.stroke(object.box)
            }
        }
        if parameters.drawLabels {
            for object in objects where !object.box.isEmpty {
                drawLabel(for: object, in: context, imageHeight: height)
            }
        }

        return context.makeImage() ?? frame
    }

    /// Build one translucent RGBA layer holding every object's fill.
    ///
    /// Composited as a single image rather than one draw per object: at 1080p with eight
    /// tracks that is one blend instead of eight.
    private func maskOverlay(_ objects: [TrackedObject], width: Int, height: Int) -> CGImage? {
        let alpha = max(0, min(1, parameters.maskOpacity))
        guard alpha > 0 else { return nil }
        let inverse = 1 - alpha

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            for object in objects {
                guard object.mask.width == width, object.mask.height == height else { continue }
                let (r, g, b) = OverlayPalette.color(forID: object.id)
                // The buffer is premultipliedLast, so store colour already scaled by alpha
                // and over-composite: C_out = C_new + C_old * (1 - a_new).
                let red = Float(r) * alpha
                let green = Float(g) * alpha
                let blue = Float(b) * alpha
                object.mask.forEachSetIndex { index in
                    let offset = index * 4
                    base[offset] = UInt8(min(255, red + Float(base[offset]) * inverse))
                    base[offset + 1] = UInt8(min(255, green + Float(base[offset + 1]) * inverse))
                    base[offset + 2] = UInt8(min(255, blue + Float(base[offset + 2]) * inverse))
                    base[offset + 3] = UInt8(
                        min(255, (alpha + Float(base[offset + 3]) / 255 * inverse) * 255))
                }
            }
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// `#id prompt 0.92` on a filled chip above the box, tucked inside the frame when the
    /// box is at the top edge.
    private func drawLabel(for object: TrackedObject, in context: CGContext, imageHeight: Int) {
        let text = "#\(object.id) \(object.prompt) \(String(format: "%.2f", object.score))"
        let fontSize = max(11, min(28, parameters.strokeWidth * 5))
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, fontSize, nil)
        // Core Text attribute names rather than the AppKit/UIKit ones, so this file stays
        // buildable on both platforms without importing a UI framework.
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                CGColor(red: 1, green: 1, blue: 1, alpha: 1),
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: text, attributes: attributes))
        let bounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)

        let padding: CGFloat = 3
        let chipWidth = bounds.width + padding * 2
        let chipHeight = bounds.height + padding * 2
        var chipY = object.box.minY - chipHeight
        if chipY < 0 { chipY = object.box.minY }  // box hugs the top; put the chip inside

        let (r, g, b) = OverlayPalette.color(forID: object.id)
        context.setFillColor(
            red: CGFloat(r) / 255, green: CGFloat(g) / 255, blue: CGFloat(b) / 255, alpha: 0.85)
        context.fill(
            CGRect(x: object.box.minX, y: chipY, width: chipWidth, height: chipHeight))

        // The context is already y-flipped for image coordinates, so flip once more
        // locally or the glyphs come out mirrored.
        context.saveGState()
        context.textMatrix = .identity
        context.translateBy(x: object.box.minX + padding, y: chipY + chipHeight - padding)
        context.scaleBy(x: 1, y: -1)
        context.textPosition = CGPoint(x: 0, y: -bounds.minY - bounds.height)
        CTLineDraw(line, context)
        context.restoreGState()
    }
}
