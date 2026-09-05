// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAIShared

/// Direct coverage for ``OverlayPalette``.
///
/// `ImageSegmenterTests` already pins these values, but through
/// `SegmentationVisualization`'s wrappers — so the shared type had no coverage that survives
/// those wrappers being removed. This suite owns the anchors and the range edges; the image
/// segmenter's tests stay as they are, checking that its wrappers still delegate.
@Suite("OverlayPalette")
struct OverlayPaletteTests {
    @Test("heatmapRGB runs blue to green to red")
    func heatmapAnchors() {
        #expect(OverlayPalette.heatmapRGB(0.0) == (0, 0, 255))
        #expect(OverlayPalette.heatmapRGB(0.5) == (0, 255, 0))
        #expect(OverlayPalette.heatmapRGB(1.0) == (255, 0, 0))
    }

    @Test("heatmapRGB clamps instead of wrapping or overflowing")
    func heatmapClamps() {
        // Callers pass raw probabilities. A logit that hasn't been squashed, or a -1
        // sentinel, must saturate at the ends — `UInt8(r * 255)` traps on out-of-range.
        #expect(OverlayPalette.heatmapRGB(-0.5) == (0, 0, 255))
        #expect(OverlayPalette.heatmapRGB(-1000) == (0, 0, 255))
        #expect(OverlayPalette.heatmapRGB(1.5) == (255, 0, 0))
        #expect(OverlayPalette.heatmapRGB(1000) == (255, 0, 0))
    }

    @Test("hsvToRGB wraps at h=1 back to the h=0 hue")
    func hueWrapsAtOne() {
        // `Int(h6) % 6` indexes the sextant table, and h=1 puts h6 exactly on 6. Falling
        // into the default branch there would return magenta instead of red.
        #expect(OverlayPalette.hsvToRGB(h: 1.0, s: 1.0, v: 1.0) == (255, 0, 0))
        #expect(
            OverlayPalette.hsvToRGB(h: 1.0, s: 0.85, v: 0.95)
                == OverlayPalette.hsvToRGB(h: 0.0, s: 0.85, v: 0.95))
    }

    @Test("hsvToRGB collapses to grey when saturation is zero")
    func zeroSaturationIsGrey() {
        for hue in [Float(0), 0.25, 0.5, 0.75] {
            #expect(OverlayPalette.hsvToRGB(h: hue, s: 0, v: 0.5) == (127, 127, 127))
        }
    }

    @Test("color(forIndex:total:) anchors index 0 at red")
    func indexZeroIsRed() {
        // colorsys.hsv_to_rgb(0, 0.85, 0.95) = (0.95, 0.1425, 0.1425) → truncated.
        for total in [1, 4, 16] {
            #expect(OverlayPalette.color(forIndex: 0, total: total) == (242, 36, 36))
        }
    }

    @Test("color(forIndex:total:) gives every segment a distinct colour")
    func indicesAreDistinct() {
        // The whole point is telling instances apart in one overlay. Two segments landing
        // on the same hue would read as one blob.
        for total in [2, 6, 12] {
            let colors = (0..<total).map { OverlayPalette.color(forIndex: $0, total: total) }
            let unique = Set(colors.map { "\($0.0),\($0.1),\($0.2)" })
            #expect(unique.count == total, "total \(total) produced \(unique.count) colours")
        }
    }

    @Test("color(forIndex:total:) survives a zero total")
    func totalZeroDoesNotDivideByZero() {
        #expect(OverlayPalette.color(forIndex: 0, total: 0) == (242, 36, 36))
    }
}
