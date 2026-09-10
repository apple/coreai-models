// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Colors for compositing masks over an image.
public enum OverlayPalette {
    /// Blue (0.0) → green (0.5) → red (1.0) heat-map color.
    public static func heatmapRGB(_ probability: Float) -> (UInt8, UInt8, UInt8) {
        let p = max(0, min(1, probability))
        let r: Float
        let g: Float
        let b: Float
        if p < 0.5 {
            let t = p * 2  // 0→1 over first half
            r = 0
            g = t
            b = 1 - t
        } else {
            let t = (p - 0.5) * 2  // 0→1 over second half
            r = t
            g = 1 - t
            b = 0
        }
        return (UInt8(r * 255), UInt8(g * 255), UInt8(b * 255))
    }

    /// Evenly-spaced hue wheel color for position `index` out of `total`.
    public static func color(forIndex index: Int, total: Int) -> (UInt8, UInt8, UInt8) {
        let count = max(1, total)
        return hsvToRGB(h: Float(index) / Float(count), s: 0.85, v: 0.95)
    }

    /// HSV → RGB, all components in [0, 1].
    public static func hsvToRGB(h: Float, s: Float, v: Float) -> (UInt8, UInt8, UInt8) {
        let h6 = h * 6
        let i = Int(h6) % 6
        let f = h6 - floor(h6)
        let p = v * (1 - s)
        let q = v * (1 - s * f)
        let t = v * (1 - s * (1 - f))
        let (r, g, b): (Float, Float, Float)
        switch i {
        case 0: (r, g, b) = (v, t, p)
        case 1: (r, g, b) = (q, v, p)
        case 2: (r, g, b) = (p, v, t)
        case 3: (r, g, b) = (p, q, v)
        case 4: (r, g, b) = (t, p, v)
        default: (r, g, b) = (v, p, q)
        }
        return (UInt8(r * 255), UInt8(g * 255), UInt8(b * 255))
    }
}
