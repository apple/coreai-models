// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Testing

@testable import CoreAIImageSegmenter

@Suite("Image Segmentation")
struct ImageSegmentationTests {
    // MARK: - PointQuery

    @Test("PointQuery: empty init is segment-everything")
    func pointQueryEmptyDefault() {
        let pointQuery = PointQuery()
        #expect(pointQuery.queries.isEmpty)
    }

    @Test("PointQuery(points:) wraps a flat list as a single query")
    func pointQuerySinglePoint() {
        let pointQuery = PointQuery(points: [.init(x: 10, y: 20)])
        #expect(pointQuery.queries.count == 1)
        #expect(pointQuery.queries[0].count == 1)
    }

    @Test("PointQuery(points:) keeps a box prompt in one query (Q=1, P=2)")
    func pointQueryBoxPrompt() {
        let pointQuery = PointQuery(points: [
            .init(x: 0, y: 0, label: .boxTopLeft),
            .init(x: 50, y: 50, label: .boxBottomRight),
        ])
        #expect(pointQuery.queries.count == 1)
        #expect(pointQuery.queries[0].count == 2)
        #expect(pointQuery.queries[0][0].label == .boxTopLeft)
        #expect(pointQuery.queries[0][1].label == .boxBottomRight)
    }

    @Test("PointQuery(queries:) preserves explicit per-query layout")
    func pointQueryExplicitQueries() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 1, y: 1), .init(x: 2, y: 2)],
            [.init(x: 3, y: 3)],
        ])
        #expect(pointQuery.queries.count == 2)
        #expect(pointQuery.queries[0].count == 2)
        #expect(pointQuery.queries[1].count == 1)
    }

    @Test("PointQuery(points: []) yields a single empty query, not segment-everything")
    func pointQueryEmptyPointsBecomesSingleEmptyQuery() {
        let pointQuery = PointQuery(points: [])
        #expect(pointQuery.queries.count == 1)
        #expect(pointQuery.queries[0].isEmpty)
    }

    @Test("PointQuery() with no args triggers segment-everything")
    func pointQueryDefaultInitIsSegmentEverything() {
        let pointQuery = PointQuery()
        #expect(pointQuery.queries.isEmpty)
    }

    @Test("Decode: predictedScores override sigmoid(logits) path")
    func decodeUsesPredictedScoresWhenPresent() {
        // Two queries; predictedScores are *already* in [0, 1]. The presence logit is also
        // provided but should be ignored when predictedScores is non-empty.
        let output = SegmentationOutput(
            predictedMasks: [Float](repeating: 0, count: 1 * 2 * 2 * 2),
            masksShape: [1, 2, 2, 2],
            predictedBoxes: [],  // EfficientSAM-style: no model-emitted boxes
            predictedLogits: [],
            predictedScores: [0.9, 0.4],
            presenceLogits: [-10.0],  // would zero scores via sigmoid if we used the SAM3 path
            semanticSegment: [],
            semanticSegmentShape: []
        )

        let response = SegmentationPostprocessor.decode(
            output: output,
            inputSize: CGSize(width: 4, height: 4),
            parameters: SegmentationParameters(maxSegments: 2)
        )

        #expect(response.segments.count == 2)
        #expect(abs(response.segments[0].score - 0.9) < 1e-5)
        #expect(abs(response.segments[1].score - 0.4) < 1e-5)
        // Empty predictedBoxes → CGRect.zero.
        #expect(response.segments[0].box == .zero)
    }

    // MARK: - SegmentationPostprocessor

    @Test("Decode: absent presence logits treated as 1.0")
    func decodeEmptyPresenceLogits() {
        let logit: Float = 2.0
        let expectedScore = SegmentationPostprocessor.sigmoid(logit)

        let output = SegmentationOutput(
            predictedMasks: [Float](repeating: 0, count: 1 * 1 * 2 * 2),
            masksShape: [1, 1, 2, 2],
            predictedBoxes: [0.0, 0.0, 1.0, 1.0],
            predictedLogits: [logit],
            presenceLogits: [],
            semanticSegment: [],
            semanticSegmentShape: []
        )

        let response = SegmentationPostprocessor.decode(
            output: output,
            inputSize: CGSize(width: 4, height: 4)
        )

        #expect(response.segments.count == 1)
        #expect(abs(response.segments[0].score - expectedScore) < 1e-5)
    }

    @Test("Decode: segments sorted by score descending")
    func decodeScoresSortedDescending() {
        let output = SegmentationOutput(
            predictedMasks: [Float](repeating: 0, count: 1 * 2 * 2 * 2),
            masksShape: [1, 2, 2, 2],
            predictedBoxes: [Float](repeating: 0, count: 1 * 2 * 4),
            predictedLogits: [-2.0, 2.0],
            presenceLogits: [],
            semanticSegment: [],
            semanticSegmentShape: []
        )

        let response = SegmentationPostprocessor.decode(
            output: output,
            inputSize: CGSize(width: 4, height: 4),
            parameters: SegmentationParameters(maxSegments: 2)
        )

        #expect(response.segments.count == 2)
        #expect(response.segments[0].score > response.segments[1].score)
    }

    @Test("Decode: maxSegments caps output count")
    func decodeMaxSegmentsLimitsOutput() {
        let output = SegmentationOutput(
            predictedMasks: [Float](repeating: 0, count: 1 * 3 * 2 * 2),
            masksShape: [1, 3, 2, 2],
            predictedBoxes: [Float](repeating: 0, count: 1 * 3 * 4),
            predictedLogits: [1.0, 2.0, 3.0],
            presenceLogits: [],
            semanticSegment: [],
            semanticSegmentShape: []
        )

        let response = SegmentationPostprocessor.decode(
            output: output,
            inputSize: CGSize(width: 4, height: 4),
            parameters: SegmentationParameters(maxSegments: 1)
        )

        #expect(response.segments.count == 1)
    }

    @Test("Decode: undersized predictedMasks bails out with empty response (no crash)")
    func decodeUndersizedMasksBailsOut() {
        // shape claims [1, 2, 3, 3] = 18 floats, but predictedMasks only has 4.
        let output = SegmentationOutput(
            predictedMasks: [Float](repeating: 0, count: 4),
            masksShape: [1, 2, 3, 3],
            predictedBoxes: [],
            predictedLogits: [0.5, 0.5],
            presenceLogits: [],
            semanticSegment: [],
            semanticSegmentShape: []
        )
        let response = SegmentationPostprocessor.decode(
            output: output,
            inputSize: CGSize(width: 4, height: 4)
        )
        #expect(response.segments.isEmpty)
    }

    @Test("Decode: undersized predictedScores bails out (direct-scores path)")
    func decodeUndersizedScoresBailsOut() {
        // shape implies queryCount=2 but predictedScores has only 1.
        let output = SegmentationOutput(
            predictedMasks: [Float](repeating: 0, count: 1 * 2 * 2 * 2),
            masksShape: [1, 2, 2, 2],
            predictedBoxes: [],
            predictedLogits: [],
            predictedScores: [0.9],  // missing slot for q=1
            presenceLogits: [],
            semanticSegment: [],
            semanticSegmentShape: []
        )
        let response = SegmentationPostprocessor.decode(
            output: output,
            inputSize: CGSize(width: 4, height: 4)
        )
        #expect(response.segments.isEmpty)
    }

    @Test("Decode: undersized predictedBoxes (non-empty but short) bails out")
    func decodeUndersizedBoxesBailsOut() {
        // queryCount=2 → expects 8 box floats; only 4 provided.
        let output = SegmentationOutput(
            predictedMasks: [Float](repeating: 0, count: 1 * 2 * 2 * 2),
            masksShape: [1, 2, 2, 2],
            predictedBoxes: [0, 0, 1, 1],
            predictedLogits: [0.5, 0.5],
            presenceLogits: [],
            semanticSegment: [],
            semanticSegmentShape: []
        )
        let response = SegmentationPostprocessor.decode(
            output: output,
            inputSize: CGSize(width: 4, height: 4)
        )
        #expect(response.segments.isEmpty)
    }

    @Test("Decode: predictedBoxes.isEmpty is fine (EfficientSAM path)")
    func decodeEmptyBoxesAllowed() {
        let output = SegmentationOutput(
            predictedMasks: [Float](repeating: 0, count: 1 * 1 * 2 * 2),
            masksShape: [1, 1, 2, 2],
            predictedBoxes: [],  // EfficientSAM: no boxes
            predictedLogits: [],
            predictedScores: [0.7],
            presenceLogits: [],
            semanticSegment: [],
            semanticSegmentShape: []
        )
        let response = SegmentationPostprocessor.decode(
            output: output,
            inputSize: CGSize(width: 4, height: 4)
        )
        #expect(response.segments.count == 1)
        #expect(response.segments[0].box == .zero)
    }

    // MARK: - bilinearUpsampleToBool / bilinearUpsampleToFloat

    @Test("Bilinear upsample (Bool): identity case (2×2 → 2×2) round-trips through threshold")
    func bilinearUpsampleBoolIdentity() {
        let source: [Float] = [0.3, 0.7, 0.8, 0.1]
        let result = SegmentationPostprocessor.bilinearUpsampleToBool(
            source: source, sourceHeight: 2, sourceWidth: 2, destinationHeight: 2, destinationWidth: 2,
            threshold: 0.5
        )
        // sH==dH, sW==dW collapses to source >= threshold per cell.
        #expect(result == [false, true, true, false])
    }

    @Test("Bilinear upsample (Bool): 2×2 → 4×4 produces gradients across cells, not staircase")
    func bilinearUpsampleBoolGradient() {
        // source (row-major):
        //   0.3 0.7
        //   0.8 0.1
        let source: [Float] = [0.3, 0.7, 0.8, 0.1]
        let result = SegmentationPostprocessor.bilinearUpsampleToBool(
            source: source, sourceHeight: 2, sourceWidth: 2, destinationHeight: 4, destinationWidth: 4,
            threshold: 0.5
        )

        #expect(result.count == 16)

        // Source corner samples land on destination corners (anchor invariant) — the bilinear
        // formula clamps so the dst (0,0) cell sees pure src[0,0] = 0.3 < 0.5 → false.
        #expect(!result[0 * 4 + 0])  // src[0,0] = 0.3 → false
        #expect(result[0 * 4 + 3])  // src[0,1] = 0.7 → true
        #expect(result[3 * 4 + 0])  // src[1,0] = 0.8 → true
        #expect(!result[3 * 4 + 3])  // src[1,1] = 0.1 → false

        // Mid-row interior cells blend between corners — verifies the resampling actually
        // interpolates rather than nearest-neighbor-snapping. Row 0 col 2 sees ~0.6 (between
        // 0.3 and 0.7), which crosses the 0.5 threshold; col 1 sees ~0.4, which doesn't.
        #expect(!result[0 * 4 + 1])
        #expect(result[0 * 4 + 2])
    }

    @Test("Bilinear upsample (Float): 2×2 → 4×4 produces smooth interpolated values")
    func bilinearUpsampleFloatGradient() {
        // source (row-major):
        //   0.0 1.0
        //   0.0 1.0
        // → expect a horizontal ramp at every destination row.
        let source: [Float] = [0.0, 1.0, 0.0, 1.0]
        let result = SegmentationPostprocessor.bilinearUpsampleToFloat(
            source: source, sourceHeight: 2, sourceWidth: 2, destinationHeight: 4, destinationWidth: 4
        )

        #expect(result.count == 16)
        // Row 0 should be the same horizontal ramp as row 3 (no vertical gradient in source).
        for row in 0..<4 {
            let r0 = result[row * 4 + 0]
            let r1 = result[row * 4 + 1]
            let r2 = result[row * 4 + 2]
            let r3 = result[row * 4 + 3]
            // Strictly increasing across columns proves bilinear interpolation rather than
            // staircase repetition.
            #expect(r0 < r1)
            #expect(r1 < r2)
            #expect(r2 < r3)
            // Source corners (clamped) hit destination corners.
            #expect(abs(r0 - 0.0) < 1e-6)
            #expect(abs(r3 - 1.0) < 1e-6)
        }
    }

    // MARK: - SegmentationVisualization

    @Test("heatmapRGB: pure blue at 0, green at 0.5, red at 1.0")
    func heatmapRGBBoundaryValues() {
        #expect(SegmentationVisualization.heatmapRGB(0.0) == (0, 0, 255))
        #expect(SegmentationVisualization.heatmapRGB(0.5) == (0, 255, 0))
        #expect(SegmentationVisualization.heatmapRGB(1.0) == (255, 0, 0))
        #expect(SegmentationVisualization.heatmapRGB(0.25) == (0, 127, 127))  // blue-green midpoint
    }

    // Reference values cross-checked against Python's `colorsys.hsv_to_rgb` (CPython 3.13).
    // Each Swift component is `UInt8(c * 255)` which truncates, so use the same `int(c*255)`
    // truncation when computing references.
    @Test("hsvToRGB: primaries and grayscale match colorsys reference")
    func hsvToRGBPrimaries() {
        // Pure red at the wrap-around (h=0).
        #expect(SegmentationVisualization.hsvToRGB(h: 0.0, s: 1.0, v: 1.0) == (255, 0, 0))
        // Pure green at h=120°.
        #expect(SegmentationVisualization.hsvToRGB(h: 1.0 / 3, s: 1.0, v: 1.0) == (0, 255, 0))
        // Pure blue at h=240°.
        #expect(SegmentationVisualization.hsvToRGB(h: 2.0 / 3, s: 1.0, v: 1.0) == (0, 0, 255))
        // Cyan, yellow, magenta.
        #expect(SegmentationVisualization.hsvToRGB(h: 0.5, s: 1.0, v: 1.0) == (0, 255, 255))
        #expect(SegmentationVisualization.hsvToRGB(h: 1.0 / 6, s: 1.0, v: 1.0) == (255, 255, 0))
        #expect(SegmentationVisualization.hsvToRGB(h: 5.0 / 6, s: 1.0, v: 1.0) == (255, 0, 255))
        // Saturation 0 collapses to grayscale at v.
        #expect(SegmentationVisualization.hsvToRGB(h: 0.0, s: 0.0, v: 1.0) == (255, 255, 255))
        #expect(SegmentationVisualization.hsvToRGB(h: 0.0, s: 0.0, v: 0.0) == (0, 0, 0))
    }

    @Test("instanceColor: index 0 of N is the (s=0.85, v=0.95) red anchor")
    func instanceColorAnchor() {
        // colorsys.hsv_to_rgb(0, 0.85, 0.95) = (0.95, 0.1425, 0.1425) → truncated (242, 36, 36).
        for total in [1, 4, 16] {
            #expect(SegmentationVisualization.instanceColor(index: 0, total: total) == (242, 36, 36))
        }
    }

    @Test("instanceColor: index N/2 of total=N lands on cyan")
    func instanceColorMidpoint() {
        // colorsys.hsv_to_rgb(0.5, 0.85, 0.95) = (0.1425, 0.95, 0.95) → truncated (36, 242, 242).
        #expect(SegmentationVisualization.instanceColor(index: 1, total: 2) == (36, 242, 242))
        #expect(SegmentationVisualization.instanceColor(index: 4, total: 8) == (36, 242, 242))
    }

    @Test("instanceColor: total=0 is treated as 1 (no divide-by-zero)")
    func instanceColorTotalZero() {
        // hue clamps to 0/1 → red anchor.
        #expect(SegmentationVisualization.instanceColor(index: 0, total: 0) == (242, 36, 36))
    }

    @Test("renderSemanticOverlay: output dimensions match the map")
    func renderSemanticOverlayDimensions() throws {
        let w = 4
        let h = 3
        let map = SemanticSegmentationMap(
            probabilities: [Float](repeating: 0.5, count: w * h),
            width: w, height: h
        )
        let base = try #require(makeSolidCGImage(width: w, height: h, r: 100, g: 100, b: 100))
        let overlay = try #require(SegmentationVisualization.renderSemanticOverlay(onto: base, map: map))

        #expect(overlay.width == w)
        #expect(overlay.height == h)
    }

    // MARK: - Sigmoid

    @Test("sigmoid(0) == 0.5 and sigmoid(x) + sigmoid(-x) == 1")
    func sigmoidKnownValues() {
        #expect(SegmentationPostprocessor.sigmoid(0) == 0.5)
        let x: Float = 2.5
        #expect(abs(SegmentationPostprocessor.sigmoid(x) + SegmentationPostprocessor.sigmoid(-x) - 1.0) < 1e-6)
    }

    // MARK: - Helpers

    private func makeSolidCGImage(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) -> CGImage? {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * bytesPerRow + x * bytesPerPixel
                pixels[i] = r
                pixels[i + 1] = g
                pixels[i + 2] = b
                pixels[i + 3] = 255
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}
