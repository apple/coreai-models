// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation
import Testing

@testable import CoreAILanguageModels

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
private let testScalarType: NDArray.ScalarType = .float16
#else
private let testScalarType: NDArray.ScalarType = .float32
#endif

/// Synthesizes a solid-color CGImage in memory, with no file I/O and no network.
private func makeSolidCGImage(width: Int, height: Int) -> CGImage {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    ctx.setFillColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return ctx.makeImage()!
}

/// Synthesizes an image from a per-pixel color function. `row` 0 is the top row of the
/// image (row 0 of the backing buffer).
private func makeCGImage(
    width: Int, height: Int, color: (_ row: Int, _ col: Int) -> (r: UInt8, g: UInt8, b: UInt8)
) -> CGImage {
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    for row in 0..<height {
        for col in 0..<width {
            let c = color(row, col)
            let i = (row * width + col) * 4
            bytes[i] = c.r
            bytes[i + 1] = c.g
            bytes[i + 2] = c.b
            bytes[i + 3] = 255
        }
    }
    let provider = CGDataProvider(data: Data(bytes) as CFData)!
    return CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
}

private let ladder = [384, 768, 1152]

@Suite("Gemma4 vision size-ladder Tests")
struct VisionSizeLadderTests {
    // MARK: - chooseImageSize

    @Test("Picks the smallest size that fits without upscaling")
    func choosesSmallestFittingSize() {
        let image = makeSolidCGImage(width: 300, height: 300)
        let chosen = StaticShapeEngine.chooseImageSize(for: image, imageSizes: ladder)
        #expect(chosen == 384)
    }

    @Test("Picks the exact size on a boundary match")
    func choosesExactBoundarySize() {
        let image = makeSolidCGImage(width: 768, height: 500)
        let chosen = StaticShapeEngine.chooseImageSize(for: image, imageSizes: ladder)
        #expect(chosen == 768)
    }

    @Test("Falls back to the largest size when the image exceeds every size")
    func fallsBackToLargestSize() {
        let image = makeSolidCGImage(width: 2000, height: 1500)
        let chosen = StaticShapeEngine.chooseImageSize(for: image, imageSizes: ladder)
        #expect(chosen == 1152)
    }

    @Test("Uses the longer side, regardless of orientation")
    func usesLongerSide() {
        // Portrait: height is the long side.
        let image = makeSolidCGImage(width: 300, height: 900)
        let chosen = StaticShapeEngine.chooseImageSize(for: image, imageSizes: ladder)
        #expect(chosen == 1152)
    }

    // MARK: - letterbox

    @Test("Never upscales; real extent rounds up to the block size")
    func letterboxDoesNotUpscale() throws {
        let image = makeSolidCGImage(width: 100, height: 100)
        let result = try StaticShapeEngine.letterbox(cgImage: image, bucketSize: 384, blockPx: 48)
        // 100 rounds up to the next 48-multiple (144), not down to 100 and not up to 384.
        #expect(result.realWidthPx == 144)
        #expect(result.realHeightPx == 144)
        #expect(result.chw.count == 3 * 384 * 384)
    }

    @Test("Downscales an oversized image so its long side fits the bucket")
    func letterboxDownscalesOversizedImage() throws {
        let image = makeSolidCGImage(width: 2000, height: 1000)
        let result = try StaticShapeEngine.letterbox(cgImage: image, bucketSize: 384, blockPx: 48)
        #expect(result.realWidthPx <= 384)
        #expect(result.realHeightPx <= 384)
        // Long side (width, 2000) should have scaled down to essentially fill the bucket.
        #expect(result.realWidthPx == 384)
        // Short side scales proportionally: 1000 * (384/2000) = 192, already block-aligned.
        #expect(result.realHeightPx == 192)
    }

    @Test("Image is drawn at its own size; the rest of the real extent and the canvas are zero")
    func letterboxZerosPadding() throws {
        let bucketSize = 192
        let blockPx = 48
        let image = makeSolidCGImage(width: 100, height: 100)
        let result = try StaticShapeEngine.letterbox(cgImage: image, bucketSize: bucketSize, blockPx: blockPx)
        #expect(result.realWidthPx == 144)
        #expect(result.realHeightPx == 144)

        // CHW layout: chw[c*H*W + row*W + col].
        func pixel(_ channel: Int, _ row: Int, _ col: Int) -> Float {
            result.chw[channel * bucketSize * bucketSize + row * bucketSize + col]
        }
        #expect(pixel(0, 0, 0) > 0)
        #expect(pixel(0, 99, 99) > 0)  // last drawn pixel
        #expect(pixel(0, 100, 100) == 0)  // inside the block-rounded extent, beyond the image
        #expect(pixel(0, bucketSize - 1, bucketSize - 1) == 0)  // beyond both real extents
        #expect(pixel(0, 0, bucketSize - 1) == 0)  // beyond the real width, within real height
        #expect(pixel(0, bucketSize - 1, 0) == 0)  // beyond the real height, within real width
    }

    @Test("Small image keeps its aspect ratio and size instead of stretching to the block-rounded rect")
    func letterboxDoesNotStretchSmallImage() throws {
        let bucketSize = 192
        let image = makeSolidCGImage(width: 50, height: 50)
        let result = try StaticShapeEngine.letterbox(cgImage: image, bucketSize: bucketSize, blockPx: 48)
        #expect(result.realWidthPx == 96)
        #expect(result.realHeightPx == 96)

        func pixel(_ row: Int, _ col: Int) -> Float { result.chw[row * bucketSize + col] }
        #expect(pixel(49, 49) > 0)
        #expect(pixel(50, 50) == 0)
        #expect(pixel(0, 50) == 0)
        #expect(pixel(50, 0) == 0)
    }

    @Test("Content is anchored at the top-left of the buffer, not flipped")
    func letterboxAnchorsTopLeftWithoutFlip() throws {
        let bucketSize = 192
        // Top half red, bottom half blue; left half bright, right half dim in the green channel.
        let image = makeCGImage(width: 96, height: 96) { row, col in
            (r: row < 48 ? 255 : 0, g: col < 48 ? 200 : 100, b: row < 48 ? 0 : 255)
        }
        let result = try StaticShapeEngine.letterbox(cgImage: image, bucketSize: bucketSize, blockPx: 48)
        #expect(result.realWidthPx == 96)
        #expect(result.realHeightPx == 96)

        func pixel(_ channel: Int, _ row: Int, _ col: Int) -> Float {
            result.chw[channel * bucketSize * bucketSize + row * bucketSize + col]
        }
        // Top-left quadrant: red, bright green.
        #expect(pixel(0, 10, 10) > 0.9)
        #expect(pixel(2, 10, 10) < 0.1)
        #expect(pixel(1, 10, 10) > 0.7)
        // Bottom-right quadrant: blue, dim green.
        #expect(pixel(0, 85, 85) < 0.1)
        #expect(pixel(2, 85, 85) > 0.9)
        #expect(pixel(1, 85, 85) < 0.5)
        // Top-right and bottom-left quadrants.
        #expect(pixel(0, 10, 85) > 0.9)
        #expect(pixel(2, 85, 10) > 0.9)
        // Padding below and to the right of the content is zero.
        #expect(pixel(0, 120, 10) == 0 && pixel(2, 120, 10) == 0)
        #expect(pixel(0, 10, 120) == 0 && pixel(2, 10, 120) == 0)
    }

    @Test("Non-square image keeps its aspect ratio when downscaled and stays top-left anchored")
    func letterboxNonSquareDownscale() throws {
        let bucketSize = 192
        // 400x200: left half red, right half blue. Scales by 192/400 to 192x96.
        let image = makeCGImage(width: 400, height: 200) { _, col in
            (r: col < 200 ? 255 : 0, g: 0, b: col < 200 ? 0 : 255)
        }
        let result = try StaticShapeEngine.letterbox(cgImage: image, bucketSize: bucketSize, blockPx: 48)
        #expect(result.realWidthPx == 192)
        #expect(result.realHeightPx == 96)

        func pixel(_ channel: Int, _ row: Int, _ col: Int) -> Float {
            result.chw[channel * bucketSize * bucketSize + row * bucketSize + col]
        }
        #expect(pixel(0, 5, 10) > 0.9)  // left half, top rows
        #expect(pixel(2, 90, 180) > 0.9)  // right half, last content rows
        #expect(pixel(0, 100, 10) == 0 && pixel(2, 100, 180) == 0)  // below the content
    }

    // MARK: - buildAttentionMask

    @Test("Masks every padding patch as a key, leaves real keys and all queries unmasked")
    func buildAttentionMaskMarksPaddingKeys() {
        // 3x3 grid, real region top-left 2x2 -- not a contiguous prefix (matches the
        // python-side masking test's non-trivial case).
        let grid = 3
        let numPatches = grid * grid
        let mask = StaticShapeEngine.buildAttentionMask(
            numPatches: numPatches, grid: grid, realPatchWidth: 2, realPatchHeight: 2)
        #expect(mask.count == numPatches * numPatches)

        let realKeys: Set<Int> = [0, 1, 3, 4]  // (row,col) in {0,1}x{0,1}, flat = row*3+col
        for key in 0..<numPatches {
            let base = key * numPatches
            let expected: Float16 = realKeys.contains(key) ? 0 : causalMaskSentinel
            for q in 0..<numPatches {
                #expect(mask[base + q] == expected, "key=\(key) query=\(q)")
            }
        }
    }

    @Test("Fully-real grid produces an all-zero mask")
    func buildAttentionMaskAllRealIsAllZero() {
        let grid = 4
        let numPatches = grid * grid
        let mask = StaticShapeEngine.buildAttentionMask(
            numPatches: numPatches, grid: grid, realPatchWidth: grid, realPatchHeight: grid)
        #expect(mask.allSatisfy { $0 == 0 })
    }

    // MARK: - compactRealSoftTokens

    @Test("Selects the top-left real rectangle in row-major order, dropping padding cells")
    func compactRealSoftTokensSelectsRealRegion() throws {
        let pooledGrid = 3
        let hiddenDim = 2
        var outArray = NDArray(shape: [1, pooledGrid * pooledGrid, hiddenDim], scalarType: testScalarType)
        // Row i's two elements are both `i`, so the selected rows are identifiable by value.
        fillNDArray(&outArray, as: LogitsScalarType.self, count: pooledGrid * pooledGrid * hiddenDim) {
            LogitsScalarType($0 / hiddenDim)
        }

        let compacted = try StaticShapeEngine.compactRealSoftTokens(
            from: outArray, pooledGrid: pooledGrid, realPooledWidth: 2, realPooledHeight: 2)
        #expect(compacted.shape == [1, 4, hiddenDim])

        let values = readNDArray(compacted, as: LogitsScalarType.self, count: 4 * hiddenDim)
        // Real cells are row-major indices {0, 1, 3, 4} of the 3x3 pooled grid.
        let expectedRows: [LogitsScalarType] = [0, 1, 3, 4]
        for (i, expected) in expectedRows.enumerated() {
            #expect(values[i * hiddenDim] == expected)
            #expect(values[i * hiddenDim + 1] == expected)
        }
    }

    @Test("A fully-real grid is passed through as-is by the caller (no compaction needed)")
    func compactRealSoftTokensFullGridMatchesInput() throws {
        let pooledGrid = 2
        let hiddenDim = 3
        var outArray = NDArray(shape: [1, pooledGrid * pooledGrid, hiddenDim], scalarType: testScalarType)
        fillNDArray(&outArray, as: LogitsScalarType.self, count: pooledGrid * pooledGrid * hiddenDim) {
            LogitsScalarType($0)
        }

        let compacted = try StaticShapeEngine.compactRealSoftTokens(
            from: outArray, pooledGrid: pooledGrid, realPooledWidth: pooledGrid, realPooledHeight: pooledGrid)
        let expected = readNDArray(outArray, as: LogitsScalarType.self, count: pooledGrid * pooledGrid * hiddenDim)
        let got = readNDArray(compacted, as: LogitsScalarType.self, count: pooledGrid * pooledGrid * hiddenDim)
        #expect(got == expected)
    }
}
