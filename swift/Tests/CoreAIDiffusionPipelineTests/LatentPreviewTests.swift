// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics
import XCTest

@testable import CoreAIDiffusionPipeline

final class LatentPreviewTests: XCTestCase {
    // MARK: - LatentRGBCoefficients

    func testCoefficientsInit() {
        let coeffs = LatentRGBCoefficients(
            channels: 2,
            weights: [0.3, 0.2, 0.1, -0.1, 0.3, 0.2],
            bias: [0.5, 0.5, 0.5]
        )
        XCTAssertEqual(coeffs.channels, 2)
        XCTAssertEqual(coeffs.weights.count, 6)
    }

    func testFlux2CoefficientsHave16Channels() {
        XCTAssertEqual(LatentRGBCoefficients.flux2.channels, 32)
        XCTAssertEqual(LatentRGBCoefficients.flux2.weights.count, 96)  // 16 * 3
        XCTAssertEqual(LatentRGBCoefficients.flux2.bias.count, 3)
    }

    func testSD1CoefficientsHave4Channels() {
        XCTAssertEqual(LatentRGBCoefficients.sd1.channels, 4)
        XCTAssertEqual(LatentRGBCoefficients.sd1.weights.count, 12)  // 4 * 3
    }

    // MARK: - Draft Preview

    func testDraftPreviewProducesCorrectSize() {
        let shape = [1, 4, 8, 8]
        let count = shape.reduce(1, *)
        var nd = NDArray(shape: shape, scalarType: .float32)
        nd.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count { ptr[i] = Float.random(in: -1...1) }
        }

        let image = nd.asRGB(.draft, coefficients: .sd1)
        XCTAssertNotNil(image)
        XCTAssertEqual(image?.width, 8)
        XCTAssertEqual(image?.height, 8)
    }

    func testDraftPreview16ChannelLatent() {
        let shape = [1, 32, 4, 4]
        let count = shape.reduce(1, *)
        var nd = NDArray(shape: shape, scalarType: .float32)
        nd.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count { ptr[i] = Float.random(in: -1...1) }
        }

        let image = nd.asRGB(.draft, coefficients: .flux2)
        XCTAssertNotNil(image)
        XCTAssertEqual(image?.width, 4)
        XCTAssertEqual(image?.height, 4)
    }

    func testDraftPreviewChannelMismatchReturnsNil() {
        let shape = [1, 4, 8, 8]
        let count = shape.reduce(1, *)
        var nd = NDArray(shape: shape, scalarType: .float32)
        nd.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count { ptr[i] = 0 }
        }

        let image = nd.asRGB(.draft, coefficients: .flux2)
        XCTAssertNil(image, "4-ch latent with 16-ch coefficients should return nil")
    }

    func testDraftPreviewWrongRankReturnsNil() {
        let shape = [4, 8, 8]
        let count = shape.reduce(1, *)
        var nd = NDArray(shape: shape, scalarType: .float32)
        nd.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count { ptr[i] = 0 }
        }

        let image = nd.asRGB(.draft, coefficients: .sd1)
        XCTAssertNil(image, "3D tensor should return nil")
    }

    func testDraftPreviewNoCoefficientsReturnsNil() {
        let shape = [1, 4, 8, 8]
        let count = shape.reduce(1, *)
        var nd = NDArray(shape: shape, scalarType: .float32)
        nd.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count { ptr[i] = 0 }
        }

        let image = nd.asRGB(.draft)
        XCTAssertNil(image, "Draft without coefficients should return nil")
    }

    func testFullPreviewReturnsNil() {
        let shape = [1, 4, 8, 8]
        let count = shape.reduce(1, *)
        var nd = NDArray(shape: shape, scalarType: .float32)
        nd.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count { ptr[i] = 0 }
        }

        let image = nd.asRGB(.full)
        XCTAssertNil(image, ".full without decoder returns nil")
    }

    // MARK: - Pixel Sanity

    func testDraftPreviewDifferentLatentsProduceDifferentImages() {
        let shape = [1, 4, 2, 2]

        // Latent A: strong positive channel 0
        var ndA = NDArray(shape: shape, scalarType: .float32)
        ndA.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            ptr[0] = 2.0
            ptr[1] = 0
            ptr[2] = 0
            ptr[3] = 0
            for i in 4..<16 { ptr[i] = 0 }
        }

        // Latent B: all zeros
        var ndB = NDArray(shape: shape, scalarType: .float32)
        ndB.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<16 { ptr[i] = 0 }
        }

        let imageA = ndA.asRGB(.draft, coefficients: .sd1)
        let imageB = ndB.asRGB(.draft, coefficients: .sd1)
        XCTAssertNotNil(imageA)
        XCTAssertNotNil(imageB)

        // Images should differ (different inputs → different RGB)
        guard let a = imageA, let b = imageB,
            let dpA = a.dataProvider, let dpB = b.dataProvider,
            let dataA = dpA.data, let dataB = dpB.data
        else {
            XCTFail("Cannot read pixels")
            return
        }
        XCTAssertNotEqual(dataA as Data, dataB as Data, "Different latents should produce different images")
    }

    // MARK: - PipelineProgress

    func testProgressDefaultLatentIsNil() {
        let p = PipelineProgress(step: 1, totalSteps: 10)
        XCTAssertNil(p.currentLatent)
    }

    func testProgressAcceptsLatent() {
        var nd = NDArray(shape: [1, 4, 8, 8], scalarType: .float32)
        nd.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            ptr[0] = 42.0
        }
        let p = PipelineProgress(step: 5, totalSteps: 20, currentLatent: nd)
        XCTAssertNotNil(p.currentLatent)
        XCTAssertEqual(p.currentLatent?.shape, [1, 4, 8, 8])
    }
}
