// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreGraphics
import Foundation

/// Core AI latent decoder — wraps a VAE decoder model function.
public final class CoreAILatentDecoder: Sendable {
    public let function: CoreAIDiffusionModelFunction

    public init(function: CoreAIDiffusionModelFunction) {
        self.function = function
    }

    public func loadResources() async throws {
        try await function.loadResources()
    }

    public func unloadResources() async {
        await function.unloadResources()
    }

    public func decode(_ latents: NDArray, scaleFactor: Float, shiftFactor: Float) async throws -> NDArray {
        // Read latents, scale, run model, return as NDArray
        var shape: [Int] = []
        var inputFloats = [Float]()
        latents.view(as: Float.self).withUnsafePointer { ptr, s, _ in
            shape = (0..<s.count).map { s[$0] }
            let count = shape.reduce(1, *)
            inputFloats.reserveCapacity(count)
            for i in 0..<count {
                inputFloats.append(ptr[i] / scaleFactor + shiftFactor)
            }
        }

        let outputFloats = try await function.run(floatInputs: [(inputFloats, shape)])

        // Infer output shape: [1, 3, H*8, W*8] for VAE decoder
        let outH = shape[2] * 8
        let outW = shape[3] * 8
        let outShape = [1, 3, outH, outW]
        var result = NDArray(shape: outShape, scalarType: .float32)
        let resultView = result.mutableView(as: Float.self)
        resultView.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<outputFloats.count { ptr[i] = outputFloats[i] }
        }
        return result
    }
}

/// Core AI latent encoder — wraps a VAE encoder model function (for img2img).
public final class CoreAILatentEncoder: Sendable {
    public let function: CoreAIDiffusionModelFunction

    public init(function: CoreAIDiffusionModelFunction) {
        self.function = function
    }

    public func loadResources() async throws {
        try await function.loadResources()
    }

    public func unloadResources() async {
        await function.unloadResources()
    }

    public func encode(_ image: CGImage, scaleFactor: Float) async throws -> NDArray {
        let width = image.width
        let height = image.height

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: 4 * width,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
            let ptr = context.data?.bindMemory(to: UInt8.self, capacity: width * height * 4)
        else {
            throw CoreAIComponentError.imageConversionFailed
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let pixelCount = height * width
        var redChannel = [Float](repeating: 0, count: pixelCount)
        var greenChannel = [Float](repeating: 0, count: pixelCount)
        var blueChannel = [Float](repeating: 0, count: pixelCount)

        vDSP_vfltu8(ptr, 4, &redChannel, 1, vDSP_Length(pixelCount))
        vDSP_vfltu8(ptr + 1, 4, &greenChannel, 1, vDSP_Length(pixelCount))
        vDSP_vfltu8(ptr + 2, 4, &blueChannel, 1, vDSP_Length(pixelCount))

        vDSP.divide(redChannel, 127.5, result: &redChannel)
        vDSP.add(-1.0, redChannel, result: &redChannel)
        vDSP.divide(greenChannel, 127.5, result: &greenChannel)
        vDSP.add(-1.0, greenChannel, result: &greenChannel)
        vDSP.divide(blueChannel, 127.5, result: &blueChannel)
        vDSP.add(-1.0, blueChannel, result: &blueChannel)

        let inputFloats = redChannel + greenChannel + blueChannel
        let outputFloats = try await function.run(floatInputs: [(inputFloats, [1, 3, height, width])])

        // Scale and return as NDArray
        let outShape = [1, 4, height / 8, width / 8]
        var result = NDArray(shape: outShape, scalarType: .float32)
        let resultView = result.mutableView(as: Float.self)
        resultView.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<outputFloats.count { ptr[i] = outputFloats[i] * scaleFactor }
        }
        return result
    }
}
