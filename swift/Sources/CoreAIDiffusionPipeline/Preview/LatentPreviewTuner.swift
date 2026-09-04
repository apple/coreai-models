// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation

/// Collects latent/RGB pairs during generation for fitting preview coefficients.
///
/// Usage:
///   let tuner = LatentPreviewTuner(outputDir: URL(...))
///   pipeline.generateImages(configuration: config) { progress in
///       if let latent = progress.currentLatent {
///           tuner.record(latent: latent, step: progress.step)
///       }
///       return true
///   }
///   // After generation, record the final decoded image:
///   tuner.recordDecoded(image: result.images[0])
///   // Fit coefficients from accumulated pairs:
///   let coefficients = tuner.fitCoefficients()
///   // Or export raw data for offline fitting:
///   try tuner.exportPairs()
public class LatentPreviewTuner {
    private let outputDir: URL
    private var latents: [(step: Int, data: [Float], shape: [Int])] = []
    private var decodedImage: CGImage?

    public init(outputDir: URL) {
        self.outputDir = outputDir
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    }

    /// Record a latent tensor from a progress callback.
    public func record(latent: NDArray, step: Int) {
        let view = latent.view(as: Float.self)
        var data = [Float](repeating: 0, count: latent.shape.reduce(1, *))
        view.withUnsafePointer { ptr, _, _ in
            for i in 0..<data.count { data[i] = ptr[i] }
        }
        latents.append((step: step, data: data, shape: latent.shape))
    }

    /// Record the final decoded image (ground truth RGB).
    public func recordDecoded(image: CGImage) {
        decodedImage = image
    }

    /// Export latent/image pairs as .npy files for offline coefficient fitting.
    ///
    /// Writes to outputDir:
    ///   latent_step_N.npy  — raw latent at each recorded step
    ///   decoded.png        — final decoded image
    public func exportPairs() throws {
        for (step, data, shape) in latents {
            let url = outputDir.appendingPathComponent("latent_step_\(step).bin")
            let header = shape.map(String.init).joined(separator: ",")
            var fileData = Data(header.utf8)
            fileData.append(Data(bytes: data, count: data.count * MemoryLayout<Float>.size))
            try fileData.write(to: url)
        }

        if let image = decodedImage {
            let url = outputDir.appendingPathComponent("decoded.png")
            guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
                return
            }
            CGImageDestinationAddImage(dest, image, nil)
            CGImageDestinationFinalize(dest)
        }
    }

    /// Fit a [C, 3] linear projection from the final-step latent to the decoded image.
    ///
    /// Uses least-squares regression: for each pixel, predict RGB from the C latent channels.
    /// Returns nil if no latent/image pair is available.
    public func fitCoefficients() -> LatentRGBCoefficients? {
        guard let lastLatent = latents.last,
            let image = decodedImage
        else { return nil }

        let shape = lastLatent.shape
        guard shape.count == 4, shape[0] == 1 else { return nil }
        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]
        let spatialCount = height * width

        // Extract RGB from the decoded image, downsampled to latent resolution
        guard let rgbData = extractRGB(from: image, width: width, height: height) else { return nil }

        // Least-squares: solve X * W = Y where X=[N, C], Y=[N, 3], W=[C, 3]
        // Using normal equations: W = (X^T X)^-1 X^T Y

        // Build X: transpose latent from CHW to NxC
        let latentData = lastLatent.data
        var x = [Float](repeating: 0, count: spatialCount * channels)
        for c in 0..<channels {
            for p in 0..<spatialCount {
                x[p * channels + c] = latentData[c * spatialCount + p]
            }
        }

        // Build Y: RGB in [N, 3] layout, normalized to [0, 1]
        var y = [Float](repeating: 0, count: spatialCount * 3)
        for p in 0..<spatialCount {
            y[p * 3 + 0] = Float(rgbData[p * 3 + 0]) / 255.0
            y[p * 3 + 1] = Float(rgbData[p * 3 + 1]) / 255.0
            y[p * 3 + 2] = Float(rgbData[p * 3 + 2]) / 255.0
        }

        // X^T X: [C, C]
        var xtx = [Float](repeating: 0, count: channels * channels)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(channels), Int32(channels), Int32(spatialCount),
            1.0, x, Int32(channels), x, Int32(channels),
            0.0, &xtx, Int32(channels)
        )

        // Add small ridge for numerical stability
        for i in 0..<channels {
            xtx[i * channels + i] += 1e-4
        }

        // X^T Y: [C, 3]
        var xty = [Float](repeating: 0, count: channels * 3)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(channels), 3, Int32(spatialCount),
            1.0, x, Int32(channels), y, 3,
            0.0, &xty, 3
        )

        // Solve (X^T X) W = X^T Y via LAPACK (symmetric positive definite)
        var n = Int32(channels)
        var nrhs = Int32(3)
        var info = Int32(0)
        var uplo = Int8(UInt8(ascii: "U"))
        sposv_(&uplo, &n, &nrhs, &xtx, &n, &xty, &n, &info)

        guard info == 0 else { return nil }

        // xty now contains the solution W: [C, 3]
        // Compute bias as mean residual
        var predicted = [Float](repeating: 0, count: spatialCount * 3)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(spatialCount), 3, Int32(channels),
            1.0, x, Int32(channels), xty, 3,
            0.0, &predicted, 3
        )

        var bias: [Float] = [0, 0, 0]
        for c in 0..<3 {
            var sum: Float = 0
            for p in 0..<spatialCount {
                sum += y[p * 3 + c] - predicted[p * 3 + c]
            }
            bias[c] = sum / Float(spatialCount)
        }

        return LatentRGBCoefficients(channels: channels, weights: xty, bias: bias)
    }

    /// Downsample a CGImage to the target resolution and extract interleaved RGB bytes.
    private func extractRGB(from image: CGImage, width: Int, height: Int) -> [UInt8]? {
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard
            let context = CGContext(
                data: &pixels,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )
        else { return nil }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBX → RGB
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3 + 0] = pixels[i * 4 + 0]
            rgb[i * 3 + 1] = pixels[i * 4 + 1]
            rgb[i * 3 + 2] = pixels[i * 4 + 2]
        }
        return rgb
    }
}
