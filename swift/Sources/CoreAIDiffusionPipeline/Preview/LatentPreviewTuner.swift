// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation
import ImageIO

/// Fits a linear latent-to-RGB projection for real-time diffusion previews.
///
/// The tuning workflow has two phases:
///
/// **Phase 1 — collect.** Run generation with `--tune-preview <run_dir>`.
/// Each run records `latent_step_N.bin` (raw float32 tensor) and `decoded.png`
/// (VAE-decoded ground truth) into the given directory. Repeat with different
/// prompts so the fit generalises across content:
///
/// ```
/// # Collect from multiple prompts:
/// diffusion-runner --model <path> --prompt "a cat" --tune-preview /tmp/tune/run_0
/// diffusion-runner --model <path> --prompt "a landscape" --tune-preview /tmp/tune/run_1
/// ```
///
/// **Phase 2 — fit.** Pass the parent directory to `--tune-fit`. It reads all
/// collected pairs jointly, fits a single [C, 3] projection + bias via
/// least-squares, exports per-step draft previews, and prints the resulting
/// coefficients:
///
/// ```
/// # Fit jointly over all collected runs:
/// diffusion-runner --model <path> --tune-fit /tmp/tune
/// ```
///
/// Programmatic usage:
///
/// ```swift
/// let tuner = LatentPreviewTuner(outputDir: URL(...))
/// pipeline.generateImages(configuration: config) { progress in
///     if let latent = progress.currentLatent {
///         tuner.record(latent: latent, step: progress.step)
///     }
///     return true
/// }
/// tuner.recordDecoded(image: result.images[0])
/// let coefficients = tuner.fitCoefficients()
/// try tuner.exportPairs()
/// ```
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
        var copy = latent
        var data = [Float](repeating: 0, count: latent.shape.reduce(1, *))
        copy.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<data.count { data[i] = ptr[i] }
        }
        latents.append((step: step, data: data, shape: latent.shape))
        if latents.count == 1 {
            let range = data.isEmpty ? "empty" : "[\(data.min()!), \(data.max()!)]"
            print("[tune] First latent recorded: shape=\(latent.shape), floats=\(data.count), range=\(range)")
        }
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
            let header = shape.map(String.init).joined(separator: ",") + "\n"
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
        else {
            print("[tune] fit failed: latents=\(latents.count), hasImage=\(decodedImage != nil)")
            return nil
        }

        let shape = lastLatent.shape
        print("[tune] lastLatent shape=\(shape), data.count=\(lastLatent.data.count)")
        guard shape.count == 4, shape[0] == 1 else {
            print("[tune] fit failed: shape rank \(shape.count) != 4")
            return nil
        }
        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]
        let spatialCount = height * width
        print("[tune] C=\(channels) H=\(height) W=\(width) N=\(spatialCount)")

        guard let rgbData = Self.extractRGB(from: image, width: width, height: height) else {
            print(
                "[tune] fit failed: extractRGB returned nil (image \(image.width)x\(image.height) → \(width)x\(height))"
            )
            return nil
        }
        print("[tune] rgbData.count=\(rgbData.count) (expected \(width * height * 3))")

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

        // Add ridge for numerical stability
        for i in 0..<channels {
            xtx[i * channels + i] += 1.0
        }
        let diag = (0..<channels).map { xtx[$0 * channels + $0] }
        let xRange = x.isEmpty ? (Float(0), Float(0)) : (x.min()!, x.max()!)
        print("[tune] latent range: [\(xRange.0), \(xRange.1)]")
        print("[tune] XtX diagonal (after ridge): \(diag)")

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
        var lda = n
        var ldb = n
        var nrhs = Int32(3)
        var info = Int32(0)
        var uplo = Int8(UInt8(ascii: "U"))
        sposv_(&uplo, &n, &nrhs, &xtx, &lda, &xty, &ldb, &info)

        guard info == 0 else {
            print("[tune] sposv_ failed: info=\(info) (n=\(channels))")
            return nil
        }
        print("[tune] sposv_ succeeded, computing bias...")

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

    /// Generate preview PNGs for each recorded step.
    ///
    /// - draft: cheap linear projection via fitted coefficients
    /// - full: VAE decode via the provided closure (if non-nil)
    public func exportStepPreviews(
        coefficients: LatentRGBCoefficients,
        vaeDecode: ((_ latent: [Float], _ shape: [Int]) async throws -> CGImage)? = nil
    ) async {
        for (step, data, shape) in latents {
            guard shape.count == 4, shape[0] == 1 else { continue }

            // Draft preview
            var latentArray = NDArray(shape: shape, scalarType: .float32)
            latentArray.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
                for i in 0..<data.count { ptr[i] = data[i] }
            }
            if let preview = latentArray.asRGB(.draft, coefficients: coefficients) {
                savePNG(preview, to: outputDir.appendingPathComponent("latent_\(step)_draft.png"))
            }

            // Full (VAE) preview
            if let vaeDecode {
                do {
                    let decoded = try await vaeDecode(data, shape)
                    savePNG(decoded, to: outputDir.appendingPathComponent("latent_\(step)_vae.png"))
                } catch {
                    print("[tune] VAE decode failed at step \(step): \(error)")
                }
            }
        }
        let modes = vaeDecode != nil ? "draft + VAE" : "draft only"
        print("[tune] Exported \(latents.count) step previews (\(modes))")
    }

    private func savePNG(_ image: CGImage, to url: URL) {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            return
        }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
    }

    /// Downsample a CGImage to the target resolution and extract interleaved RGB bytes.
    private static func extractRGB(from image: CGImage, width: Int, height: Int) -> [UInt8]? {
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

    // MARK: - Joint Fit from Directory

    /// Read all collected latent/image pairs from a directory tree and fit coefficients jointly.
    ///
    /// Expected layout (one subdirectory per generation):
    /// ```
    /// dir/
    ///   prompt_0/
    ///     latent_step_0.bin ... latent_step_N.bin
    ///     decoded.png
    ///   prompt_1/
    ///     ...
    /// ```
    ///
    /// Falls back to treating `dir` itself as a single generation if no subdirectories are found.
    /// Takes the highest-numbered latent step from each generation as the "final" latent,
    /// stacks all final-step latent/RGB pairs into one large X matrix, and fits via
    /// the existing cblas_sgemm + sposv_ path.
    public class func fitFromDirectory(_ dir: URL) -> LatentRGBCoefficients? {
        let fm = FileManager.default

        // Discover generation directories
        var generationDirs: [URL] = []
        if let contents = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.isDirectoryKey])
        {
            generationDirs = contents.filter { url in
                (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
            }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        }

        // Fall back: if dir itself has latent files, treat it as a single generation
        if generationDirs.isEmpty {
            if fm.fileExists(atPath: dir.appendingPathComponent("decoded.png").path) {
                generationDirs = [dir]
            } else {
                print("[tune-fit] No generation subdirectories or decoded.png found in \(dir.path)")
                return nil
            }
        }

        // Collect the final-step latent + decoded image from each generation
        var allLatents: [(data: [Float], shape: [Int])] = []
        var allImages: [CGImage] = []

        for genDir in generationDirs {
            guard let files = try? fm.contentsOfDirectory(at: genDir, includingPropertiesForKeys: nil)
            else { continue }

            let latentFiles =
                files
                .filter { $0.lastPathComponent.hasPrefix("latent_step_") && $0.pathExtension == "bin" }
                .sorted {
                    extractStepNumber(from: $0.lastPathComponent) < extractStepNumber(from: $1.lastPathComponent)
                }

            guard let lastFile = latentFiles.last else {
                print("[tune-fit] No latent files in \(genDir.lastPathComponent), skipping")
                continue
            }

            let decodedURL = genDir.appendingPathComponent("decoded.png")
            guard fm.fileExists(atPath: decodedURL.path) else {
                print("[tune-fit] No decoded.png in \(genDir.lastPathComponent), skipping")
                continue
            }

            guard let (data, shape) = readLatentBin(lastFile) else {
                print("[tune-fit] Could not read \(lastFile.lastPathComponent), skipping")
                continue
            }

            guard let source = CGImageSourceCreateWithURL(decodedURL as CFURL, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                print("[tune-fit] Could not read decoded.png in \(genDir.lastPathComponent), skipping")
                continue
            }

            allLatents.append((data: data, shape: shape))
            allImages.append(image)
            let step = extractStepNumber(from: lastFile.lastPathComponent)
            print("[tune-fit] Loaded \(genDir.lastPathComponent): latent \(shape), step \(step)")
        }

        guard !allLatents.isEmpty else {
            print("[tune-fit] No valid latent/image pairs found")
            return nil
        }

        // Validate all latents share the same shape
        let refShape = allLatents[0].shape
        guard refShape.count == 4, refShape[0] == 1 else {
            print("[tune-fit] Expected 4D latent [1,C,H,W], got \(refShape)")
            return nil
        }
        let channels = refShape[1]
        let height = refShape[2]
        let width = refShape[3]
        let spatialCount = height * width

        for (i, latent) in allLatents.enumerated() {
            if latent.shape != refShape {
                print("[tune-fit] Shape mismatch: generation \(i) has \(latent.shape), expected \(refShape)")
                return nil
            }
        }

        let totalN = allLatents.count * spatialCount
        print("[tune-fit] Fitting from \(allLatents.count) generation(s), C=\(channels) H=\(height) W=\(width)")
        print("[tune-fit] Total pixels: \(totalN)")

        // Stack all pairs into X=[totalN, C] and Y=[totalN, 3]
        var x = [Float](repeating: 0, count: totalN * channels)
        var y = [Float](repeating: 0, count: totalN * 3)

        for (genIdx, (latentData, _)) in allLatents.enumerated() {
            let image = allImages[genIdx]
            guard let rgbData = extractRGB(from: image, width: width, height: height) else {
                print("[tune-fit] extractRGB failed for generation \(genIdx)")
                return nil
            }

            let offset = genIdx * spatialCount
            for c in 0..<channels {
                for p in 0..<spatialCount {
                    x[(offset + p) * channels + c] = latentData[c * spatialCount + p]
                }
            }
            for p in 0..<spatialCount {
                y[(offset + p) * 3 + 0] = Float(rgbData[p * 3 + 0]) / 255.0
                y[(offset + p) * 3 + 1] = Float(rgbData[p * 3 + 1]) / 255.0
                y[(offset + p) * 3 + 2] = Float(rgbData[p * 3 + 2]) / 255.0
            }
        }

        // Normal equations: W = (X^T X + I)^{-1} X^T Y

        // X^T X: [C, C]
        var xtx = [Float](repeating: 0, count: channels * channels)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(channels), Int32(channels), Int32(totalN),
            1.0, x, Int32(channels), x, Int32(channels),
            0.0, &xtx, Int32(channels)
        )

        // Ridge regularisation
        for i in 0..<channels {
            xtx[i * channels + i] += 1.0
        }

        // X^T Y: [C, 3]
        var xty = [Float](repeating: 0, count: channels * 3)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            Int32(channels), 3, Int32(totalN),
            1.0, x, Int32(channels), y, 3,
            0.0, &xty, 3
        )

        // Solve via LAPACK sposv_ (symmetric positive definite)
        var n = Int32(channels)
        var lda = n
        var ldb = n
        var nrhs = Int32(3)
        var info = Int32(0)
        var uplo = Int8(UInt8(ascii: "U"))
        sposv_(&uplo, &n, &nrhs, &xtx, &lda, &xty, &ldb, &info)

        guard info == 0 else {
            print("[tune-fit] sposv_ failed: info=\(info)")
            return nil
        }

        // Compute bias as mean residual
        var predicted = [Float](repeating: 0, count: totalN * 3)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(totalN), 3, Int32(channels),
            1.0, x, Int32(channels), xty, 3,
            0.0, &predicted, 3
        )

        var bias: [Float] = [0, 0, 0]
        for c in 0..<3 {
            var sum: Float = 0
            for p in 0..<totalN {
                sum += y[p * 3 + c] - predicted[p * 3 + c]
            }
            bias[c] = sum / Float(totalN)
        }

        print("[tune-fit] Fit succeeded")
        return LatentRGBCoefficients(channels: channels, weights: xty, bias: bias)
    }

    // MARK: - Binary File Helpers

    /// Extract the step number from a filename like "latent_step_5.bin".
    private class func extractStepNumber(from filename: String) -> Int {
        let name = (filename as NSString).deletingPathExtension
        guard let range = name.range(of: "latent_step_") else { return 0 }
        return Int(name[range.upperBound...]) ?? 0
    }

    /// Read a `.bin` file written by `exportPairs()`.
    ///
    /// Format: ASCII shape header `"1,4,64,64\n"` followed by raw Float32 data.
    private class func readLatentBin(_ url: URL) -> (data: [Float], shape: [Int])? {
        guard let fileData = try? Data(contentsOf: url) else { return nil }

        // Find the newline delimiter separating the shape header from float data.
        guard let newlineIndex = fileData.firstIndex(of: 0x0A) else {
            print("[tune-fit] No header delimiter in \(url.lastPathComponent)")
            return nil
        }

        guard let header = String(data: fileData[fileData.startIndex..<newlineIndex], encoding: .ascii)
        else { return nil }
        let shape = header.split(separator: ",").compactMap { Int($0) }
        guard !shape.isEmpty else { return nil }

        let dataStart = newlineIndex + 1
        let elementCount = shape.reduce(1, *)
        let expectedBytes = elementCount * MemoryLayout<Float>.size
        guard fileData.count >= dataStart + expectedBytes else {
            print(
                "[tune-fit] File too small: \(url.lastPathComponent) has \(fileData.count) bytes, need \(dataStart + expectedBytes)"
            )
            return nil
        }

        var floats = [Float](repeating: 0, count: elementCount)
        fileData.withUnsafeBytes { ptr in
            let src = (ptr.baseAddress! + dataStart).assumingMemoryBound(to: Float.self)
            for i in 0..<elementCount { floats[i] = src[i] }
        }

        return (data: floats, shape: shape)
    }
}
