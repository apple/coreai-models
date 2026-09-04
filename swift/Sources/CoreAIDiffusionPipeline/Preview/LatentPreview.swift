// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics

// MARK: - Preview Quality

public enum PreviewQuality {
    /// Cheap linear channel→RGB projection via vDSP. ~0.1ms for 128×128.
    case draft
    /// Full VAE decode. Pixel-accurate.
    case full
}

// MARK: - Latent-to-RGB Coefficients

/// Per-model linear projection from latent channels to RGB.
///
/// Fitted by encoding images through the VAE and regressing latent channels
/// against original RGB values. Stored as a flat [C×3] weight matrix for vDSP.
public struct LatentRGBCoefficients: Sendable {
    /// Flat row-major [C, 3] — one row per latent channel, columns R/G/B.
    public let weights: [Float]
    /// Per-channel bias [3].
    public let bias: [Float]
    public let channels: Int

    public init(channels: Int, weights: [Float], bias: [Float]) {
        precondition(weights.count == channels * 3)
        precondition(bias.count == 3)
        self.channels = channels
        self.weights = weights
        self.bias = bias
    }
}

extension LatentRGBCoefficients {
    /// FLUX.2 VAE (16 latent channels post-unpatchify).
    // TODO: fit against actual VAE — placeholder values
    public static let flux2 = LatentRGBCoefficients(
        channels: 16,
        weights: [
            0.298, 0.207, 0.208, 0.187, 0.286, 0.173,
            0.158, 0.189, 0.264, -0.134, -0.141, -0.168,
            0.117, 0.068, 0.052, -0.064, 0.117, -0.029,
            0.049, -0.049, 0.118, -0.082, 0.061, -0.071,
            0.093, 0.027, -0.043, -0.028, 0.093, 0.038,
            0.041, -0.037, 0.082, -0.067, 0.043, -0.058,
            0.078, 0.022, -0.035, -0.022, 0.075, 0.031,
            0.033, -0.030, 0.067, -0.055, 0.035, -0.047,
        ],
        bias: [0.5, 0.5, 0.5]
    )

    /// Stable Diffusion 1.x / 2.x VAE (4 latent channels).
    public static let sd1 = LatentRGBCoefficients(
        channels: 4,
        weights: [
            0.298, 0.207, 0.208,
            0.187, 0.286, 0.173,
            0.158, 0.189, 0.264,
            -0.134, -0.141, -0.168,
        ],
        bias: [0.5, 0.5, 0.5]
    )
}

// MARK: - NDArray → CGImage

extension NDArray {
    /// Convert a latent tensor [1, C, H, W] to an RGB preview.
    ///
    /// - `.draft`: vDSP matrix multiply (latent channels → RGB), then
    ///   `DiffusionUtilities.pixelsToCGImage` for the CHW→CGImage step.
    /// - `.full`: returns nil — use the pipeline's VAE decoder directly.
    public func asRGB(
        _ quality: PreviewQuality,
        coefficients: LatentRGBCoefficients? = nil
    ) -> CGImage? {
        switch quality {
        case .draft:
            guard let coeffs = coefficients else { return nil }
            return draftProjection(coefficients: coeffs)
        case .full:
            return nil
        }
    }

    /// Project C-channel latent to 3-channel RGB using vDSP matrix multiply.
    ///
    /// Input:  self = [1, C, H, W] in BCHW layout
    /// Output: [Float] of length 3*H*W in CHW layout (R plane, G plane, B plane)
    ///
    /// The math: for each spatial position p,
    ///   rgb[c, p] = sum_k(latent[k, p] * weights[k, c]) + bias[c]
    ///
    /// Implemented as a single GEMM: [N, C] × [C, 3] → [N, 3]
    /// where N = H*W, then transpose to CHW and add bias.
    private func draftProjection(coefficients: LatentRGBCoefficients) -> CGImage? {
        guard shape.count == 4, shape[0] == 1 else { return nil }
        let channels = shape[1]
        let height = shape[2]
        let width = shape[3]
        guard channels == coefficients.channels else { return nil }

        let spatialCount = height * width

        // Transpose latent from CHW to NxC (HW-major, channel-minor)
        let view = self.view(as: Float.self)
        var nxc = [Float](repeating: 0, count: spatialCount * channels)
        view.withUnsafePointer { ptr, _, _ in
            for c in 0..<channels {
                for p in 0..<spatialCount {
                    nxc[p * channels + c] = ptr[c * spatialCount + p]
                }
            }
        }

        // GEMM: [N, C] × [C, 3] → [N, 3]
        var nx3 = [Float](repeating: 0, count: spatialCount * 3)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(spatialCount), 3, Int32(channels),
            1.0,
            nxc, Int32(channels),
            coefficients.weights, 3,
            0.0,
            &nx3, 3
        )

        // Transpose N×3 (interleaved RGB) → CHW (3 planes) and add bias.
        // pixelsToCGImage expects [-1, 1] and applies x*0.5+0.5, so remap.
        var chw = [Float](repeating: 0, count: 3 * spatialCount)
        for c in 0..<3 {
            let bias = coefficients.bias[c]
            let planeOffset = c * spatialCount
            for p in 0..<spatialCount {
                let val = nx3[p * 3 + c] + bias
                chw[planeOffset + p] = val * 2.0 - 1.0
            }
        }

        return try? DiffusionUtilities.pixelsToCGImage(chw, height: height, width: width)
    }
}
