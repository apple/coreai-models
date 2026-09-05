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
    /// Faster linear decode using vDSP. Expected to run under 0.1ms per step.
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
    /// FLUX.2 VAE (32 latent channels post-unpatchify).
    /// Fitted over 20 diverse prompts via `--tune-fit`.
    public static let flux2 = LatentRGBCoefficients(
        channels: 32,
        weights: [
            -0.001234, -0.150142, +0.062948,
            +0.057283, +0.091131, +0.185561,
            -0.102564, -0.095619, +0.041728,
            +0.173053, +0.105305, +0.109426,
            -0.058650, -0.034254, +0.017639,
            +0.151894, +0.075108, -0.000086,
            -0.124391, -0.109024, -0.085880,
            -0.118337, -0.233165, -0.061803,
            -0.289475, -0.120156, -0.189252,
            -0.012170, -0.021968, -0.042516,
            +0.163237, +0.172269, +0.111639,
            -0.029559, -0.026128, -0.192946,
            -0.001641, -0.013507, +0.060971,
            +0.094575, +0.030036, -0.109289,
            +0.042975, -0.031263, -0.148144,
            -0.317512, -0.150491, -0.141777,
            +0.125586, +0.259677, +0.298747,
            +0.038469, -0.078418, -0.121917,
            +0.079620, +0.013650, -0.003717,
            -0.043233, -0.133767, -0.252056,
            -0.046421, +0.004856, -0.023006,
            -0.098169, -0.064560, -0.085343,
            -0.033093, -0.147993, -0.075416,
            +0.058341, +0.008623, -0.019948,
            -0.100011, -0.112523, -0.036843,
            -0.065810, -0.081621, -0.062039,
            +0.086651, +0.151725, +0.077021,
            -0.010121, -0.007171, -0.022947,
            -0.110737, -0.093478, +0.030356,
            -0.085815, -0.099332, -0.038691,
            -0.192690, -0.284099, -0.176111,
            -0.053914, -0.056429, -0.030389,
        ],
        bias: [-0.444866, -0.702195, -0.420561]
    )

    /// Stable Diffusion 3.5 VAE (16 latent channels, post-scale/shift).
    /// Fitted over 20 diverse prompts via `--tune-fit`.
    public static let sd3 = LatentRGBCoefficients(
        channels: 16,
        weights: [
            +2.627717, +8.884879, +8.919661,
            -9.405011, +2.030191, -8.060559,
            +2.635179, +1.377024, -4.091317,
            +1.928998, +5.690139, +1.821600,
            +15.098738, +7.807263, +7.071290,
            -2.805313, -3.986329, +5.017890,
            +1.851985, +3.534414, -1.023579,
            -15.243811, -8.246144, +2.143155,
            -4.823180, -6.584886, +7.060158,
            +12.014882, +8.901643, +6.437391,
            +6.352098, -2.148832, +10.259954,
            +8.025915, +2.099451, -0.958590,
            +4.778718, +8.851213, +2.280541,
            -3.817901, -5.800869, -3.999441,
            -5.648732, -6.321621, -3.382349,
            +0.777097, +0.846617, +4.236656,
        ],
        bias: [-46.833546, -29.227682, -20.308897]
    )

    /// Stable Diffusion 1.x / 2.x VAE (4 latent channels, post-scale).
    /// Single-image fit from SD 2.1. Re-tune with `--tune-fit` for better results.
    public static let sd1 = LatentRGBCoefficients(
        channels: 4,
        weights: [
            -0.007860, +0.426769, +0.586632,
            +0.314895, +0.114553, +0.191145,
            +0.025540, +0.054536, +0.269923,
            -0.529141, -0.665231, -0.442618,
        ],
        bias: [0.396795, 0.323238, 0.341631]
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
