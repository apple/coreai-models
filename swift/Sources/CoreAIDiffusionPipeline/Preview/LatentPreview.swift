// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics

// MARK: - Latent-to-RGB Coefficients

/// Per-model linear projection from latent channels to RGB.
///
/// Fitted by encoding images through the VAE and regressing latent channels
/// against original RGB values. Stored as a flat [C×3] weight matrix for BLAS (cblas_sgemm).
///
/// Requires latent coefficients fit for the specific model's VAE, since the
/// latent-to-RGB mapping differs per model. Fit a set for the model you run and
/// pass it to ``NDArray/asRGB(coefficients:)``:
///
/// ```
/// # 1. Collect latent/image pairs across a prompt set (once per prompt):
/// diffusion-runner --model <exported-model> --prompt "…" --tune-preview <dir>/prompt_N
/// # 2. Fit a single [C, 3] projection jointly and print copy-paste Swift:
/// diffusion-runner --tune-fit <dir>
/// ```
public struct LatentRGBCoefficients: Sendable {
    /// Fixed RGB output channel count for the latent-to-RGB projection.
    public static let rgbChannels = 3
    /// Flat row-major [C, 3] — one row per latent channel, columns R/G/B.
    public let weights: [Float]
    /// Per-channel bias [3].
    public let bias: [Float]
    public let channels: Int

    public init(channels: Int, weights: [Float], bias: [Float]) {
        precondition(weights.count == channels * Self.rgbChannels)
        precondition(bias.count == Self.rgbChannels)
        self.channels = channels
        self.weights = weights
        self.bias = bias
    }
}

// MARK: - NDArray → CGImage

extension NDArray {
    /// Project a latent tensor [1, C, H, W] to an RGB preview via a cblas_sgemm
    /// (BLAS) matrix multiply (latent channels → RGB), then `DiffusionUtilities.pixelsToCGImage`
    /// for the CHW→CGImage step.
    ///
    /// `coefficients` must match the model's VAE; fit a set via
    /// `diffusion-runner --tune-preview` / `--tune-fit`.
    public func asRGB(
        coefficients: LatentRGBCoefficients
    ) -> CGImage? {
        return draftProjection(coefficients: coefficients)
    }

    /// Project C-channel latent to 3-channel RGB using a cblas_sgemm (BLAS) matrix multiply.
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
        let rgb = LatentRGBCoefficients.rgbChannels

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
        var nx3 = [Float](repeating: 0, count: spatialCount * rgb)
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
        var chw = [Float](repeating: 0, count: rgb * spatialCount)
        for c in 0..<rgb {
            let bias = coefficients.bias[c]
            let planeOffset = c * spatialCount
            for p in 0..<spatialCount {
                let val = nx3[p * rgb + c] + bias
                chw[planeOffset + p] = val * 2.0 - 1.0
            }
        }

        return try? DiffusionUtilities.pixelsToCGImage(chw, height: height, width: width)
    }
}
