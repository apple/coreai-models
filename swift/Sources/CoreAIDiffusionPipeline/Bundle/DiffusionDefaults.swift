// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Fallback values for `DiffusionConfig` fields, applied when a bundle's
/// `metadata.json` `diffusion` block omits them.
///
/// The exporter normally writes every field a pipeline reads, so these only
/// cover older or hand-authored bundles. Grouped by pipeline family so the
/// call sites read as `DiffusionDefaults.Video.textDim` rather than a bare
/// literal.
public enum DiffusionDefaults {
    /// FLUX.2 / shared image-diffusion fallbacks.
    public enum Image {
        public static let imageSize = 1024
        public static let batchNormEps: Float = 1e-5
        public static let ropeAxesDims = [32, 32, 32, 32]
        /// VAE encode/decode scale factor.
        public static let scaleFactor: Float = 0.18215
        public static let decoderShiftFactor: Float = 0.0
    }

    /// Sana Sprint fallbacks.
    public enum Sana {
        /// TrigFlow max angle (π/2) for the SCM schedule.
        public static let maxTimesteps: Float = 1.5708
        public static let decoderScaleFactor: Float = 1.0
    }

    /// Wan video-diffusion fallbacks.
    public enum Video {
        public static let textDim = 4096
        public static let latentChannels = 16
        public static let steps = 50
        public static let guidanceScale: Float = 5.0
        public static let schedulerShift: Float = 3.0
        public static let frameCount = 81
    }

    /// diffusion-runner CLI fallbacks when neither the caller nor the bundle
    /// specify a value.
    public enum Runner {
        public static let steps = 20
        public static let guidanceScale: Float = 7.5
    }
}
