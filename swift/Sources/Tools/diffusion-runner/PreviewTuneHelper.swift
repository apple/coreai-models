// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIDiffusionPipeline
import CoreAIShared
import CoreGraphics
import Foundation

/// Collects latents during generation and exports pairs for offline fitting.
///
/// Used by `--tune-preview <dir>` (collect mode). The companion `--tune-fit <dir>`
/// mode calls `runFit(dir:)` to read collected pairs and fit coefficients jointly.
struct PreviewTuneHelper {
    private let tuner: LatentPreviewTuner

    init(outputDir: URL) {
        self.tuner = LatentPreviewTuner(outputDir: outputDir)
    }

    /// Progress handler that records latents and prints step info.
    func progressHandler(_ progress: PipelineProgress) -> Bool {
        print("  Step \(progress.step)/\(progress.totalSteps)")
        if let latent = progress.currentLatent {
            tuner.record(latent: latent, step: progress.step)
        }
        return true
    }

    /// Call after generation with the decoded image. Exports latent/image pairs only.
    func finish(image: CGImage) {
        tuner.recordDecoded(image: image)

        do {
            try tuner.exportPairs()
            print("\nLatent/image pairs exported to tuner output directory.")
        } catch {
            print("Warning: could not export pairs: \(error)")
        }
    }

    // MARK: - Fit Mode (--tune-fit)

    /// Read all collected latent/image pairs from `dir` and fit coefficients jointly.
    static func runFit(dir: URL) {
        guard let coefficients = LatentPreviewTuner.fitFromDirectory(dir) else {
            print("Could not fit coefficients from directory \(dir.path)")
            return
        }

        printTable(coefficients)
        printSwiftCode(coefficients)
    }

    // MARK: - Output Formatting

    static func printTable(_ c: LatentRGBCoefficients) {
        let sep = "  ─────┼─────────────┼─────────────┼─────────────"
        print("")
        print("  Fitted LatentRGBCoefficients (\(c.channels) channels)")
        print(sep)
        print("    Ch │           R │           G │           B")
        print(sep)

        for ch in 0..<c.channels {
            let r = c.weights[ch * 3 + 0]
            let g = c.weights[ch * 3 + 1]
            let b = c.weights[ch * 3 + 2]
            print(String(format: "  %4d │ %+11.5f │ %+11.5f │ %+11.5f", ch, r, g, b))
        }

        print(sep)
        print(String(format: "  bias │ %+11.5f │ %+11.5f │ %+11.5f", c.bias[0], c.bias[1], c.bias[2]))
        print("")
    }

    static func printSwiftCode(_ c: LatentRGBCoefficients) {
        print("// Copy into LatentRGBCoefficients extension:")
        print("public static let tuned = LatentRGBCoefficients(")
        print("    channels: \(c.channels),")
        print("    weights: [")
        for ch in stride(from: 0, to: c.weights.count, by: 3) {
            let r = c.weights[ch]
            let g = c.weights[ch + 1]
            let b = c.weights[ch + 2]
            let trailing = (ch + 3 < c.weights.count) ? "," : ""
            print(String(format: "        %+.6f, %+.6f, %+.6f%@", r, g, b, trailing))
        }
        print("    ],")
        print(
            String(
                format: "    bias: [%.6f, %.6f, %.6f]",
                c.bias[0], c.bias[1], c.bias[2]))
        print(")")
    }
}
