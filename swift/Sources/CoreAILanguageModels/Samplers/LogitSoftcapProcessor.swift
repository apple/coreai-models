// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// Final-logit soft capping, applied on the CPU after the forward pass.
///
/// Gemma-family models squash their output logits with `c · tanh(logits / c)`. The
/// `tanh` is best run on the CPU rather than in the graph, so the iOS export leaves it out of the
/// graph (see `models/ios/gemma4_text.py`) and the runner applies it here instead,
/// between reading `out_logits` and sampling.
///
/// Applying it runner-side keeps sampling *and* any `--save-logits` / `--print-logits`
/// output on the same capped values the reference implementation produces, so parity
/// comparisons stay meaningful.
public struct LogitSoftcapProcessor {
    /// Applies `cap · tanh(logits / cap)` to `logits` in place.
    ///
    /// The arithmetic runs in `Float` even when ``LogitsScalarType`` is `Float16`:
    /// `tanh` of a half-precision quotient loses too much of the small-difference
    /// structure that sampling depends on. Results are rounded back to
    /// ``LogitsScalarType`` on the way out — which is where any remaining divergence
    /// from the reference fp32 implementation comes from.
    ///
    /// - Parameters:
    ///   - logits: Mutable logits array (vocab-sized). Modified in-place.
    ///   - cap: The soft cap `c`. Non-positive values are a no-op.
    public static func apply(to logits: inout [LogitsScalarType], cap: Float) {
        guard cap > 0, !logits.isEmpty else { return }

        // vForce's tanh wants Float32, so widen (and pre-divide), transform in place,
        // then narrow back. The buffer is ~1 MB at Gemma's vocab size — negligible next
        // to the forward pass that produced these logits.
        let inverseCap = 1 / cap
        var scaled = logits.map { Float($0) * inverseCap }
        var elementCount = Int32(scaled.count)
        scaled.withUnsafeMutableBufferPointer { buffer in
            vvtanhf(buffer.baseAddress!, buffer.baseAddress!, &elementCount)
        }
        for i in logits.indices {
            logits[i] = LogitsScalarType(scaled[i] * cap)
        }
    }
}
