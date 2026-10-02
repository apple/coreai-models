// Precomputed dual-RoPE cos/sin rows for static-shape models.
//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Supplies `rope_cos` / `rope_sin` for models that take precomputed rotary rows
/// instead of `position_ids`.
///
/// Two reasons a graph asks for this rather than gathering RoPE in-graph from a
/// position input:
///
/// - A large context overflows the position input. At 131k a 16-bit position
///   wraps, and a 32-bit one fails to compile.
/// - Interleaved attention types can use different rotary parameters per layer.
///   Passing one combined table keeps the graph free of per-layer branching.
///
/// The combined table is the concatenation `sliding ‖ global`, so `width =
/// slidingHeadDim + globalHeadDim`. Per-dim frequencies are position-independent
/// and precomputed once at init; each step only does `width` sincos per token.
struct DualRoPEInputHandler: StaticInputHandler {
    static let cosInputName = "rope_cos"
    static let sinInputName = "rope_sin"

    let inputNames: [String] = [cosInputName, sinInputName]

    private let cosDescriptors: BucketedInputDescriptors
    private let sinDescriptors: BucketedInputDescriptors

    /// Per-dim angular frequency, indexed by position in the combined table.
    private let theta: [Double]

    init(
        rope: RoPEConfig,
        cosDescriptors: BucketedInputDescriptors,
        sinDescriptors: BucketedInputDescriptors
    ) throws {
        for headDim in [rope.slidingHeadDim, rope.globalHeadDim] where headDim <= 0 || headDim % 2 != 0 {
            throw InferenceRuntimeError.invalidState(
                "RoPE head dims must be positive and even, got \(rope.slidingHeadDim) / "
                    + "\(rope.globalHeadDim)")
        }
        let theta = Self.buildTheta(rope)
        // The combined table is `sliding ‖ global`, so the graph's row width must
        // equal the theta count exactly. A wider row would keep stale pooled-buffer
        // contents in its tail, which reads as a plausible-but-wrong RoPE row.
        for (name, descriptors) in [(Self.cosInputName, cosDescriptors), (Self.sinInputName, sinDescriptors)] {
            for descriptor in descriptors.descriptors where descriptor.shape.last != theta.count {
                throw InferenceRuntimeError.invalidState(
                    "\(name) row width \(descriptor.shape.last ?? -1) != combined RoPE table "
                        + "width \(theta.count)")
            }
        }
        self.theta = theta
        self.cosDescriptors = cosDescriptors
        self.sinDescriptors = sinDescriptors
    }

    /// Builds the combined per-dim theta vector (`sliding ‖ global`). For
    /// position `p`, `rope_cos[d] = cos(p · theta[d])` and likewise for sin.
    ///
    /// - Sliding sub-range `[0, slidingHeadDim)`: standard full-rotary RoPE — the
    ///   `slidingHeadDim / 2` inverse frequencies repeated twice (GPT-NeoX layout).
    /// - Global sub-range: partial rotary — only the first
    ///   `floor(partialRotaryFactor · globalHeadDim / 2)` of the
    ///   `globalHeadDim / 2` frequencies rotate; the rest are 0 (NoPE), repeated twice.
    static func buildTheta(_ rope: RoPEConfig) -> [Double] {
        let slidingHeadDim = rope.slidingHeadDim
        let globalHeadDim = rope.globalHeadDim
        var theta = [Double](repeating: 0, count: slidingHeadDim + globalHeadDim)

        let slidingHalf = slidingHeadDim / 2
        for j in 0..<slidingHeadDim {
            let k = j % slidingHalf
            theta[j] = pow(rope.slidingRopeTheta, -(Double(2 * k) / Double(slidingHeadDim)))
        }

        let globalHalf = globalHeadDim / 2
        let rotaryAngles = Int((rope.partialRotaryFactor * Double(globalHeadDim)) / 2.0)
        for j in 0..<globalHeadDim {
            let m = j % globalHalf
            theta[slidingHeadDim + j] =
                m < rotaryAngles
                ? pow(rope.globalRopeTheta, -(Double(2 * m) / Double(globalHeadDim))) : 0
        }
        return theta
    }

    func registerBuffers(into buffers: inout InputBuffers) {
        cosDescriptors.registerBuffers(name: Self.cosInputName, into: &buffers)
        sinDescriptors.registerBuffers(name: Self.sinInputName, into: &buffers)
    }

    func fill(_ context: InputContext, into buffers: inout InputBuffers) throws {
        let key = StaticBucketKey(batchSize: context.batchSize, contextBucket: context.contextBucket)
        let span = InstrumentsProfiler.beginRopeBuild()
        do {
            try fillTable(
                name: Self.cosInputName, descriptors: cosDescriptors, key: key,
                context: context, into: &buffers, transform: cos)
            try fillTable(
                name: Self.sinInputName, descriptors: sinDescriptors, key: key,
                context: context, into: &buffers, transform: sin)
        } catch {
            span.end()
            throw error
        }
        span.end()
    }

    private func fillTable(
        name: String,
        descriptors: BucketedInputDescriptors,
        key: StaticBucketKey,
        context: InputContext,
        into buffers: inout InputBuffers,
        transform: (Double) -> Double
    ) throws {
        let descriptor = try descriptors.require(key, input: name)
        buffers.ensureCapacity(name: name, descriptor: descriptor)
        let batchSize = context.batchSize
        let alignedStep = context.alignedStep
        let theta = self.theta
        try buffers.withMutableBuffer(name) { array in
            array.mutableView(as: Float16.self)
                .withUnsafeMutablePointer { ptr, shape, strides in
                    // shape: (1, q_len, width)
                    for i in 0..<batchSize {
                        let position = Double(alignedStep + i)
                        let rowBase = i &* strides[1]
                        for d in 0..<theta.count {
                            let value = transform(position * theta[d])
                            ptr[rowBase &+ d &* strides[2]] = Float16(value)
                        }
                    }
                }
        }
    }
}
