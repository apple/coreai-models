// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAIShared
import Foundation

// The Gemma 4 MTP/EAGLE draft head run entirely on the CPU, from the pack
// `models/gemma4/export_draft_pack.py` writes. It computes what `StaticShapeDrafter` runs as a
// static-shape program, and drafts the same tokens up to rounding, so the target's accelerator is
// left to the target alone.

/// The draft head's arithmetic over a ``DraftPack`` loaded with ``DraftPack/Role/head``: each
/// projection is a BNNSGraph, the attention, norms and rotary tables are Accelerate (BLAS and vDSP)
/// over fp32 activations.
struct CPUDraftHead {
    /// One projection `y = W · x` as a BNNSGraph, built once at load. BNNSGraph takes fp32
    /// weights only, so the pack's fp16 weights are widened here.
    final class Projection {
        let rows: Int
        let columns: Int
        private let context: BNNSGraph.Context
        /// Argument positions of the output and the input.
        private let outputPosition: Int
        private let inputPosition: Int

        init(_ name: String, in pack: DraftPack) throws {
            let shape = pack.shape(name)
            precondition(shape.count == 2, "projection '\(name)' is not a matrix")
            let (m, n) = (shape[0], shape[1])
            let w = pack.f16(name)
            let weights = (0..<(m * n)).map { Float(w[$0]) }
            context = try BNNSGraph.makeContext { builder in
                let x = builder.argument(name: "x", dataType: Float.self, shape: [1, n])
                return [x.matmul(other: builder.constant(values: weights, shape: [m, n]), transposeOther: true)]
            }
            inputPosition = context.argumentPosition(argument: "x")
            outputPosition = 1 - inputPosition
            rows = m
            columns = n
        }

        /// `y = W · x`; `x` holds `columns` values and `y` receives `rows`.
        func apply(_ x: [Float], into y: inout [Float]) {
            precondition(x.count >= columns && y.count >= rows, "projection reads \(columns) values into \(rows)")
            var input = x
            input.withUnsafeMutableBufferPointer { xp in
                y.withUnsafeMutableBufferPointer { yp in
                    var arguments = [xp, xp]
                    arguments[outputPosition] = yp
                    arguments[inputPosition] = xp
                    // The graph's shapes are fixed at build and its buffers are the caller's, so
                    // execution cannot fail for a reason the caller could act on.
                    try! context.executeFunction(arguments: arguments)
                }
            }
        }
    }

    /// One layer's weights, looked up once rather than by name on every step.
    struct Layer {
        let inputNorm, qNorm, postAttentionNorm, preFeedforwardNorm, postFeedforwardNorm: UnsafePointer<Float16>
        let q, o, gate, up, down: Projection
        let scalar: Float
    }

    /// The target's shared K/V for one draft block, position-major: row `s` of a cache holds its
    /// `channels` values at `[s * channels]`, oldest position first. Valid only while the drafter
    /// that built it is drafting.
    struct SharedKV {
        let slidingK, slidingV: UnsafePointer<Float>
        let slidingRows, slidingChannels: Int
        let fullK, fullV: UnsafePointer<Float>
        let fullRows, fullChannels: Int
    }

    let hidden, backboneHidden, heads, headDim, fullHeadDim, ffn: Int
    let eps: Float = 1e-6
    let layers: [Layer]
    let preProjection, postProjection, centroids: Projection
    let finalNorm: UnsafePointer<Float16>
    let invFreqSliding, invFreqFull: [Float]
    let centroidCount: Int

    init(pack: DraftPack, backboneHidden: Int) throws {
        let index = pack.index
        // `DraftPack.require(.head, ...)` has checked that every value is present and positive.
        hidden = index.hidden!
        heads = index.heads!
        headDim = index.headDim!
        fullHeadDim = index.fullHeadDim!
        ffn = index.ffn!
        centroidCount = index.centroids
        self.backboneHidden = backboneHidden
        layers = try (0..<index.layers).map { layer in
            let p = "model.layers.\(layer)."
            return Layer(
                inputNorm: pack.f16(p + "input_layernorm.weight"),
                qNorm: pack.f16(p + "self_attn.q_norm.weight"),
                postAttentionNorm: pack.f16(p + "post_attention_layernorm.weight"),
                preFeedforwardNorm: pack.f16(p + "pre_feedforward_layernorm.weight"),
                postFeedforwardNorm: pack.f16(p + "post_feedforward_layernorm.weight"),
                q: try Projection(p + "self_attn.q_proj.weight", in: pack),
                o: try Projection(p + "self_attn.o_proj.weight", in: pack),
                gate: try Projection(p + "mlp.gate_proj.weight", in: pack),
                up: try Projection(p + "mlp.up_proj.weight", in: pack),
                down: try Projection(p + "mlp.down_proj.weight", in: pack),
                scalar: Float(pack.f16(p + "layer_scalar")[0]))
        }
        preProjection = try Projection("pre_projection.weight", in: pack)
        postProjection = try Projection("post_projection.weight", in: pack)
        centroids = try Projection("centroids_f16", in: pack)
        finalNorm = pack.f16("model.norm.weight")
        invFreqSliding = DraftScan.inverseFrequencies(headDim: headDim, theta: index.slidingTheta!, partial: 1)
        invFreqFull = DraftScan.inverseFrequencies(
            headDim: fullHeadDim, theta: index.fullTheta!, partial: index.fullPartialRotaryFactor!)
    }

    /// One draft step. `embeds` is `cat(embed(token), backbone_hidden)`, `2 * backboneHidden` wide.
    ///
    /// - Returns: the head's state after the final norm (what the candidate scan scores), its
    ///   centroid logits, and the backbone-wide state that seeds the next step.
    func step(embeds: [Float], kv: SharedKV, position: Int) -> (x: [Float], centroidLogits: [Float], hidden: [Float]) {
        var x = [Float](repeating: 0, count: hidden)
        Self.mv(preProjection, embeds, m: hidden, n: 2 * backboneHidden, into: &x)
        let ropeSliding = Self.rope(position: position, invFreq: invFreqSliding)
        let ropeFull = Self.rope(position: position, invFreq: invFreqFull)
        for (layerIndex, layer) in layers.enumerated() {
            let full = layerIndex == layers.count - 1
            let hd = full ? fullHeadDim : headDim
            let qWidth = heads * hd

            var y = x
            Self.rmsnorm(&y, layer.inputNorm, eps: eps)
            var q = [Float](repeating: 0, count: qWidth)
            Self.mv(layer.q, y, m: qWidth, n: hidden, into: &q)
            // Gemma's query norm is per head, before RoPE, and its attention scale is 1.
            q.withUnsafeMutableBufferPointer { qp in
                for h in 0..<heads { Self.rmsnorm(qp.baseAddress! + h * hd, n: hd, layer.qNorm, eps: eps) }
            }
            let (cos, sin) = full ? ropeFull : ropeSliding
            Self.applyRotary(&q, heads: heads, headDim: hd, cos: cos, sin: sin)

            var attention = [Float](repeating: 0, count: qWidth)
            if full {
                Self.attend(
                    q: q, k: kv.fullK, v: kv.fullV, rows: kv.fullRows, channels: kv.fullChannels,
                    headDim: hd, heads: heads, into: &attention)
            } else {
                Self.attend(
                    q: q, k: kv.slidingK, v: kv.slidingV, rows: kv.slidingRows, channels: kv.slidingChannels,
                    headDim: hd, heads: heads, into: &attention)
            }
            var o = [Float](repeating: 0, count: hidden)
            Self.mv(layer.o, attention, m: hidden, n: qWidth, into: &o)
            Self.rmsnorm(&o, layer.postAttentionNorm, eps: eps)
            vDSP_vadd(x, 1, o, 1, &x, 1, vDSP_Length(hidden))

            var z = x
            Self.rmsnorm(&z, layer.preFeedforwardNorm, eps: eps)
            var gate = [Float](repeating: 0, count: ffn)
            var up = [Float](repeating: 0, count: ffn)
            Self.mv(layer.gate, z, m: ffn, n: hidden, into: &gate)
            Self.mv(layer.up, z, m: ffn, n: hidden, into: &up)
            Self.geluTimes(&gate, up)
            var down = [Float](repeating: 0, count: hidden)
            Self.mv(layer.down, gate, m: hidden, n: ffn, into: &down)
            Self.rmsnorm(&down, layer.postFeedforwardNorm, eps: eps)
            vDSP_vadd(x, 1, down, 1, &x, 1, vDSP_Length(hidden))
            var scalar = layer.scalar
            vDSP_vsmul(x, 1, &scalar, &x, 1, vDSP_Length(hidden))
        }
        Self.rmsnorm(&x, finalNorm, eps: eps)
        var next = [Float](repeating: 0, count: backboneHidden)
        Self.mv(postProjection, x, m: backboneHidden, n: hidden, into: &next)
        var centroidLogits = [Float](repeating: 0, count: centroidCount)
        Self.mv(centroids, x, m: centroidCount, n: hidden, into: &centroidLogits)
        return (x, centroidLogits, next)
    }

    // MARK: - Kernels

    /// `y = W · x` for `W` of `m` rows by `n` columns.
    static func mv(_ w: Projection, _ x: [Float], m: Int, n: Int, into y: inout [Float]) {
        precondition(w.rows == m && w.columns == n, "projection is \(w.rows)x\(w.columns), not \(m)x\(n)")
        w.apply(x, into: &y)
    }

    /// RMS-normalizes `n` values at `x` in place and applies the fp16 gain `g`.
    static func rmsnorm(_ x: UnsafeMutablePointer<Float>, n: Int, _ g: UnsafePointer<Float16>, eps: Float) {
        var squares: Float = 0
        vDSP_svesq(x, 1, &squares, vDSP_Length(n))
        let scale = 1 / (squares / Float(n) + eps).squareRoot()
        for i in 0..<n { x[i] *= scale * Float(g[i]) }
    }

    static func rmsnorm(_ x: inout [Float], _ g: UnsafePointer<Float16>, eps: Float) {
        let n = x.count
        x.withUnsafeMutableBufferPointer { rmsnorm($0.baseAddress!, n: n, g, eps: eps) }
    }

    /// `g = gelu(g) * u` with the tanh approximation of GELU.
    static func geluTimes(_ g: inout [Float], _ u: [Float]) {
        let c: Float = 0.7978845608028654  // sqrt(2 / pi)
        var t = g.map { c * ($0 + 0.044715 * $0 * $0 * $0) }
        var count = Int32(t.count)
        t.withUnsafeMutableBufferPointer { vvtanhf($0.baseAddress!, $0.baseAddress!, &count) }
        for i in 0..<g.count { g[i] = 0.5 * g[i] * (1 + t[i]) * u[i] }
    }

    /// The rotate-half tables for `position`: one cosine and one sine per frequency pair.
    static func rope(position: Int, invFreq: [Float]) -> (cos: [Float], sin: [Float]) {
        let angles = invFreq.map { Float(position) * $0 }
        return (angles.map(cosf), angles.map(sinf))
    }

    /// Rotate-half RoPE in place over `heads` heads of width `headDim`.
    static func applyRotary(_ q: inout [Float], heads: Int, headDim: Int, cos: [Float], sin: [Float]) {
        let half = headDim / 2
        for h in 0..<heads {
            let o = h * headDim
            for d in 0..<half {
                let a = q[o + d]
                let b = q[o + d + half]
                q[o + d] = a * cos[d] - b * sin[d]
                q[o + d + half] = b * cos[d] + a * sin[d]
            }
        }
    }

    /// Softmax attention of `heads` query heads over `rows` cached positions, scale 1. A cache row
    /// is `channels` wide and holds `channels / headDim` K/V heads; query heads are split evenly
    /// across them.
    static func attend(
        q: [Float], k: UnsafePointer<Float>, v: UnsafePointer<Float>, rows: Int, channels: Int,
        headDim: Int, heads: Int, into out: inout [Float]
    ) {
        guard rows > 0 else {
            for i in 0..<out.count { out[i] = 0 }
            return
        }
        let groups = max(1, channels / headDim)
        let perGroup = max(1, heads / groups)
        var scores = [Float](repeating: 0, count: perGroup * rows)
        q.withUnsafeBufferPointer { qp in
            out.withUnsafeMutableBufferPointer { op in
                scores.withUnsafeMutableBufferPointer { sp in
                    let s = sp.baseAddress!
                    for g in 0..<groups {
                        let qg = qp.baseAddress! + g * perGroup * headDim
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasTrans,
                            Int32(perGroup), Int32(rows), Int32(headDim), 1,
                            qg, Int32(headDim), k + g * headDim, Int32(channels), 0, s, Int32(rows))
                        for h in 0..<perGroup {
                            let row = s + h * rows
                            var top: Float = 0
                            vDSP_maxv(row, 1, &top, vDSP_Length(rows))
                            var shift = -top
                            vDSP_vsadd(row, 1, &shift, row, 1, vDSP_Length(rows))
                            var n = Int32(rows)
                            vvexpf(row, row, &n)
                            var total: Float = 0
                            vDSP_sve(row, 1, &total, vDSP_Length(rows))
                            var inverse = total > 0 ? 1 / total : 0
                            vDSP_vsmul(row, 1, &inverse, row, 1, vDSP_Length(rows))
                        }
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            Int32(perGroup), Int32(headDim), Int32(rows), 1,
                            s, Int32(rows), v + g * headDim, Int32(channels), 0,
                            op.baseAddress! + g * perGroup * headDim, Int32(headDim))
                    }
                }
            }
        }
    }
}

/// The Gemma 4 MTP/EAGLE draft head on the CPU: proposes from the target's final hidden state and
/// its shared K/V, like ``StaticShapeDrafter``, without a Core AI program.
///
/// The sliding layers attend to the target's sliding cache, the most recent `sliding_window`
/// positions; the full-attention layer attends to the last ``fullAttentionWindow`` positions of the
/// full cache, which keeps a step's cost and the drafter's memory flat as the context grows. Every draft is verified, so the
/// window can cost acceptance at long context, never correctness.
///
/// Single-owner: one task drives it at a time, and it holds no lock; set the provider closures
/// before the first draft.
package final class CPUDrafter: DrafterEngine, @unchecked Sendable {
    package let maxContextLength: Int
    /// The target's token embedding for a token id (`StaticShapeEngine.drafterEmbedding`).
    package var embeddingProvider: ((Int32) async throws -> [Float])?
    /// In-place access to the target's shared caches (`StaticShapeEngine.withSharedKVCache`).
    /// Drafting without it throws.
    package var sharedKVProvider:
        (
            (StaticShapeEngine.SharedKVCacheKind, (StaticShapeEngine.SharedKVCache) -> Void) throws -> Void
        )?

    /// Most recent positions of the full cache the full-attention layer attends to, by default.
    package static let defaultFullAttentionWindow = 1024

    private let pack: DraftPack
    private let head: CPUDraftHead
    private var scanner: DraftScan.Scanner
    private let slidingWindow: Int
    private var seedHidden: [Float]?
    private var embedCache = EmbeddingCache()
    /// Most recent positions of the full cache the full-attention layer attends to.
    let fullAttentionWindow: Int
    /// fp32 copies of the full cache's last ``fullAttentionWindow`` positions, a ring: position `p`
    /// is in row `p % fullAttentionWindow`. Attention does not depend on row order, since each key
    /// already carries its position's rotation. Holds the positions in `keptStart..<keptEnd`, and
    /// `keptTokens` the tokens they were cached for, so a round reads only the positions that are
    /// new or whose token changed.
    private var fullK: [Float] = [], fullV: [Float] = []
    private var keptStart = 0, keptEnd = 0
    private var keptTokens = TokenHistory()
    /// The sliding cache's window for the current round, rebuilt every round: the ring moves.
    private var slidingK: [Float] = [], slidingV: [Float] = []
    private var slidingChannels = 0

    /// - Parameters:
    ///   - pack: `draft_head.safetensors` from `models/gemma4/export_draft_pack.py`.
    ///   - vocabSize: the target's vocabulary; when given, the pack must match it.
    ///   - topK: centroids whose candidates the scan scores; nil for ``DraftScan/defaultTopK``.
    ///   - fullAttentionWindow: most recent positions the full-attention layer attends to.
    /// - Throws: When the pack is malformed or was exported for another target.
    package init(
        pack url: URL, backboneHiddenSize: Int, maxContextLength: Int, vocabSize: Int? = nil, topK: Int? = nil,
        fullAttentionWindow: Int = CPUDrafter.defaultFullAttentionWindow
    ) throws {
        precondition(fullAttentionWindow > 0, "fullAttentionWindow must be positive")
        let pack = try DraftPack(safetensors: url, role: .head)
        let index = pack.index
        let inputWidth = pack.shape("pre_projection.weight").last ?? 0
        let outputWidth = pack.shape("post_projection.weight").first ?? 0
        guard inputWidth == 2 * backboneHiddenSize, outputWidth == backboneHiddenSize else {
            throw InferenceRuntimeError.invalidArgument(
                "\(url.lastPathComponent) is a draft head for a target \(outputWidth) wide, but the target's "
                    + "hidden size is \(backboneHiddenSize); export the pack from the target's own assistant")
        }
        if let vocabSize, index.vocab != vocabSize {
            throw InferenceRuntimeError.invalidArgument(
                "\(url.lastPathComponent) covers a vocabulary of \(index.vocab), but the target's is \(vocabSize)")
        }
        self.pack = pack
        head = try CPUDraftHead(pack: pack, backboneHidden: backboneHiddenSize)
        scanner = DraftScan.Scanner(pack: pack, topK: try DraftScan.topK(requested: topK, centroids: index.centroids))
        slidingWindow = index.slidingWindow!
        self.fullAttentionWindow = fullAttentionWindow
        self.maxContextLength = maxContextLength
    }

    package func reset(to tokenIndex: Int) async throws {
        seedHidden = nil
        // Positions at or past the rollback point will be cached again.
        keptTokens.truncate(to: min(keptTokens.count, max(0, tokenIndex)))
        keptEnd = min(keptEnd, keptTokens.count)
        keptStart = min(keptStart, keptEnd)
    }

    package func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {
        guard let seed = features?.seedRow(confirmedLength: confirmedLength) else { return }
        seedHidden = seed
    }

    package func draftAll(
        confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy,
        using rng: inout some RandomNumberGenerator
    ) async throws -> DraftProposal {
        guard let caches = sharedKVProvider else {
            throw InferenceRuntimeError.invalidState(
                "CPUDrafter has no shared K/V provider: connect it to the target before drafting")
        }
        // Before the first verify there is no hidden state to seed from, so the round proposes the
        // anchor, which the target rejects.
        guard var lastHidden = seedHidden else {
            let anchor = confirmedPrefix.last ?? 0
            return DraftProposal(tokens: Array(repeating: anchor, count: count))
        }
        // The anchor is the last confirmed token; the target has cached the positions before it.
        let kvLen = max(1, confirmedPrefix.count - 1)
        let (slidingRows, slidingChannels, fullChannels) = try readSharedKV(
            caches, cachedTokens: confirmedPrefix[..<kvLen])
        let fullRows = min(kvLen, fullAttentionWindow)

        var lastToken = confirmedPrefix.last ?? 0
        let basePosition = confirmedPrefix.count - 1
        var tokens: [Int32] = []
        tokens.reserveCapacity(count)
        for i in 0..<count {
            var fused = try await embedding(for: lastToken)
            fused.append(contentsOf: lastHidden)
            guard fused.count == 2 * head.backboneHidden else {
                throw InferenceRuntimeError.invalidState(
                    "the target supplied \(fused.count) values for the draft head's \(2 * head.backboneHidden)-wide "
                        + "input; the pack was exported for a different target")
            }
            let (x, centroidLogits, next) = slidingK.withUnsafeBufferPointer { sk in
                slidingV.withUnsafeBufferPointer { sv in
                    fullK.withUnsafeBufferPointer { fk in
                        fullV.withUnsafeBufferPointer { fv in
                            head.step(
                                embeds: fused,
                                kv: CPUDraftHead.SharedKV(
                                    slidingK: sk.baseAddress!, slidingV: sv.baseAddress!,
                                    slidingRows: slidingRows, slidingChannels: slidingChannels,
                                    fullK: fk.baseAddress!, fullV: fv.baseAddress!,
                                    fullRows: fullRows, fullChannels: fullChannels),
                                position: basePosition + i)
                        }
                    }
                }
            }
            let token = scanner.bestToken(x: x, centroidLogits: centroidLogits)
            tokens.append(token)
            lastToken = token
            lastHidden = next
        }
        return DraftProposal(tokens: tokens)
    }

    private func embedding(for token: Int32) async throws -> [Float] {
        if let row = embedCache[token] { return row }
        guard let provider = embeddingProvider else {
            throw InferenceRuntimeError.invalidState(
                "CPUDrafter has no embedding provider: connect it to the target before drafting")
        }
        let row = try await provider(token)
        embedCache.insert(row, for: token)
        return row
    }

    /// Why a target cache `channels` wide cannot serve this head's `headWidth`-wide query heads, or
    /// nil when it can: the cache must hold whole K/V heads, and they must split the query heads
    /// evenly, as `attend` assumes.
    static func layoutProblem(cache: String, channels: Int, headWidth: Int, queryHeads: Int) -> String? {
        guard channels >= headWidth, channels % headWidth == 0 else {
            return "the target's \(cache) cache is \(channels) channels wide, which does not hold whole "
                + "\(headWidth)-wide heads; the pack was exported for a different model"
        }
        let kvHeads = channels / headWidth
        guard kvHeads <= queryHeads, queryHeads % kvHeads == 0 else {
            return "the target's \(cache) cache holds \(kvHeads) K/V heads, which do not split the draft head's "
                + "\(queryHeads) query heads evenly; the pack was exported for a different model"
        }
        return nil
    }

    /// Checks, before the first draft, that the target's shared caches fit this head: whole K/V heads
    /// that split its query heads evenly, and a sliding ring at least as deep as the head's window.
    ///
    /// - Throws: `invalidArgument` naming the mismatch.
    package func checkTarget(
        _ caches: (StaticShapeEngine.SharedKVCacheKind, (StaticShapeEngine.SharedKVCache) -> Void) throws -> Void
    ) throws {
        var problems: [String] = []
        try caches(.slidingK) { c in
            if let p = Self.layoutProblem(
                cache: "sliding", channels: c.channels, headWidth: head.headDim, queryHeads: head.heads)
            {
                problems.append(p)
            }
            if slidingWindow > c.ctx {
                problems.append(
                    "the draft head attends to \(slidingWindow) sliding positions, but the target's sliding ring "
                        + "holds \(c.ctx); the pack was exported for a different model")
            }
        }
        try caches(.fullK) { c in
            if let p = Self.layoutProblem(
                cache: "full", channels: c.channels, headWidth: head.fullHeadDim, queryHeads: head.heads)
            {
                problems.append(p)
            }
        }
        if let first = problems.first { throw InferenceRuntimeError.invalidArgument(first) }
    }

    /// The sliding-cache key row `row` of the last round, oldest position first, or nil past it.
    func slidingKey(row: Int) -> ArraySlice<Float>? {
        guard !slidingK.isEmpty, slidingChannels > 0, (row + 1) * slidingChannels <= slidingK.count else { return nil }
        return slidingK[(row * slidingChannels)..<((row + 1) * slidingChannels)]
    }

    /// The mirrored full-cache value row of `position`, or nil when the ring does not hold it.
    func mirroredValue(at position: Int) -> ArraySlice<Float>? {
        guard (keptStart..<keptEnd).contains(position), !fullV.isEmpty else { return nil }
        let channels = fullV.count / fullAttentionWindow
        let row = position % fullAttentionWindow
        return fullV[(row * channels)..<((row + 1) * channels)]
    }

    /// The mirrored full-cache key row of `position`, or nil when the ring does not hold it.
    func mirroredKey(at position: Int) -> ArraySlice<Float>? {
        guard (keptStart..<keptEnd).contains(position), !fullK.isEmpty else { return nil }
        let channels = fullK.count / fullAttentionWindow
        let row = position % fullAttentionWindow
        return fullK[(row * channels)..<((row + 1) * channels)]
    }

    /// Brings the full-cache ring up to the last ``fullAttentionWindow`` positions of
    /// `cachedTokens` and reads the sliding window before them.
    ///
    /// Element (slot `r`, channel `d`) of a shared cache is at
    /// `base[(d / factor) * ctx * factor + r * factor + d % factor]`; the sliding cache is a ring,
    /// so position `p` is in slot `p % ctx`.
    private func readSharedKV(
        _ caches: (StaticShapeEngine.SharedKVCacheKind, (StaticShapeEngine.SharedKVCache) -> Void) throws -> Void,
        cachedTokens: ArraySlice<Int32>
    ) throws -> (slidingRows: Int, slidingChannels: Int, fullChannels: Int) {
        let kvLen = cachedTokens.count
        let window = fullAttentionWindow
        let start = max(0, kvLen - window)
        // Kept rows stay valid up to the first position whose token changed (a rollback, or another
        // prompt). A window reaching below the kept rows, after a rollback past them, is read anew:
        // the target's full cache is addressed by position, so those rows are still there.
        let unchanged = min(keptTokens.resolve(input: Array(cachedTokens)).commonPrefix, keptEnd)
        let first = start < keptStart ? start : max(start, unchanged)
        /// Copies `positions` of `c` into `out`, one `c.channels`-wide row each: row `p % ringRows`
        /// of a ring of `ringRows`, or appended in order when `ringRows` is nil. `c` is the sliding
        /// cache when `targetRing` (position `p` in slot `p % c.ctx`), else the full cache.
        func read(
            _ c: StaticShapeEngine.SharedKVCache, positions: Range<Int>, targetRing: Bool, ringRows: Int?,
            into out: inout [Float]
        ) {
            let end = out.count
            if ringRows == nil { out.append(contentsOf: repeatElement(0, count: positions.count * c.channels)) }
            let f = c.factor
            out.withUnsafeMutableBufferPointer { dst in
                for (i, p) in positions.enumerated() {
                    let slot = targetRing ? p % c.ctx : p
                    let row = dst.baseAddress! + (ringRows.map { (p % $0) * c.channels } ?? (end + i * c.channels))
                    for d in 0..<c.channels { row[d] = Float(c.base[(d / f) * c.ctx * f + slot * f + d % f]) }
                }
            }
        }
        var fullChannels = 0
        var layoutError: String?
        for (kind, isKey) in [(StaticShapeEngine.SharedKVCacheKind.fullK, true), (.fullV, false)] {
            try caches(kind) { c in
                guard kvLen <= c.ctx else {
                    layoutError = "the full cache holds \(c.ctx) positions, fewer than the \(kvLen) cached"
                    return
                }
                fullChannels = c.channels
                let size = window * c.channels
                // The first round, or a target of another width, starts the ring over.
                let fresh = (isKey ? fullK : fullV).count != size
                if fresh {
                    if isKey {
                        fullK = [Float](repeating: 0, count: size)
                    } else {
                        fullV = [Float](repeating: 0, count: size)
                    }
                }
                let from = fresh ? start : first
                if isKey {
                    read(c, positions: from..<kvLen, targetRing: false, ringRows: window, into: &fullK)
                } else {
                    read(c, positions: from..<kvLen, targetRing: false, ringRows: window, into: &fullV)
                }
            }
        }
        if let layoutError {
            // A partly written ring holds nothing to keep.
            keptTokens.clear()
            (keptStart, keptEnd) = (0, 0)
            throw InferenceRuntimeError.invalidState("CPUDrafter: \(layoutError)")
        }
        keptTokens.truncate(to: unchanged)
        keptTokens.append(contentsOf: cachedTokens[(cachedTokens.startIndex + unchanged)...])
        keptStart = start
        keptEnd = kvLen

        var slidingRows = 0
        var slidingChannels = 0
        defer { self.slidingChannels = slidingChannels }
        slidingK.removeAll(keepingCapacity: true)
        slidingV.removeAll(keepingCapacity: true)
        for (kind, isKey) in [(StaticShapeEngine.SharedKVCacheKind.slidingK, true), (.slidingV, false)] {
            try caches(kind) { c in
                slidingRows = min(kvLen, slidingWindow, c.ctx)
                slidingChannels = c.channels
                if isKey {
                    read(c, positions: (kvLen - slidingRows)..<kvLen, targetRing: true, ringRows: nil, into: &slidingK)
                } else {
                    read(c, positions: (kvLen - slidingRows)..<kvLen, targetRing: true, ringRows: nil, into: &slidingV)
                }
            }
        }
        for (name, channels, width) in [
            ("sliding", slidingChannels, head.headDim), ("full", fullChannels, head.fullHeadDim),
        ] {
            if let problem = Self.layoutProblem(
                cache: name, channels: channels, headWidth: width, queryHeads: head.heads)
            {
                throw InferenceRuntimeError.invalidState(problem)
            }
        }
        return (slidingRows, slidingChannels, fullChannels)
    }
}
