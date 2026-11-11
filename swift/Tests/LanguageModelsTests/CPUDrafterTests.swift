// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// The CPU draft head: its kernels against plain references, its pack validation, and a whole draft
/// step on a small random head against a straightforward Double implementation of the same layers.
///
/// Every draft is verified, so a wrong kernel would only lower acceptance, which nothing on a device
/// reports as an error; these tests are where it shows.
@Suite("CPU draft head")
struct CPUDrafterTests {
    private func close(_ a: Float, _ b: Double, rel: Double = 2e-3, abs: Double = 1e-4) -> Bool {
        Swift.abs(Double(a) - b) <= abs + rel * Swift.max(Swift.abs(Double(a)), Swift.abs(b))
    }

    /// A deterministic value in -1...1 for index `i` of a tensor seeded with `seed`.
    private static func value(_ i: Int, _ seed: Int) -> Float {
        sinf(Float(i) * 0.731 + Float(seed) * 1.37)
    }

    // MARK: - Kernels

    @Test("An fp16 projection matches a Double matrix-vector product", arguments: [(5, 11), (3, 64), (1024, 300)])
    func fp16Projection(m: Int, n: Int) {
        let w = (0..<(m * n)).map { Float16(Self.value($0, 1)) }
        let x = (0..<n).map { Self.value($0, 2) }
        var y = [Float](repeating: 0, count: m)
        w.withUnsafeBufferPointer { CPUDraftHead.mv(.f16($0.baseAddress!), x, m: m, n: n, into: &y) }
        for r in 0..<m {
            let ref = (0..<n).reduce(0.0) { $0 + Double(w[r * n + $1]) * Double(x[$1]) }
            #expect(close(y[r], ref), "row \(r)")
        }
    }

    @Test("An int8 projection applies each row's scale", arguments: [(4, 9), (1024, 256)])
    func int8Projection(m: Int, n: Int) {
        let q = (0..<(m * n)).map { Int8(truncatingIfNeeded: ($0 * 37) % 255 - 127) }
        let scale = (0..<m).map { Float16(0.01 * Float($0 % 7 + 1)) }
        let x = (0..<n).map { Self.value($0, 3) }
        var y = [Float](repeating: 0, count: m)
        q.withUnsafeBufferPointer { qp in
            scale.withUnsafeBufferPointer { sp in
                CPUDraftHead.mv(.i8(qp.baseAddress!, scale: sp.baseAddress!), x, m: m, n: n, into: &y)
            }
        }
        for r in 0..<m {
            let dot = (0..<n).reduce(0.0) { $0 + Double(q[r * n + $1]) * Double(x[$1]) }
            #expect(close(y[r], dot * Double(scale[r])), "row \(r)")
        }
    }

    @Test("RMS norm divides by the root mean square and applies the gain")
    func rmsNorm() {
        var x: [Float] = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        let gain = (0..<8).map { Float16(1 + 0.1 * Float($0)) }
        let rms = (x.reduce(0.0) { $0 + Double($1 * $1) } / 8 + 1e-6).squareRoot()
        let expected = (0..<8).map { Double(x[$0]) / rms * Double(gain[$0]) }
        gain.withUnsafeBufferPointer { CPUDraftHead.rmsnorm(&x, $0.baseAddress!, eps: 1e-6) }
        for i in 0..<8 { #expect(close(x[i], expected[i], rel: 1e-5)) }
    }

    @Test("GELU times the up projection uses the tanh approximation")
    func geluTimes() {
        var g = stride(from: Float(-4), through: 4, by: 0.5).map { $0 }
        let u = g.indices.map { Float($0) * 0.1 + 0.5 }
        let expected = g.indices.map { i -> Double in
            let v = Double(g[i])
            return 0.5 * v * (1 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))) * Double(u[i])
        }
        CPUDraftHead.geluTimes(&g, u)
        for i in g.indices { #expect(close(g[i], expected[i], rel: 1e-5, abs: 1e-6)) }
    }

    @Test("Partial rotary turns the leading pairs and leaves the rest in place")
    func partialRotary() {
        let hd = 16
        let inv = CPUDraftHead.inverseFrequencies(headDim: hd, theta: 1_000_000, partial: 0.25)
        #expect(inv.count == 8 && inv[0] == 1 && inv[1] > 0 && inv[2...].allSatisfy { $0 == 0 })
        let (cos, sin) = CPUDraftHead.rope(position: 9, invFreq: inv)
        let q0 = (0..<(2 * hd)).map { Float($0 % 7) - 3 }
        var q = q0
        CPUDraftHead.applyRotary(&q, heads: 2, headDim: hd, cos: cos, sin: sin)
        for h in 0..<2 {
            for d in 0..<8 {
                let (a, b) = (q0[h * hd + d], q0[h * hd + d + 8])
                if d < 2 {
                    #expect(close(q[h * hd + d], Double(a * cos[d] - b * sin[d]), rel: 1e-5))
                    #expect(close(q[h * hd + d + 8], Double(b * cos[d] + a * sin[d]), rel: 1e-5))
                } else {
                    #expect(q[h * hd + d] == a && q[h * hd + d + 8] == b)
                }
            }
        }
    }

    @Test("Attention matches softmax(q·K)·V per head", arguments: [(1, 37), (2, 64), (1, 1)])
    func attention(groups: Int, rows: Int) {
        let heads = 4
        let hd = 8
        let channels = groups * hd
        let q = (0..<(heads * hd)).map { Self.value($0, 4) }
        let k = (0..<(rows * channels)).map { Self.value($0, 5) }
        let v = (0..<(rows * channels)).map { Self.value($0, 6) }
        var out = [Float](repeating: 0, count: heads * hd)
        k.withUnsafeBufferPointer { kp in
            v.withUnsafeBufferPointer { vp in
                CPUDraftHead.attend(
                    q: q, k: kp.baseAddress!, v: vp.baseAddress!, rows: rows, channels: channels,
                    headDim: hd, heads: heads, into: &out)
            }
        }
        let perGroup = heads / groups
        for h in 0..<heads {
            let base = (h / perGroup) * hd
            let scores = (0..<rows).map { s in
                (0..<hd).reduce(0.0) { $0 + Double(q[h * hd + $1]) * Double(k[s * channels + base + $1]) }
            }
            let top = scores.max()!
            let weights = scores.map { exp($0 - top) }
            let total = weights.reduce(0, +)
            for d in 0..<hd {
                let ref = (0..<rows).reduce(0.0) { $0 + weights[$1] * Double(v[$1 * channels + base + d]) } / total
                #expect(close(out[h * hd + d], ref), "head \(h) channel \(d)")
            }
        }
    }

    // MARK: - A small head

    /// The geometry of the random head the step tests use: two sliding layers and a full one.
    private enum G {
        static let vocab = 16, centroids = 4, hidden = 8, backbone = 6, heads = 2, headDim = 4
        static let fullHeadDim = 8, ffn = 12, layers = 3
    }

    private struct Pack {
        var tensors: [(name: String, dtype: String, shape: [Int], values: [Float])] = []

        mutating func add(_ name: String, _ shape: [Int], seed: Int, dtype: String = "float16") {
            let n = shape.reduce(1, *)
            tensors.append((name, dtype, shape, (0..<n).map { CPUDrafterTests.value($0, seed) * 0.5 }))
        }
    }

    private func headPack(int8Attention: Bool = false, dropping: String? = nil, index extra: [String: Any] = [:])
        throws -> (url: URL, pack: Pack)
    {
        var p = Pack()
        p.add("embed_perm_f16", [G.vocab, G.hidden], seed: 10)
        p.tensors.append(("token_ordering_i32", "int32", [G.vocab], (0..<G.vocab).map { Float((G.vocab - 1 - $0)) }))
        p.add("centroids_f16", [G.centroids, G.hidden], seed: 11)
        p.add("pre_projection.weight", [G.hidden, 2 * G.backbone], seed: 12)
        p.add("post_projection.weight", [G.backbone, G.hidden], seed: 13)
        p.add("model.norm.weight", [G.hidden], seed: 14)
        for L in 0..<G.layers {
            let hd = L == G.layers - 1 ? G.fullHeadDim : G.headDim
            let s = 20 + 10 * L
            let pre = "model.layers.\(L)."
            for (i, norm) in [
                "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ].enumerated() {
                p.add(pre + norm + ".weight", [G.hidden], seed: s + i)
            }
            p.add(pre + "self_attn.q_norm.weight", [hd], seed: s + 4)
            p.add(pre + "layer_scalar", [1], seed: s + 5)
            for (name, shape, seed) in [
                ("self_attn.q_proj", [G.heads * hd, G.hidden], s + 6),
                ("self_attn.o_proj", [G.hidden, G.heads * hd], s + 7),
                ("mlp.gate_proj", [G.ffn, G.hidden], s + 8), ("mlp.up_proj", [G.ffn, G.hidden], s + 9),
                ("mlp.down_proj", [G.hidden, G.ffn], s + 10),
            ] {
                if int8Attention && name.hasPrefix("self_attn") {
                    p.add(pre + name + ".weight", shape, seed: seed, dtype: "int8")
                    p.tensors.append(
                        (pre + name + ".weight.scale", "float16", [shape[0]], Array(repeating: 0.004, count: shape[0])))
                } else {
                    p.add(pre + name + ".weight", shape, seed: seed)
                }
            }
        }
        if let dropping { p.tensors.removeAll { $0.name == dropping } }

        // A safetensors file: an 8-byte header length, the JSON header, then the tensor bytes.
        var body = Data()
        var header: [String: Any] = [:]
        for t in p.tensors {
            let begin = body.count
            switch t.dtype {
            case "float16": for v in t.values { withUnsafeBytes(of: Float16(v)) { body.append(contentsOf: $0) } }
            case "int32": for v in t.values { withUnsafeBytes(of: Int32(v)) { body.append(contentsOf: $0) } }
            default: for v in t.values { body.append(UInt8(bitPattern: Int8(clamping: Int((v * 250).rounded())))) }
            }
            let dtype = ["float16": "F16", "int32": "I32", "int8": "I8"][t.dtype]!
            header[t.name] = ["dtype": dtype, "shape": t.shape, "data_offsets": [begin, body.count]]
        }
        var metadata: [String: Any] = [
            "format": "gemma4_draft_head", "version": 1, "vocab": G.vocab, "centroids": G.centroids,
            "per_centroid": G.vocab / G.centroids, "layers": G.layers, "hidden": G.hidden, "heads": G.heads,
            "head_dim": G.headDim, "full_head_dim": G.fullHeadDim, "ffn": G.ffn, "sliding_window": 4,
            "sliding_theta": 10_000.0, "full_theta": 1_000_000.0, "full_partial_rotary_factor": 0.25,
        ]
        metadata.merge(extra) { $1 }
        header["__metadata__"] = metadata.mapValues { "\($0)" }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        // Pad the header so the tensors start on an 8-byte boundary, as safetensors writers do.
        while (8 + json.count) % 8 != 0 { json.append(UInt8(ascii: " ")) }
        var file = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(json)
        file.append(body)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("cpu-head-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("draft_head.safetensors")
        try file.write(to: url)
        return (url, p)
    }

    /// The draft step written out plainly in Double, from the values the pack was built from
    /// (rounded to the stored type), for comparison with ``CPUDraftHead/step(embeds:kv:position:)``.
    private func referenceStep(
        _ p: Pack, embeds: [Double], sliding: (k: [Double], v: [Double], rows: Int),
        full: (k: [Double], v: [Double], rows: Int), position: Int, int8Attention: Bool
    ) -> (x: [Double], logits: [Double], hidden: [Double]) {
        func t(_ name: String) -> [Double] {
            let e = p.tensors.first { $0.name == name }!
            switch e.dtype {
            case "int8":
                let scale = Double(Float16(0.004))
                return e.values.map { Double(Int8(clamping: Int(($0 * 250).rounded()))) * scale }
            default: return e.values.map { Double(Float16($0)) }
            }
        }
        func mv(_ name: String, _ x: [Double]) -> [Double] {
            let w = t(name)
            let n = x.count
            return (0..<(w.count / n)).map { r in (0..<n).reduce(0.0) { $0 + w[r * n + $1] * x[$1] } }
        }
        func norm(_ x: [Double], _ g: [Double]) -> [Double] {
            let rms = (x.reduce(0) { $0 + $1 * $1 } / Double(x.count) + 1e-6).squareRoot()
            return x.indices.map { x[$0] / rms * g[$0] }
        }
        var x = mv("pre_projection.weight", embeds)
        for L in 0..<G.layers {
            let isFull = L == G.layers - 1
            let hd = isFull ? G.fullHeadDim : G.headDim
            let pre = "model.layers.\(L)."
            var q = mv(pre + "self_attn.q_proj.weight", norm(x, t(pre + "input_layernorm.weight")))
            let qn = t(pre + "self_attn.q_norm.weight")
            for h in 0..<G.heads {
                let head = norm(Array(q[(h * hd)..<((h + 1) * hd)]), qn)
                for d in 0..<hd { q[h * hd + d] = head[d] }
            }
            let theta = isFull ? 1_000_000.0 : 10_000.0
            let rotated = isFull ? hd / 2 / 4 : hd / 2
            for h in 0..<G.heads {
                for d in 0..<rotated {
                    let angle = Double(position) * pow(theta, -2 * Double(d) / Double(hd))
                    let (a, b) = (q[h * hd + d], q[h * hd + d + hd / 2])
                    q[h * hd + d] = a * cos(angle) - b * sin(angle)
                    q[h * hd + d + hd / 2] = b * cos(angle) + a * sin(angle)
                }
            }
            let cache = isFull ? full : sliding
            var attention = [Double](repeating: 0, count: G.heads * hd)
            for h in 0..<G.heads {
                let scores = (0..<cache.rows).map { s in
                    (0..<hd).reduce(0.0) { $0 + q[h * hd + $1] * cache.k[s * hd + $1] }
                }
                let top = scores.max() ?? 0
                let weights = scores.map { exp($0 - top) }
                let total = weights.reduce(0, +)
                for d in 0..<hd {
                    attention[h * hd + d] =
                        (0..<cache.rows).reduce(0.0) { $0 + weights[$1] * cache.v[$1 * hd + d] } / total
                }
            }
            let o = norm(mv(pre + "self_attn.o_proj.weight", attention), t(pre + "post_attention_layernorm.weight"))
            x = x.indices.map { x[$0] + o[$0] }
            let z = norm(x, t(pre + "pre_feedforward_layernorm.weight"))
            let gate = mv(pre + "mlp.gate_proj.weight", z)
            let up = mv(pre + "mlp.up_proj.weight", z)
            let act = gate.indices.map { i -> Double in
                let v = gate[i]
                return 0.5 * v * (1 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))) * up[i]
            }
            let down = norm(mv(pre + "mlp.down_proj.weight", act), t(pre + "post_feedforward_layernorm.weight"))
            let scalar = t(pre + "layer_scalar")[0]
            x = x.indices.map { (x[$0] + down[$0]) * scalar }
        }
        x = norm(x, t("model.norm.weight"))
        return (x, mv("centroids_f16", x), mv("post_projection.weight", x))
    }

    @Test("A draft step matches the reference layers, with fp16 or int8 attention", arguments: [false, true])
    func stepMatchesReference(int8Attention: Bool) throws {
        let (url, p) = try headPack(int8Attention: int8Attention)
        let pack = try DraftPack(safetensors: url, role: .head)
        let head = CPUDraftHead(pack: pack, backboneHidden: G.backbone)
        let embeds = (0..<(2 * G.backbone)).map { Self.value($0, 7) }
        // One K/V head each: the sliding cache's rows are `headDim` wide, the full cache's `fullHeadDim`.
        let sliding = (
            k: (0..<(3 * G.headDim)).map { Self.value($0, 8) }, v: (0..<(3 * G.headDim)).map { Self.value($0, 9) }
        )
        let full = (
            k: (0..<(5 * G.fullHeadDim)).map { Self.value($0, 15) },
            v: (0..<(5 * G.fullHeadDim)).map { Self.value($0, 16) }
        )
        let got = sliding.k.withUnsafeBufferPointer { sk in
            sliding.v.withUnsafeBufferPointer { sv in
                full.k.withUnsafeBufferPointer { fk in
                    full.v.withUnsafeBufferPointer { fv in
                        head.step(
                            embeds: embeds,
                            kv: CPUDraftHead.SharedKV(
                                slidingK: sk.baseAddress!, slidingV: sv.baseAddress!, slidingRows: 3,
                                slidingChannels: G.headDim, fullK: fk.baseAddress!, fullV: fv.baseAddress!, fullRows: 5,
                                fullChannels: G.fullHeadDim),
                            position: 6)
                    }
                }
            }
        }
        let ref = referenceStep(
            p, embeds: embeds.map(Double.init),
            sliding: (sliding.k.map(Double.init), sliding.v.map(Double.init), 3),
            full: (full.k.map(Double.init), full.v.map(Double.init), 5), position: 6, int8Attention: int8Attention)
        for i in 0..<G.hidden { #expect(close(got.x[i], ref.x[i]), "x[\(i)]") }
        for i in 0..<G.centroids { #expect(close(got.centroidLogits[i], ref.logits[i]), "centroid \(i)") }
        for i in 0..<G.backbone { #expect(close(got.hidden[i], ref.hidden[i]), "hidden[\(i)]") }

        // The scan over every centroid is the full-vocabulary argmax, mapped through the ordering.
        var scanner = DraftScan.Scanner(pack: pack, topK: G.centroids)
        let table = p.tensors[0].values.map { Double(Float16($0)) }
        let scores = (0..<G.vocab).map { r in (0..<G.hidden).reduce(0.0) { $0 + table[r * G.hidden + $1] * ref.x[$1] } }
        let best = scores.indices.max { scores[$0] < scores[$1] }!
        #expect(scanner.bestToken(x: got.x, centroidLogits: got.centroidLogits) == Int32(G.vocab - 1 - best))
    }

    // MARK: - Pack validation

    private func loadError(_ load: () throws -> Void) -> String {
        do {
            try load()
            return ""
        } catch {
            return "\(error)"
        }
    }

    @Test("A head pack missing a layer tensor names it and the exporter")
    func missingLayerTensor() throws {
        let (url, _) = try headPack(dropping: "model.layers.1.mlp.up_proj.weight")
        let message = loadError { _ = try DraftPack(safetensors: url, role: .head) }
        #expect(message.contains("model.layers.1.mlp.up_proj.weight") && message.contains("export_draft_pack.py"))
    }

    @Test("A head pack without its RoPE parameters is rejected")
    func missingRope() throws {
        let (url, _) = try headPack(index: ["full_theta": 0])
        #expect(loadError { _ = try DraftPack(safetensors: url, role: .head) }.contains("full_theta"))
    }

    @Test("A scan-only pack loads for the scan but not as a head")
    func scanPackIsNotAHead() throws {
        let (url, _) = try headPack(dropping: "centroids_f16")
        _ = try DraftPack(safetensors: url, role: .scan)
        #expect(loadError { _ = try DraftPack(safetensors: url, role: .head) }.contains("centroids_f16"))
    }

    @Test("The drafter refuses a pack for a target of another width")
    func wrongBackbone() throws {
        let (url, _) = try headPack()
        let message = loadError {
            _ = try CPUDrafter(pack: url, backboneHiddenSize: G.backbone + 2, maxContextLength: 64)
        }
        #expect(message.contains("hidden size is \(G.backbone + 2)"))
        _ = try CPUDrafter(pack: url, backboneHiddenSize: G.backbone, maxContextLength: 64, vocabSize: G.vocab)
    }

    @Test("Before the first verify, the drafter proposes the anchor")
    func coldStartProposesTheAnchor() async throws {
        let (url, _) = try headPack()
        let drafter = try CPUDrafter(pack: url, backboneHiddenSize: G.backbone, maxContextLength: 64)
        drafter.sharedKVProvider = { _, _ in }
        var rng = SystemRandomNumberGenerator()
        let proposal = try await drafter.draftAll(confirmedPrefix: [3, 5, 9], count: 3, policy: .greedy, using: &rng)
        #expect(proposal.tokens == [9, 9, 9])
    }

    @Test("A safetensors file of another kind is not taken for a draft head")
    func otherSafetensors() throws {
        let (url, _) = try headPack(index: ["format": "something_else"])
        #expect(loadError { _ = try DraftPack(safetensors: url, role: .head) }.contains("not a draft head pack"))
    }

    @Test("Attention does not depend on the order of the cached rows")
    func attentionIgnoresRowOrder() {
        let (hd, rows, heads) = (8, 5, 2)
        let q = (0..<(heads * hd)).map { Self.value($0, 30) }
        let k = (0..<(rows * hd)).map { Self.value($0, 31) }
        let v = (0..<(rows * hd)).map { Self.value($0, 32) }
        let order = [3, 0, 4, 1, 2]
        let shuffledK = order.flatMap { k[($0 * hd)..<(($0 + 1) * hd)] }
        let shuffledV = order.flatMap { v[($0 * hd)..<(($0 + 1) * hd)] }
        func run(_ k: [Float], _ v: [Float]) -> [Float] {
            var out = [Float](repeating: 0, count: heads * hd)
            k.withUnsafeBufferPointer { kp in
                v.withUnsafeBufferPointer { vp in
                    CPUDraftHead.attend(
                        q: q, k: kp.baseAddress!, v: vp.baseAddress!, rows: rows, channels: hd, headDim: hd,
                        heads: heads, into: &out)
                }
            }
            return out
        }
        let a = run(k, v)
        let b = run(shuffledK, shuffledV)
        for i in a.indices { #expect(close(a[i], Double(b[i]), rel: 1e-5, abs: 1e-6)) }
    }

    /// A stand-in for the target's shared caches: one K/V head per cache, channel-major, the full
    /// cache addressed by position and the sliding cache a ring. A value encodes its position and a
    /// version, so a test can tell which write a mirrored row came from.
    private final class FakeCaches: @unchecked Sendable {
        static let fullDepth = 2048
        static let slidingDepth = 4
        var full: [LogitsScalarType]
        var sliding: [LogitsScalarType]

        init() {
            full = [LogitsScalarType](repeating: 0, count: G.fullHeadDim * Self.fullDepth)
            sliding = [LogitsScalarType](repeating: 0, count: G.headDim * Self.slidingDepth)
        }

        static func value(position: Int, channel: Int, version: Int) -> Float {
            Float(position % 997) / 997 + Float(channel) / 64 + Float(version)
        }

        func write(_ positions: Range<Int>, version: Int) {
            for p in positions {
                for d in 0..<G.fullHeadDim {
                    full[d * Self.fullDepth + p] = LogitsScalarType(
                        Self.value(position: p, channel: d, version: version))
                }
                for d in 0..<G.headDim {
                    sliding[d * Self.slidingDepth + p % Self.slidingDepth] = LogitsScalarType(
                        Self.value(position: p, channel: d, version: version))
                }
            }
        }

        func provide(_ kind: StaticShapeEngine.SharedKVCacheKind, _ body: (StaticShapeEngine.SharedKVCache) -> Void) {
            switch kind {
            case .fullK, .fullV:
                full.withUnsafeBufferPointer {
                    body(.init(base: $0.baseAddress!, channels: G.fullHeadDim, ctx: Self.fullDepth, factor: 1))
                }
            case .slidingK, .slidingV:
                sliding.withUnsafeBufferPointer {
                    body(.init(base: $0.baseAddress!, channels: G.headDim, ctx: Self.slidingDepth, factor: 1))
                }
            }
        }
    }

    /// Whether the drafter's mirrored key for `position` is the one written at `version`.
    private func mirrors(_ drafter: CPUDrafter, _ position: Int, version: Int) -> Bool {
        guard let row = drafter.mirroredKey(at: position) else { return false }
        return row.enumerated().allSatisfy { d, value in
            let expected = Float(LogitsScalarType(FakeCaches.value(position: position, channel: d, version: version)))
            return value == expected
        }
    }

    @Test("Past the full-attention window, the drafter keeps only the window and re-reads after a rollback")
    func fullCacheRing() async throws {
        let (url, _) = try headPack()
        let drafter = try CPUDrafter(pack: url, backboneHiddenSize: G.backbone, maxContextLength: 2048)
        let window = CPUDrafter.defaultFullAttentionWindow
        let caches = FakeCaches()
        drafter.sharedKVProvider = { kind, body in caches.provide(kind, body) }
        drafter.embeddingProvider = { _ in [Float](repeating: 0.1, count: G.backbone) }
        let seed = DrafterFeatures(
            values: [Float](repeating: 0.2, count: G.backbone), positionCount: 1, hiddenDim: G.backbone,
            startPosition: 0)
        var rng = SystemRandomNumberGenerator()
        func draft(_ prefix: [Int32]) async throws {
            try await drafter.commit(confirmedLength: prefix.count, features: seed)
            _ = try await drafter.draftAll(confirmedPrefix: prefix, count: 1, policy: .greedy, using: &rng)
        }

        // 1,100 cached positions: only the last 1,024 are held, whatever the context length.
        var tokens = (0..<1101).map { Int32($0 % 13) }
        caches.write(0..<1100, version: 0)
        try await draft(tokens)
        #expect(drafter.mirroredKey(at: 1100 - window - 1) == nil)
        #expect(mirrors(drafter, 1100 - window, version: 0) && mirrors(drafter, 1099, version: 0))

        // The target rolls back to 1,050 and caches other tokens from there on, up to 1,080.
        tokens = Array(tokens[..<1050]) + (1050..<1081).map { Int32(($0 + 5) % 13) }
        caches.write(1050..<1080, version: 1)
        try await draft(tokens)
        #expect(mirrors(drafter, 1049, version: 0) && mirrors(drafter, 1050, version: 1))
        #expect(mirrors(drafter, 1079, version: 1) && drafter.mirroredKey(at: 1080) == nil)
        // The window now starts below the rows kept before; they are read from the target again.
        #expect(mirrors(drafter, 1080 - window, version: 0))

        // A rollback to 100 positions: the window is everything, read anew.
        try await drafter.reset(to: 101)
        tokens = Array(tokens[..<201])
        try await draft(tokens)
        #expect(mirrors(drafter, 0, version: 0) && mirrors(drafter, 199, version: 0))
        #expect(drafter.mirroredKey(at: 200) == nil)
    }
}
