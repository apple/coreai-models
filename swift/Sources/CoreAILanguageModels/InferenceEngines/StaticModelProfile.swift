// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - Static Model Profile

/// A model family's contribution to the static-shape engine: the extra input providers it
/// needs beyond the universal decoder set. Selected as *data* from the descriptor's declared
/// inputs/states — no per-model engine subclass, no `matches(model:)` returning an engine.
struct StaticModelProfile {
    let name: String
    let extraHandlers: [any SyncInputHandler]

    /// Assemble the full provider list for a model: universal base providers (gated on the
    /// descriptor actually declaring each input) + the family-specific profile providers.
    static func assembleInputHandlers(
        descriptor desc: InferenceFunctionDescriptor,
        bundleURL: URL
    ) throws -> (providers: [any SyncInputHandler], profileName: String) {
        let inputs = Set(desc.inputNames)
        var providers: [any SyncInputHandler] = []

        // ── Base providers (universal) ──
        // `transformer_input` / `embedding_table` are produced by the engine directly
        // (embedding gather needs the model's gather graph), not by a SyncInputHandler.
        if let pos = desc.inputNames.first(where: { $0.contains("pos") }) {
            providers.append(PositionIdsProvider(name: pos))
        }
        if inputs.contains("causal_mask") {
            providers.append(CausalMaskProvider(name: "causal_mask"))
        }
        // Base step is the non-sliding step counter (`in_step`); sliding_in_step is a profile input.
        if let step = desc.inputNames.first(where: {
            $0.contains("step") && !$0.contains("pos") && !$0.contains("sliding")
        }) {
            providers.append(StepProvider(name: step))
        }

        // ── Profile selection (pure function of declared inputs/states) ──
        let profile = try resolve(descriptor: desc, bundleURL: bundleURL)
        providers.append(contentsOf: profile.extraHandlers)
        return (providers, profile.name)
    }

    static func resolve(descriptor desc: InferenceFunctionDescriptor, bundleURL: URL) throws -> StaticModelProfile {
        let inputs = Set(desc.inputNames)
        let states = Set(desc.stateNames)

        // Models with per-layer embeddings + dual-RoPE + a sliding-window KV cache.
        if inputs.contains("ple_embeddings") || inputs.contains("rope_cos")
            || states.contains("sliding_key_cache")
        {
            var extras: [any SyncInputHandler] = []
            let meta = StaticBundleMetadata.load(bundleURL: bundleURL)

            if inputs.contains("rope_cos") {
                // Always cover rope_cos/rope_sin when declared. Uses metadata dual-RoPE params
                // when available; falls back to identity (no rotation) so coverage passes and the
                // graph runs even if params are missing (numeric parity then needs the metadata).
                extras.append(RoPEProvider(config: meta?.rope))
            }
            // A declared ple_embeddings input with a missing/broken sidecar is fatal — let it throw.
            if inputs.contains("ple_embeddings") {
                extras.append(PLEProvider(table: try PerLayerEmbeddings(bundleURL: bundleURL)))
            }
            if inputs.contains("sliding_in_step") {
                let ring = slidingRingDepth(descriptor: desc)
                extras.append(SlidingStepProvider(ringDepth: ring))
            }
            if inputs.contains("sliding_causal_mask") {
                extras.append(
                    SlidingMaskProvider(
                        ringDepth: slidingRingDepth(descriptor: desc),
                        window: meta?.slidingWindow ?? 0))
            }
            return StaticModelProfile(name: "ple-dual-rope-sliding", extraHandlers: extras)
        }

        // Hybrid models with a DeltaNet/SSM update flag (conv/recurrent states handled generically).
        if inputs.contains("ssm_update_flag") {
            return StaticModelProfile(
                name: "ssm-hybrid",
                extraHandlers: [SSMUpdateFlagProvider(name: "ssm_update_flag")])
        }

        return StaticModelProfile(name: "vanilla", extraHandlers: [])
    }

    /// Ring depth = the sliding cache's sequence (last) dim from the descriptor.
    private static func slidingRingDepth(descriptor desc: InferenceFunctionDescriptor) -> Int {
        if case .ndArray(let d) = desc.stateDescriptor(of: "sliding_key_cache"), let last = d.shape.last {
            return last
        }
        return 0
    }
}

// MARK: - Bundle metadata (rope / sliding window)

/// Reads the rope + sliding-window fields from the bundle's `metadata.json`. These live under
/// `language.*` and are not in the base `ModelConfig`, so we parse them here on demand.
struct StaticBundleMetadata {
    struct RoPE {
        let slidingHeadDim: Int
        let globalHeadDim: Int
        let slidingTheta: Double
        let globalTheta: Double
        let partialRotaryFactor: Double
    }
    let rope: RoPE?
    let slidingWindow: Int?

    static func load(bundleURL: URL) -> StaticBundleMetadata? {
        // `ModelBundle` resolves the manifest metadata.json (handling the .aimodel's own internal
        // metadata) and preserves its raw bytes. `bundleURL` may be the bundle dir or the .aimodel
        // inside it, so try both. We read `rope`/`sliding_window` from the raw bytes because they
        // aren't part of the typed `LanguageConfig`.
        guard
            let bundle = (try? ModelBundle(at: bundleURL))
                ?? (try? ModelBundle(at: bundleURL.deletingLastPathComponent())),
            let json = try? JSONSerialization.jsonObject(with: bundle.raw) as? [String: Any],
            let lang = json["language"] as? [String: Any]
        else { return nil }

        var rope: RoPE? = nil
        if let r = lang["rope"] as? [String: Any] {
            func dbl(_ k: String) -> Double? { (r[k] as? NSNumber)?.doubleValue }
            func int(_ k: String) -> Int? { (r[k] as? NSNumber)?.intValue }
            rope = RoPE(
                slidingHeadDim: int("sliding_head_dim") ?? 0,
                globalHeadDim: int("global_head_dim") ?? 0,
                slidingTheta: dbl("sliding_rope_theta") ?? 10000,
                globalTheta: dbl("global_rope_theta") ?? 1_000_000,
                partialRotaryFactor: dbl("partial_rotary_factor") ?? 1.0)
        }
        return StaticBundleMetadata(rope: rope, slidingWindow: (lang["sliding_window"] as? NSNumber)?.intValue)
    }
}

// MARK: - SSM / hybrid provider

/// `ssm_update_flag` — 1.0 on genuinely-new positions, 0.0 on re-sent/padding, so the DeltaNet
/// recurrence treats re-sent tokens as exact no-ops. (fp16, one element per block column.)
final class SSMUpdateFlagProvider: SyncInputHandler {
    private let name: String
    var inputNames: [String] { [name] }
    init(name: String) { self.name = name }
    func prepare(_ ctx: InputContext) async throws -> [String: NDArray] {
        guard let d = ctx.descriptors[name] else { return [:] }
        var flag = NDArray(descriptor: d)
        var view = flag.mutableView(as: LogitsScalarType.self)
        guard var span = view.contiguousElements else {
            throw InferenceRuntimeError.invalidState("ssm_update_flag non-contiguous")
        }
        // Offset into the block where genuinely-new (not re-sent) tokens begin.
        let newTokenStart = ctx.processedTokenCount - ctx.alignedStep
        for i in 0..<ctx.batchSize {
            span[i] = (i >= newTokenStart && i < ctx.tokens.count) ? 1.0 : 0.0
        }
        return [name: flag]
    }
}

// MARK: - Providers for PLE / dual-RoPE / sliding-window models

/// `sliding_in_step` — Int32 ring-write offset (`alignedStep % ringDepth`) so the graph needs
/// no in-graph remainder op.
final class SlidingStepProvider: SyncInputHandler {
    private let ringDepth: Int
    var inputNames: [String] { ["sliding_in_step"] }
    init(ringDepth: Int) { self.ringDepth = ringDepth }
    func prepare(_ ctx: InputContext) async throws -> [String: NDArray] {
        guard let d = ctx.descriptors["sliding_in_step"] else { return [:] }
        var step = NDArray(descriptor: d)
        var view = step.mutableView(as: Int32.self)
        guard var span = view.contiguousElements else {
            throw InferenceRuntimeError.invalidState("sliding_in_step non-contiguous")
        }
        span[0] = ringDepth > 0 ? Int32(ctx.alignedStep % ringDepth) : Int32(ctx.alignedStep)
        return ["sliding_in_step": step]
    }
}

/// `sliding_causal_mask` — like the causal mask but windowed to the last `window` keys and
/// indexed into the ring by `pos % ringDepth`.
final class SlidingMaskProvider: SyncInputHandler {
    private let ringDepth: Int
    private let window: Int
    var inputNames: [String] { ["sliding_causal_mask"] }
    init(ringDepth: Int, window: Int) {
        self.ringDepth = ringDepth
        self.window = window
    }
    func prepare(_ ctx: InputContext) async throws -> [String: NDArray] {
        guard let d = ctx.descriptors["sliding_causal_mask"] else { return [:] }
        var mask = NDArray(descriptor: d)
        var view = mask.mutableView(as: LogitsScalarType.self)
        let ring = ringDepth
        let window = window
        view.withUnsafeMutablePointer { ptr, shape, strides in
            let ringLen = shape[1]  // sliding cache ring length (e.g. 576)
            for slot in 0..<ringLen {
                for query in 0..<shape[3] {
                    ptr[slot &* strides[1] &+ query &* strides[3]] = LogitsScalarType(-40000.0)
                }
            }
            // Unmask the window of keys visible to each query position.
            for query in 0..<ctx.tokens.count {
                let queryPos = ctx.alignedStep + query
                let lowest = window > 0 ? max(0, queryPos - window + 1) : 0
                for keyPos in lowest...queryPos {
                    let slot = ring > 0 ? (keyPos % ring) : keyPos
                    if slot < ringLen {
                        ptr[slot &* strides[1] &+ query &* strides[3]] = 0
                    }
                }
            }
        }
        return ["sliding_causal_mask": mask]
    }
}

/// `rope_cos` / `rope_sin` — dual RoPE, precomputed on host and passed as inputs (lets
/// the model exceed the ANE 16-bit position-id limit). The per-dim theta vector is computed once;
/// per step we fill `cos(pos·θ)` / `sin(pos·θ)` for each block position.
///
/// Sliding sub-range `[0, slidingHeadDim)` = GPT-NeoX full rotary (freqs repeated in two halves).
/// Global sub-range = partial rotary: only the first `floor(partialRotaryFactor·globalHeadDim/2)`
/// frequency pairs rotate; the rest are NoPE (θ = 0 ⇒ cos = 1, sin = 0).
final class RoPEProvider: SyncInputHandler {
    private let theta: [Double]  // per-output-dim angular frequency; length = width
    var inputNames: [String] { ["rope_cos", "rope_sin"] }

    init(config: StaticBundleMetadata.RoPE?) {
        guard let config, config.slidingHeadDim > 0 || config.globalHeadDim > 0 else {
            self.theta = []  // identity fallback: cos=1, sin=0 (no rotation)
            return
        }
        let sHd = config.slidingHeadDim
        let gHd = config.globalHeadDim
        let width = sHd + gHd
        var theta = [Double](repeating: 0, count: width)

        // Sliding: NeoX full rotary. Half-dim freqs repeated across the two halves.
        let sHalf = sHd / 2
        for i in 0..<sHalf {
            let f = pow(config.slidingTheta, -Double(2 * i) / Double(sHd))
            theta[i] = f
            theta[i + sHalf] = f
        }
        // Global: partial rotary — first `rot` pairs rotate, rest NoPE (0).
        let gHalf = gHd / 2
        let rot = Int((config.partialRotaryFactor * Double(gHd) / 2.0).rounded(.down))
        for i in 0..<gHalf {
            let f = i < rot ? pow(config.globalTheta, -Double(2 * i) / Double(gHd)) : 0.0
            theta[sHd + i] = f
            theta[sHd + gHalf + i] = f
        }
        self.theta = theta
    }

    func prepare(_ ctx: InputContext) async throws -> [String: NDArray] {
        guard let cosDesc = ctx.descriptors["rope_cos"],
            let sinDesc = ctx.descriptors["rope_sin"]
        else { return [:] }
        var cos = NDArray(descriptor: cosDesc)
        var sin = NDArray(descriptor: sinDesc)
        fill(&cos, ctx: ctx, isSin: false)
        fill(&sin, ctx: ctx, isSin: true)
        return ["rope_cos": cos, "rope_sin": sin]
    }

    private func fill(_ array: inout NDArray, ctx: InputContext, isSin: Bool) {
        let theta = self.theta
        var view = array.mutableView(as: LogitsScalarType.self)
        view.withUnsafeMutablePointer { ptr, shape, strides in
            // shape: (1, q, width). Fill the FULL width; dims beyond `theta` (or when theta is
            // empty) get angle 0 → cos=1 / sin=0 (identity), so the buffer is never uninitialized.
            let q = shape[1]
            let width = shape[2]
            for query in 0..<q {
                let pos = Double(ctx.alignedStep + query)
                for d in 0..<width {
                    let angle = d < theta.count ? pos * theta[d] : 0.0
                    let v = isSin ? Foundation.sin(angle) : Foundation.cos(angle)
                    ptr[query &* strides[1] &+ d &* strides[2]] = LogitsScalarType(v)
                }
            }
        }
    }
}

/// `ple_embeddings` — per-layer embedding INT8 rows, gathered per token from an mmap'd sidecar.
final class PLEProvider: SyncInputHandler {
    private let table: PerLayerEmbeddings
    var inputNames: [String] { ["ple_embeddings"] }
    init(table: PerLayerEmbeddings) { self.table = table }
    func prepare(_ ctx: InputContext) async throws -> [String: NDArray] {
        guard let d = ctx.descriptors["ple_embeddings"] else { return [:] }
        var ple = NDArray(descriptor: d)
        try table.gather(tokenIDs: Array(ctx.tokens), batchSize: ctx.batchSize, into: &ple)
        return ["ple_embeddings": ple]
    }
}

/// mmap'd per-layer embedding table (`*_ple.safetensors`, INT8 `[vocab, rowWidth]`), with a
/// row-major gather. Bounds-checked so no token id can index past the mapping.
final class PerLayerEmbeddings {
    private let data: Data
    private let dataStart: Int
    let vocabSize: Int
    let rowWidth: Int

    init(bundleURL: URL) throws {
        // Locate the sidecar: `<something>_ple.safetensors` beside the .aimodel.
        let dirs = [bundleURL, bundleURL.deletingLastPathComponent()]
        var found: URL?
        for dir in dirs {
            let items =
                (try? FileManager.default.contentsOfDirectory(
                    at: dir,
                    includingPropertiesForKeys: nil)) ?? []
            if let m = items.first(where: { $0.lastPathComponent.hasSuffix("_ple.safetensors") }) {
                found = m
                break
            }
        }
        guard let url = found else {
            throw InferenceRuntimeError.invalidState("No *_ple.safetensors sidecar near \(bundleURL.path)")
        }
        let mapped = try Data(contentsOf: url, options: .mappedIfSafe)
        // safetensors: [8-byte LE header length][JSON header][raw bytes]
        guard mapped.count > 8 else { throw InferenceRuntimeError.invalidState("PLE file too small") }
        let headerLen = mapped.prefix(8).withUnsafeBytes { $0.load(as: UInt64.self) }
        let headerData = mapped.subdata(in: 8..<(8 + Int(headerLen)))
        guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any],
            let entry = header["embed_tokens_per_layer"] as? [String: Any],
            let shape = entry["shape"] as? [Int], shape.count == 2,
            let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
        else {
            throw InferenceRuntimeError.invalidState("PLE safetensors header missing embed_tokens_per_layer")
        }
        self.vocabSize = shape[0]
        self.rowWidth = shape[1]
        self.dataStart = 8 + Int(headerLen) + offsets[0]
        let expected = shape[0] * shape[1]  // INT8 → 1 byte each
        guard dataStart + expected <= mapped.count else {
            throw InferenceRuntimeError.invalidState("PLE data out of bounds")
        }
        self.data = mapped
    }

    /// Gather each token's INT8 row into `dst` (shape `(1, batchSize, 1, rowWidth)`), token-major.
    /// Out-of-range ids / padding slots are left zeroed.
    func gather(tokenIDs: [Int32], batchSize: Int, into dst: inout NDArray) throws {
        let rowWidth = self.rowWidth
        let dataStart = self.dataStart
        let vocab = self.vocabSize
        var view = dst.mutableView(as: Int8.self)
        view.withUnsafeMutablePointer { dstPtr, _, _ in
            for i in 0..<(batchSize * rowWidth) { dstPtr[i] = 0 }
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let base = raw.baseAddress!.advanced(by: dataStart).assumingMemoryBound(to: Int8.self)
                for i in 0..<min(batchSize, tokenIDs.count) {
                    let tok = Int(tokenIDs[i])
                    guard tok >= 0 && tok < vocab else { continue }
                    let src = base.advanced(by: tok * rowWidth)
                    dstPtr.advanced(by: i * rowWidth).update(from: src, count: rowWidth)
                }
            }
        }
    }
}
