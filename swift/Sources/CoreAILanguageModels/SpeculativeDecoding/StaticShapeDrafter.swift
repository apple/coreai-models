// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import Foundation

/// The EAGLE/MTP draft head, run as a static-shape Core AI program.
///
/// The layer stack -- everything between the token-embedding lookup and the masked embedder's
/// candidate scan -- runs as one static-shape `draft_step_<n>` call exported by
/// `models/gemma4/export_drafter.py`, the smallest whose full-attention window `n` holds the
/// context. The two lookups stay on the CPU: the token embedding comes from the target, and the
/// candidate scan reads the table the exporter writes with the program (``DrafterComponent``).
///
/// Single-owner: one task drives it at a time, and it holds no lock; the callers set the provider
/// closures before the first draft.
package final class StaticShapeDrafter: DrafterEngine, @unchecked Sendable {
    package let maxContextLength: Int
    /// The target's token embedding for a token id (`StaticShapeEngine.drafterEmbedding`).
    package var embeddingProvider: ((Int32) async throws -> [Float])?
    /// In-place access to the target's shared caches (`StaticShapeEngine.withSharedKVCache`): K/V
    /// goes straight from the target's state into the inputs. Drafting without it throws.
    package var sharedKVProvider:
        (
            (StaticShapeEngine.SharedKVCacheKind, (StaticShapeEngine.SharedKVCache) -> Void) throws -> Void
        )?

    /// What the head expects of the target, as `export_drafter.py` records it under `contract`.
    struct Contract: Decodable, Equatable {
        /// The decode graphs' final post-norm hidden state output.
        let hiddenStatesOutput: String?
        /// `unscaled`: `embed_tokens(token)` before the square-root-of-hidden scale the backbone applies.
        let tokenEmbedding: String?
        /// `last_storing_layer`: the shared K/V of the target's last layer that stores K/V.
        let kvSource: String?

        enum CodingKeys: String, CodingKey {
            case hiddenStatesOutput = "hidden_states_output"
            case tokenEmbedding = "token_embedding"
            case kvSource = "kv_source"
        }

        /// What this runtime provides.
        static let supported = Contract(
            hiddenStatesOutput: "hidden_states", tokenEmbedding: "unscaled", kvSource: "last_storing_layer")
    }

    struct Config: Decodable {
        /// Sidecar format version; absent in exports written before it existed, which read as 1.
        let version: Int?
        /// What the head expects of the target; absent in exports written before it existed, which
        /// were written for what ``Contract/supported`` provides.
        let contract: Contract?
        let fullWindows: [Int]?
        /// The one window of an export from before per-window draft steps.
        let window: Int?
        let slidingTheta: Float
        let fullTheta: Float
        let fullPartialRotaryFactor: Float

        enum CodingKeys: String, CodingKey {
            case version, window, contract
            case fullWindows = "full_windows"
            case slidingTheta = "sliding_theta"
            case fullTheta = "full_theta"
            case fullPartialRotaryFactor = "full_partial_rotary_factor"
        }
    }

    private let model: AIModel
    /// The exported full-attention windows, ascending; one `draft_step_<n>` function each.
    private let windows: [Int]
    /// An export from before per-window draft steps: one `draft_step` function at one window.
    private let legacy: Bool
    /// Only the function in use stays loaded, like `CoreAIStaticShapeEngine` with
    /// `EngineOptions.lowMemory`: keeping every window's program loaded halved e2b's throughput at 131072.
    private var functions: [Int: InferenceFunction] = [:]
    private var window = 0
    private var inputs = InputBuffers()
    /// The tokens whose full-attention K/V is already in place, one per leading position: the target
    /// only rewrites a confirmed position with the same values, since its K/V depends on the prefix alone.
    private var fullTokens = TokenHistory()
    /// The valid key slots each mask was last written for, so a round rewrites only the slots that changed.
    private var maskValid: [String: Int] = [:]
    private let slidingWindow: Int
    private let scan: DraftPack
    private let backboneHidden: Int
    private let topK: Int
    private var hidden: NDArray
    private var centroidLogits: NDArray
    private var x: NDArray
    /// Inverse RoPE frequencies, half a head wide; 0 where proportional RoPE leaves a pair unrotated.
    private let invFreqSliding: [Float]
    private let invFreqFull: [Float]
    private var seedHidden: [Float]?
    private var embedCache = EmbeddingCache()
    /// One centroid's candidate rows as fp32, and their scores.
    private var candidateRows: [Float] = []
    private var candidateScores: [Float] = []

    /// - Parameters:
    ///   - component: the exported program and its sidecars.
    ///   - vocabSize: the target's vocabulary; when given, the scan table must match it.
    ///   - topK: centroids whose candidates the scan scores; nil for ``DraftScan/defaultTopK``.
    package static func load(
        component: DrafterComponent, backboneHiddenSize: Int, maxContextLength: Int, vocabSize: Int? = nil,
        topK: Int? = nil
    ) async throws -> StaticShapeDrafter {
        let configURL = component.config
        let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: configURL))
        // The program is static-shape like the target's, so it is specialized the way the engine's are.
        let model = try await AIModel(
            contentsOf: component.step, options: ModelStructure.chunkedStatic(batchSize: 1).specializationOptions)
        try checkContract(config.contract, name: configURL.lastPathComponent)
        let resolved = try resolveWindows(
            config: config, name: configURL.lastPathComponent, functionNames: Set(model.functionNames))
        for warning in loadWarnings(
            legacy: resolved.legacy, windows: resolved.windows, configName: configURL.lastPathComponent,
            maxContextLength: maxContextLength)
        {
            CLILogger.log("Warning: StaticShapeDrafter: \(warning)")
        }
        return try StaticShapeDrafter(
            model: model, windows: resolved.windows, legacy: resolved.legacy, config: config,
            scan: DraftPack(blob: component.scan, index: component.scanIndex, role: .scan),
            backboneHidden: backboneHiddenSize, maxContextLength: maxContextLength, vocabSize: vocabSize,
            topK: topK)
    }

    /// Throws unless every expectation the export records is one this runtime provides. A field the
    /// export omits is not checked.
    static func checkContract(_ contract: Contract?, name: String) throws {
        guard let contract else { return }
        let provided = Contract.supported
        let remedy = "update the runner or re-export the drafter with a matching models/gemma4/export_drafter.py"
        func check(_ what: String, expected: String?, provided: String?) throws {
            guard let expected, expected != provided else { return }
            throw InferenceRuntimeError.invalidState(
                "\(name) expects \(what) '\(expected)' from the target, but this runner provides "
                    + "'\(provided ?? "none")'; \(remedy)")
        }
        try check(
            "the hidden-state output", expected: contract.hiddenStatesOutput, provided: provided.hiddenStatesOutput)
        try check("a token embedding that is", expected: contract.tokenEmbedding, provided: provided.tokenEmbedding)
        try check("shared K/V from", expected: contract.kvSource, provided: provided.kvSource)
    }

    /// The full-attention windows to serve, ascending, and whether the export is from before
    /// per-window draft steps (one `draft_step` function at one window). Throws unless the sidecar
    /// version is readable, the windows are positive, ascending and unique, and the program holds
    /// a function for every one of them.
    static func resolveWindows(
        config: Config, name: String, functionNames: Set<String>
    ) throws -> (windows: [Int], legacy: Bool) {
        let reexport = "re-export the drafter with models/gemma4/export_drafter.py"
        if let version = config.version, version != 1 {
            throw InferenceRuntimeError.invalidState(
                version > 1
                    ? "\(name) has format version \(version), newer than the 1 this runner reads; update the "
                        + "runner or \(reexport) with a matching exporter"
                    : "\(name) has invalid format version \(version); \(reexport)")
        }
        // An export from before per-window draft steps still drafts, with full attention over its one window.
        let legacy = config.fullWindows == nil
        guard let windows = config.fullWindows ?? config.window.map({ [$0] }), !windows.isEmpty else {
            throw InferenceRuntimeError.invalidState(
                "\(name) lists neither full_windows nor window; \(reexport)")
        }
        guard windows.first! > 0, zip(windows, windows.dropFirst()).allSatisfy({ $0 < $1 }) else {
            throw InferenceRuntimeError.invalidState(
                "\(name) full_windows \(windows) must be positive, ascending and unique; \(reexport)")
        }
        for window in windows {
            let function = functionName(window: window, legacy: legacy)
            guard functionNames.contains(function) else {
                throw InferenceRuntimeError.invalidState(
                    "\(name) lists window \(window) but the drafter .aimodel has no '\(function)' function; "
                        + reexport)
            }
        }
        return (windows, legacy)
    }

    /// The smallest exported full-attention window that holds `kvLen` positions, or the largest
    /// when none does. `windows` is ascending.
    static func window(for kvLen: Int, in windows: [Int]) -> Int {
        windows.first { $0 >= kvLen } ?? windows[windows.count - 1]
    }

    /// Which positions of a shared cache fill a `width`-slot input: the most recent `width` of
    /// `kvLen`, read from a cache of `ctx` slots. A ring cache holds position `p` at slot
    /// `p % ctx`, so the run wraps once; a flat cache is the identity.
    struct KVWindow {
        let count: Int
        let first: Int
        private let start: Int
        private let run: Int

        init(kvLen: Int, width: Int, ctx: Int, ring: Bool) {
            count = min(kvLen, width)
            first = kvLen - count
            start = ring ? first % ctx : first
            run = ring ? min(count, ctx - start) : count
        }

        /// Cache slot holding the window's `t`-th position (absolute position `first + t`).
        func slot(_ t: Int) -> Int { t < run ? start + t : t - run }
        /// Positions before `run` come from `start` onward; the rest wrap to slot 0.
        var contiguousRun: Int { run }
    }

    /// Messages for a drafter that works but sees less than the target's context.
    static func loadWarnings(
        legacy: Bool, windows: [Int], configName: String, maxContextLength: Int
    ) -> [String] {
        var warnings: [String] = []
        if legacy {
            warnings.append(
                "\(configName) predates per-window draft steps, so full attention sees only the most "
                    + "recent \(windows[0]) positions. Re-export it for the whole context.")
        }
        if let largest = windows.last, largest < maxContextLength {
            warnings.append(
                "full attention covers \(largest) of the target's \(maxContextLength) "
                    + "positions; past that it sees only the most recent \(largest). Re-export with "
                    + "--max-context-length \(maxContextLength).")
        }
        return warnings
    }

    /// Checks the exported program and scan table against the target they will draft for.
    /// Widths come from the program's descriptors; `nil` means the program does not declare one.
    static func checkGeometry(
        scan: DraftPack.Index, scanHiddenWidth: Int, backboneHidden: Int, vocabSize: Int?,
        inputsEmbedsWidth: Int?, hiddenWidth: Int?, centroidLogitsWidth: Int?, stateWidth: Int?
    ) throws {
        let reexport = "re-export the drafter for this target with models/gemma4/export_drafter.py"
        func fail(_ what: String) -> InferenceRuntimeError {
            .invalidState("the drafter does not match the target: \(what); \(reexport)")
        }
        guard let inputsEmbedsWidth, let hiddenWidth, let centroidLogitsWidth, let stateWidth else {
            throw fail("the program does not declare inputs_embeds, hidden, centroid_logits and x")
        }
        guard inputsEmbedsWidth == 2 * backboneHidden, hiddenWidth == backboneHidden else {
            throw fail(
                "it was exported for a backbone width of \(hiddenWidth) (inputs_embeds \(inputsEmbedsWidth)) "
                    + "but the target's is \(backboneHidden)")
        }
        if let vocabSize, scan.vocab != vocabSize {
            throw fail("its scan table covers \(scan.vocab) tokens but the target has \(vocabSize)")
        }
        guard centroidLogitsWidth == scan.centroids else {
            throw fail(
                "the program scores \(centroidLogitsWidth) centroids but the scan table indexes \(scan.centroids)")
        }
        guard scanHiddenWidth == stateWidth else {
            throw fail("the scan table rows are \(scanHiddenWidth) wide but the program's state is \(stateWidth)")
        }
    }

    private init(
        model: AIModel, windows: [Int], legacy: Bool, config: Config, scan: DraftPack, backboneHidden: Int,
        maxContextLength: Int, vocabSize: Int?, topK: Int?
    ) throws {
        self.model = model
        self.windows = windows
        self.legacy = legacy
        let fn = try Self.loadFunction(model, named: Self.functionName(window: windows[0], legacy: legacy))
        functions[windows[0]] = fn
        let desc = fn.descriptor
        guard case .ndArray(let kSliding)? = desc.inputDescriptor(of: "k_sliding"), let sw = kSliding.shape.last
        else { throw InferenceRuntimeError.invalidState("draft_step has no NDArray 'k_sliding'") }
        slidingWindow = sw
        hidden = try Self.array(desc.outputDescriptor(of: "hidden"), "hidden")
        centroidLogits = try Self.array(desc.outputDescriptor(of: "centroid_logits"), "centroid_logits")
        x = try Self.array(desc.outputDescriptor(of: "x"), "x")
        func invFreq(_ name: String, theta: Float, partial: Float) -> [Float] {
            guard case .ndArray(let d)? = desc.inputDescriptor(of: name), let hd = d.shape.last else { return [] }
            let rotated = Int(Float(hd / 2) * partial)
            return (0..<hd / 2).map { $0 < rotated ? powf(theta, -2 * Float($0) / Float(hd)) : 0 }
        }
        func lastDimension(_ name: String, output: Bool) -> Int? {
            let d = output ? desc.outputDescriptor(of: name) : desc.inputDescriptor(of: name)
            if case .ndArray(let nd)? = d { return nd.shape.last }
            return nil
        }
        try Self.checkGeometry(
            scan: scan.index, scanHiddenWidth: scan.shape("embed_perm_f16").last ?? 0,
            backboneHidden: backboneHidden, vocabSize: vocabSize,
            inputsEmbedsWidth: lastDimension("inputs_embeds", output: false),
            hiddenWidth: lastDimension("hidden", output: true),
            centroidLogitsWidth: lastDimension("centroid_logits", output: true),
            stateWidth: lastDimension("x", output: true))
        invFreqSliding = invFreq("cos_sliding", theta: config.slidingTheta, partial: 1)
        invFreqFull = invFreq("cos_full", theta: config.fullTheta, partial: config.fullPartialRotaryFactor)
        self.scan = scan
        self.backboneHidden = backboneHidden
        self.topK = try DraftScan.topK(requested: topK, centroids: scan.index.centroids)
        self.maxContextLength = maxContextLength
        try select(window: windows[0])
    }

    private static func array(_ d: InferenceValue.Descriptor?, _ name: String) throws -> NDArray {
        guard case .ndArray(let nd)? = d else {
            throw InferenceRuntimeError.invalidState("draft_step has no NDArray '\(name)'")
        }
        return NDArray(descriptor: nd)
    }

    private static func functionName(window: Int, legacy: Bool) -> String {
        legacy ? "draft_step" : "draft_step_\(window)"
    }

    private static func loadFunction(_ model: AIModel, named name: String) throws -> InferenceFunction {
        guard let fn = try model.loadFunction(named: name) else {
            throw InferenceRuntimeError.invalidState("drafter .aimodel has no '\(name)' function")
        }
        return fn
    }

    private func function(for window: Int) throws -> InferenceFunction {
        if let fn = functions[window] { return fn }
        let fn = try Self.loadFunction(model, named: Self.functionName(window: window, legacy: legacy))
        functions = [window: fn]
        return fn
    }

    /// Bind `window`'s function and size the inputs for it. Only the full-attention K/V and mask
    /// change shape between windows; `InputBuffers` keeps every other input in place.
    private func select(window: Int) throws {
        guard window != self.window else { return }
        let desc = try function(for: window).descriptor
        let missing = Set(Self.writtenInputs).subtracting(desc.inputNames)
        guard missing.isEmpty else {
            throw InferenceRuntimeError.invalidState(
                "\(Self.functionName(window: window, legacy: legacy)) has no input(s) \(missing.sorted()); "
                    + "re-export it with "
                    + "models/gemma4/export_drafter.py")
        }
        for name in desc.inputNames {
            guard case .ndArray(let d)? = desc.inputDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidState("draft_step has no NDArray '\(name)'")
            }
            // Drop the old width before `ensureCapacity` can pool it: pooled, every window a
            // context grows through stays allocated (2 KiB per position on e2b).
            if inputs[name]?.shape != d.shape { inputs.removeBuffer(name: name) }
            inputs.ensureCapacity(name: name, descriptor: d)
        }
        self.window = window
        fullTokens.clear()
        maskValid = [:]
    }

    /// Must be called whenever the confirmed prefix diverges, including between generations. Text is
    /// also caught by `fullTokens`, but image positions share one placeholder id whatever the image.
    package func reset(to tokenIndex: Int) async throws {
        seedHidden = nil
        fullTokens.truncate(to: tokenIndex)
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
                "StaticShapeDrafter has no shared K/V provider: connect it to the target before drafting")
        }
        // Before the first verify there is no hidden state to seed from, so the round proposes the
        // anchor, which the target rejects.
        guard var lastHidden = seedHidden else {
            let anchor = confirmedPrefix.last ?? 0
            return DraftProposal(tokens: Array(repeating: anchor, count: count))
        }
        let kvLen = max(1, confirmedPrefix.count - 1)
        try select(window: Self.window(for: kvLen, in: windows))
        try writeKV(caches, confirmedPrefix: confirmedPrefix, kvLen: kvLen)

        var lastToken = confirmedPrefix.last ?? 0
        let basePosition = confirmedPrefix.count - 1
        var tokens: [Int32] = []
        for i in 0..<count {
            var fused = try await embedding(for: lastToken)
            fused.append(contentsOf: lastHidden)
            guard fused.count == 2 * backboneHidden else {
                throw InferenceRuntimeError.invalidState(
                    "the target supplied \(fused.count) values for the drafter's \(2 * backboneHidden)-wide "
                        + "inputs_embeds; the drafter was exported for a different target")
            }
            write("inputs_embeds", fused)
            writeRope(pos: basePosition + i)

            var outputs = InferenceFunction.MutableViews()
            outputs.insert(&hidden, for: "hidden")
            outputs.insert(&centroidLogits, for: "centroid_logits")
            outputs.insert(&x, for: "x")
            _ = try await function(for: window).run(inputs: inputs.borrowedInputs(), outputViews: consume outputs)

            let token = scan(x: Self.read(&x), centroidLogits: Self.read(&centroidLogits))
            tokens.append(token)
            lastToken = token
            lastHidden = Self.read(&hidden)
        }
        return DraftProposal(tokens: tokens)
    }

    /// The masked embedder: score only the tokens of the `topK` best centroids and take the best.
    private func scan(x: [Float], centroidLogits cl: [Float]) -> Int32 {
        let perCentroid = scan.index.perCentroid
        let hidden = scan.shape("embed_perm_f16")[1]
        var topV = [Float](repeating: -.greatestFiniteMagnitude, count: topK)
        var topI = [Int](repeating: 0, count: topK)
        for c in 0..<scan.index.centroids where cl[c] > topV[topK - 1] {
            var j = topK - 1
            while j > 0 && cl[c] > topV[j - 1] {
                topV[j] = topV[j - 1]
                topI[j] = topI[j - 1]
                j -= 1
            }
            topV[j] = cl[c]
            topI[j] = c
        }
        // A centroid's candidates are contiguous rows, so each is one fp16 -> fp32 conversion and one
        // matrix-vector product. Scoring row by row through `DraftScan.dot` spent most of the scan
        // in the runtime availability check `SIMD8<Float>(SIMD8<Float16>)` makes on every call.
        let table = scan.f16("embed_perm_f16")
        let block = perCentroid * hidden
        if candidateRows.count != block { candidateRows = [Float](repeating: 0, count: block) }
        if candidateScores.count != perCentroid { candidateScores = [Float](repeating: 0, count: perCentroid) }
        var best = -Float.greatestFiniteMagnitude
        var bestSlot = 0
        x.withUnsafeBufferPointer { xp in
            candidateRows.withUnsafeMutableBufferPointer { rows in
                candidateScores.withUnsafeMutableBufferPointer { scores in
                    for c in topI {
                        var src = vImage_Buffer(
                            data: UnsafeMutableRawPointer(mutating: table + c * block), height: 1,
                            width: vImagePixelCount(block), rowBytes: block * 2)
                        var dst = vImage_Buffer(
                            data: rows.baseAddress, height: 1, width: vImagePixelCount(block), rowBytes: block * 4)
                        guard vImageConvert_Planar16FtoPlanarF(&src, &dst, 0) == kvImageNoError else {
                            // Unconverted rows would score stale data, so this centroid is scored row by row.
                            for j in 0..<perCentroid {
                                let slot = c * perCentroid + j
                                let score = DraftScan.dot(x, table + slot * hidden, n: hidden)
                                if score > best {
                                    best = score
                                    bestSlot = slot
                                }
                            }
                            continue
                        }
                        vDSP_mmul(
                            rows.baseAddress!, 1, xp.baseAddress!, 1, scores.baseAddress!, 1,
                            vDSP_Length(perCentroid), 1, vDSP_Length(hidden))
                        var score: Float = 0
                        var j: vDSP_Length = 0
                        vDSP_maxvi(scores.baseAddress!, 1, &score, &j, vDSP_Length(perCentroid))
                        if score > best {
                            best = score
                            bestSlot = c * perCentroid + Int(j)
                        }
                    }
                }
            }
        }
        return scan.i32("token_ordering_i32")[bestSlot]
    }

    // MARK: - Inputs

    /// Every input the drafter writes, which each `draft_step_<n>` must declare.
    private static let writtenInputs = [
        "inputs_embeds", "cos_sliding", "sin_sliding", "cos_full", "sin_full",
        "k_sliding", "v_sliding", "mask_sliding", "k_full", "v_full", "mask_full",
    ]

    /// Mutate one input in place; `withMutableBuffer` keeps the write from copying the array.
    private func withInput(_ name: String, _ body: (UnsafeMutablePointer<Float16>, [Int], [Int]) -> Void) {
        // Cannot throw: `select(window:)` has checked the bound function declares every name written.
        try! inputs.withMutableBuffer(name) { array in
            array.mutableView(as: Float16.self).withUnsafeMutablePointer { p, shape, strides in
                body(p, (0..<shape.count).map { shape[$0] }, (0..<strides.count).map { strides[$0] })
            }
        }
    }

    private func write(_ name: String, _ values: [Float]) {
        withInput(name) { p, _, _ in
            for i in 0..<values.count { p[i] = Float16(values[i]) }
        }
    }

    /// The rotary tables for `pos`, in the rotate-half layout: both halves repeat the same angles.
    private func writeRope(pos: Int) {
        func tables(_ cosName: String, _ sinName: String, _ inv: [Float]) {
            let half = inv.count
            let angles = inv.map { Float(pos) * $0 }
            // One table at a time: both live in `inputs`, which a nested write would access twice.
            withInput(cosName) { c, _, _ in
                for i in 0..<half {
                    c[i] = Float16(cosf(angles[i]))
                    c[i + half] = c[i]
                }
            }
            withInput(sinName) { s, _, _ in
                for i in 0..<half {
                    s[i] = Float16(sinf(angles[i]))
                    s[i + half] = s[i]
                }
            }
        }
        tables("cos_sliding", "sin_sliding", invFreqSliding)
        tables("cos_full", "sin_full", invFreqFull)
    }

    /// The target's shared K/V into the `(1, kv_heads, head_dim, w)` inputs, each over the most recent
    /// `w` positions it holds, read from the target's caches in place: per channel, at most two runs
    /// of the window (the sliding ring wraps once), with no transpose. While the context fits the
    /// full-attention window, only positions past the prefix already in place are copied into it.
    private func writeKV(
        _ caches: (StaticShapeEngine.SharedKVCacheKind, (StaticShapeEngine.SharedKVCache) -> Void) throws -> Void,
        confirmedPrefix: [Int32], kvLen: Int
    ) throws {
        let valid = kvLen <= window ? min(fullTokens.resolve(input: confirmedPrefix).commonPrefix, kvLen) : 0
        let bars: [(StaticShapeEngine.SharedKVCacheKind, String, Bool)] = [
            (.slidingK, "k_sliding", true), (.slidingV, "v_sliding", true),
            (.fullK, "k_full", false), (.fullV, "v_full", false),
        ]
        var ringDepth = Int.max
        var layoutError: String?
        for (kind, name, ring) in bars {
            try caches(kind) { c in
                withInput(name) { p, shape, strides in
                    let hd = shape[2]
                    let w = shape[3]
                    let f = c.factor
                    if ring { ringDepth = min(ringDepth, c.ctx) }
                    guard c.channels == shape[1] * hd else {
                        layoutError = "\(name) holds \(shape[1] * hd) channels but the target's cache has \(c.channels)"
                        return
                    }
                    guard !ring || w <= c.ctx else { return }
                    let span = KVWindow(kvLen: kvLen, width: w, ctx: c.ctx, ring: ring)
                    let n = span.count
                    let first = span.first
                    let run = span.contiguousRun
                    // A sliding window, ring or full past its width, shifts every round.
                    let from = ring || first > 0 ? 0 : min(valid, n)
                    for ch in 0..<c.channels {
                        let src = c.base + (ch / f) * c.ctx * f + ch % f
                        let row = p + (ch / hd) * strides[1] + (ch % hd) * strides[2]
                        for t in from..<run { row[t] = Float16(src[span.slot(t) * f]) }
                        for t in max(run, from)..<n { row[t] = Float16(src[span.slot(t) * f]) }
                        if n < w, from == 0 { (row + n).update(repeating: 0, count: w - n) }
                    }
                }
            }
        }
        if let layoutError {
            throw InferenceRuntimeError.invalidState(
                "the drafter's K/V layout does not match the target: \(layoutError); "
                    + "it was exported for a different model")
        }
        // A window past the ring would read slots the target has already overwritten.
        guard slidingWindow <= ringDepth else {
            throw InferenceRuntimeError.invalidState(
                "drafter window \(slidingWindow) exceeds the target's sliding ring of \(ringDepth)")
        }
        if kvLen <= window {
            fullTokens.truncate(to: valid)
            fullTokens.append(contentsOf: confirmedPrefix[valid..<kvLen])
        } else {
            fullTokens.clear()
        }
        writeMasks(kvLen: kvLen)
    }

    /// Empty key slots still score and take softmax weight, so they are masked. Finite rather than
    /// -inf: the program's softmax mishandles infinities. Only the slots between the last
    /// round's valid count and this one's change, so only those are written: at the largest windows a
    /// mask is over 100K slots.
    private func writeMasks(kvLen: Int) {
        for name in ["mask_sliding", "mask_full"] {
            let previous = maskValid[name]
            var valid = 0
            withInput(name) { p, shape, _ in
                let w = shape[3]
                valid = min(kvLen, w)
                let changed = previous.map { min($0, valid)..<max($0, valid) } ?? 0..<w
                for t in changed { p[t] = t < valid ? 0 : -40000 }
            }
            maskValid[name] = valid
        }
    }

    private static func read(_ a: inout NDArray) -> [Float] {
        var out: [Float] = []
        a.mutableView(as: Float16.self).withUnsafeMutablePointer { p, shape, _ in
            let n = shape[shape.count - 1]
            out = (0..<n).map { Float(p[$0]) }
        }
        return out
    }

    private func embedding(for token: Int32) async throws -> [Float] {
        if let c = embedCache[token] { return c }
        guard let provider = embeddingProvider else {
            return [Float](repeating: 0, count: backboneHidden)
        }
        let row = try await provider(token)
        embedCache.insert(row, for: token)
        return row
    }
}
