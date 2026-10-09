// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - Prefill Strategy

/// Determines the optimal prefill strategy based on prompt size.
enum PrefillStrategy {
    case chunked(chunkSize: Int)
    case wholeBatch
    case oneAtATime
}

// MARK: - Core AI Sequential Clean Engine

/// Clean Core AI inference engine built from scratch using only public APIs.
///
/// ## Model Contract
///
/// Expects a `.aimodel` with:
/// - **2 inputs**: `input_ids` (Int32), `position_ids` (Int32)
/// - **1 output**: `logits` (LogitsScalarType)
/// - **2–4 states**: KV cache pair + optional persistent states (hybrid models), updated in-place
///
/// KV cache NDArrays start small (256 tokens) and grow dynamically with 2× expansion.
/// Passed as `states` on every forward pass; the model graph updates them in-place.
public final class CoreAISequentialEngine: InferenceEngine, IdempotentEngine, BatchedDecodeStep, @unchecked Sendable {
    public typealias ConfigType = ModelConfig

    public var supportsLogits: Bool { true }
    public var vocabSize: Int { config.vocabSize }
    public var hasRecurrentState: Bool { session.hasNonTruncatableStates }
    public let config: ModelConfig

    // Core AI function handle
    private let function: InferenceFunction
    private let functionDescriptor: InferenceFunctionDescriptor

    // Optional prefill graph. Prefill chunks run here when the asset has it. It produces
    // no logits, so the last prompt token still goes through `function`.
    private let prefillFunction: InferenceFunction?

    // I/O names from descriptor
    private let logitsName: String

    // Input handling — handler owns allocation and fill logic. Composed so an optional
    // additive `attn_mask` input (opt-in `use_attention_mask` export) rides alongside the
    // standard token/position inputs; `extras` is empty for the common 2-input contract.
    // Built FRESH per session (via this factory, stored on `GenerationSessionState.inputHandler`)
    // so independent sessions never share the handler's internal reusable buffers.
    private let makeInputHandler: () throws -> CompositeInputHandler<TokenInputHandler>

    /// Retained to build per-session state handlers in `makeSessionState()`.
    private let options: EngineOptions

    // Per-request mutable state. The `generate()` shim owns one internal session and
    // reuses it across calls, preserving implicit prefix-cache reuse and reset(to:)
    // semantics. Idempotent callers pass their own session instead.
    private let session: GenerationSessionState

    // Logits descriptor. The output buffer is allocated per call (call-local) in
    // `processTokenBatch`, so no logits scratch lives on the engine — independent sessions don't share it.
    private let logitsDescriptor: NDArrayDescriptor

    // Ring buffer mode: handled by TokenInputHandler.useCompactPositionIds

    // Track processed tokens for incremental inference (delegated to the shim session).
    public var processedTokenCount: Int { session.processedTokenCount }

    /// The request-batch size the loaded graph is built for: the pinned leading dim of `input_ids`
    /// (1 for the usual single-batch graph, N for a fixed `batch=N` graph). Auto-detected from the
    /// descriptor so tools can report it without a flag.
    public var declaredBatchSize: Int {
        guard let name = InputLayout.knownInputIdNames.first(where: functionDescriptor.inputNames.contains),
            case .ndArray(let desc) = functionDescriptor.inputDescriptor(of: name),
            let dim0 = desc.shape.first, dim0 > 0
        else { return 1 }
        return dim0
    }

    public var maxBatchCapacity: Int {
        guard let name = InputLayout.knownInputIdNames.first(where: functionDescriptor.inputNames.contains),
            case .ndArray(let desc) = functionDescriptor.inputDescriptor(of: name),
            let dim0 = desc.shape.first
        else { return 1 }
        if dim0 > 1 { return dim0 }  // static batch=N bucket
        if dim0 < 0 { return Int.max }  // dynamic batch dim — bounded by the traced max, enforced at run
        return 1  // pinned single-batch graph
    }

    public var supportsBatching: Bool {
        let hasMask = InputLayout.knownAttnMaskNames.first(where: functionDescriptor.inputNames.contains) != nil
        return hasMask && maxBatchCapacity > 1
    }

    /// Mirror of the most-recently-driven session's prefix hit count, exposed for the
    /// single-valued `InferenceEngine.lastPrefixHitCount` protocol getter. The authoritative
    /// per-session value lives on `GenerationSessionState.lastPrefixHitCount`; a concurrent
    /// scheduler reads that per session.
    public private(set) var lastPrefixHitCount: Int = 0

    /// In-flight generation is tracked per-session via `GenerationSessionState.tokenBox`, so
    /// independent sessions never cancel each other. `isBusy` reports the engine's default
    /// (single-session) state.
    public var isBusy: Bool { session.tokenBox.isBusy }

    // MARK: - Init

    init(
        config: ModelConfig,
        preparedModel: PreparedModel,
        options: EngineOptions = EngineOptions()
    ) async throws {
        self.config = config

        let modelLoadSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAICleanModelLoading",
            details: "Loading \(config.name) from prepared asset"
        )

        let model = preparedModel.model

        // Get function descriptor
        guard let descriptor = model.functionDescriptor(for: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot find function '\(config.function)' in model")
        }
        self.functionDescriptor = descriptor

        // Validate model architecture: 2 inputs (input_ids, position_ids), plus an optional
        // 3rd additive `attn_mask` input; 1+ output; at least the KV cache pair.
        // Hybrid models may declare additional persistent fixed-shape states.
        let hasAttnMask = descriptor.inputNames.contains(where: InputLayout.knownAttnMaskNames.contains)
        guard descriptor.inputNames.count == 2 || (descriptor.inputNames.count == 3 && hasAttnMask)
        else {
            throw InferenceRuntimeError.invalidInputType(
                "Expected 2 inputs (or 3 with attn_mask), got \(descriptor.inputNames.count): "
                    + "\(descriptor.inputNames)")
        }
        guard descriptor.outputNames.count >= 1 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 1 output, got \(descriptor.outputNames.count): \(descriptor.outputNames)")
        }
        guard descriptor.stateNames.count >= 2 && descriptor.stateNames.count <= 4 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected 2–4 states (KV cache + optional persistent states), got \(descriptor.stateNames.count): "
                    + "states=\(descriptor.stateNames), outputs=\(descriptor.outputNames)")
        }

        // Create state handlers from descriptor
        let stateHandlers = try StateHandlerFactory.createSyncHandlers(
            descriptor: descriptor,
            maxContextLength: config.maxContextLength,
            options: options
        )
        self.options = options
        self.session = GenerationSessionState(
            kvCache: stateHandlers.kvCache,
            additionalStates: stateHandlers.additionalStates,
            hasNonTruncatableStates: stateHandlers.hasNonTruncatableStates
        )

        let layout = try InputLayout.analyze(
            model: model, functionName: config.function, config: config,
            useCompactPositionIds: stateHandlers.isAllSlidingCache)

        self.logitsName = layout.logitsName
        let inputIdsName = layout.inputIdsName
        let positionIdsName = layout.positionIdsName

        // Create input handler from descriptors
        guard case .ndArray(let inputIdsDesc) = descriptor.inputDescriptor(of: inputIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(inputIdsName)'")
        }
        guard case .ndArray(let posIdsDesc) = descriptor.inputDescriptor(of: positionIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(positionIdsName)'")
        }

        guard case .ndArray(let logitsDesc) = descriptor.outputDescriptor(of: logitsName) else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get descriptor for '\(logitsName)'")
        }
        guard logitsDesc.scalarType == .float16 else {
            throw InferenceRuntimeError.unsupportedLogitsType(
                "Only float16 logits supported, got \(logitsDesc.scalarType)")
        }
        self.logitsDescriptor = logitsDesc

        // Factory: build a FRESH input handler on demand (one per session). Captures the immutable
        // descriptors/names/policy; each call allocates its own reusable buffers so concurrent
        // sessions never share the handler's internal scratch. When the model declares an additive
        // `attn_mask` input, a per-step causal mask rides alongside the token/position inputs (batch=1
        // reproduces the implicit is_causal path; the batched/ragged decoder varies it per row).
        let compactPositions = layout.positionPolicy == .compact
        let attnMaskName = layout.attnMaskName
        self.makeInputHandler = {
            var maskExtras: [CompositeInputHandler<TokenInputHandler>.ExtraInput] = []
            if let maskName = attnMaskName {
                guard case .ndArray(let maskDesc) = descriptor.inputDescriptor(of: maskName) else {
                    throw InferenceRuntimeError.invalidInputType(
                        "Cannot get descriptor for '\(maskName)'")
                }
                maskExtras.append(
                    .init(name: maskName) { context in
                        makeAdditiveCausalMask(context: context, descriptor: maskDesc)
                    })
            }
            let tokenHandler = TokenInputHandler(
                inputIdsName: inputIdsName,
                positionIdsName: positionIdsName,
                inputIdsDescriptor: inputIdsDesc,
                positionIdsDescriptor: posIdsDesc,
                useCompactPositionIds: compactPositions
            )
            return CompositeInputHandler(base: tokenHandler, extras: maskExtras)
        }

        CLILogger.log(
            "KV cache: capacity=\(stateHandlers.kvCache.currentCapacity), states=\(stateHandlers.kvCache.stateNames)"
        )
        if let additional = stateHandlers.additionalStates {
            CLILogger.log(
                "Additional persistent states: \(additional.stateNames.joined(separator: ", "))")
        }

        // Load inference function
        self.prefillFunction = try loadPrefillGraph(
            from: model, matching: descriptor, mainName: config.function)
        if self.prefillFunction != nil {
            CLILogger.log("Found '\(prefillGraphFunctionName)' graph — prefill skips the LM head")
        }

        guard let fn = try model.loadFunction(named: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot load function '\(config.function)'")
        }
        self.function = fn

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAICleanModelLoading",
            signpostID: modelLoadSignpost
        )

        CLILogger.log(
            "CoreAI clean engine initialized — inputs: \(descriptor.inputNames), outputs: \(descriptor.outputNames), states: \(descriptor.stateNames)"
        )

        // The engine's own default session (driven by the single-path `generate` shim) gets its own
        // input handler, same as every session minted by `makeSessionState`.
        self.session.inputHandler = try makeInputHandler()
    }

    /// Convenience initializer with direct model URL.
    public convenience init(
        config: ModelConfig,
        modelURL: URL,
        options: EngineOptions = EngineOptions()
    ) async throws {
        CLILogger.log("Loading CoreAI model asset from: \(modelURL.lastPathComponent)")
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        try await self.init(config: config, preparedModel: preparedModel, options: options)
    }

    // MARK: - Prefill Strategy

    private func selectPrefillStrategy(newTokenCount: Int) -> PrefillStrategy {
        // With a prefill graph, chunking is cheaper at any size: every chunk but the last
        // token skips the LM head, so there is no threshold to clear.
        if shouldChunkPrefill(
            tokenCount: newTokenCount,
            hasPrefillGraph: prefillFunction != nil,
            chunkThreshold: config.chunkThreshold)
        {
            return .chunked(chunkSize: config.prefillChunkSize)
        }
        return .wholeBatch
    }

    // MARK: - Token Batch Processing

    /// Process a batch of tokens in a single forward pass.
    private func processTokenBatch(
        _ tokens: ArraySlice<Int32>, sessionState: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        let batchSize = tokens.count
        guard batchSize > 0 else {
            throw InferenceRuntimeError.invalidState("Cannot process empty token batch")
        }

        _ = try sessionState.kvCache.ensureCapacity(
            forContextLength: sessionState.processedTokenCount + batchSize)

        let batchSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIClean Batch",
            details: "\(batchSize) tokens at position \(sessionState.processedTokenCount)"
        )

        let context = InputContext.dynamic(
            tokens: tokens, processedTokenCount: sessionState.processedTokenCount)
        guard var handler = sessionState.inputHandler else {
            throw InferenceRuntimeError.invalidState("session has no input handler")
        }
        let inputs = try await handler.prepare(context)
        sessionState.inputHandler = handler  // retain the handler's mutated reusable buffers

        // Reuse THIS session's logits buffer across its decode steps (perf), reallocating only when
        // the batch size changes. Per-session (on `sessionState`, not the engine), so independent
        // sessions never share the output buffer — the single lane can run beside the batched lane.
        if sessionState.logitsScratchBatchSize != batchSize || sessionState.logitsScratch == nil {
            let resolvedLogitsDesc = logitsDescriptor.resolvingDynamicDimensions([1, batchSize, config.vocabSize])
            sessionState.logitsScratch = NDArray(descriptor: resolvedLogitsDesc)
            sessionState.logitsScratchBatchSize = batchSize
        }
        var logitsArray = sessionState.logitsScratch!

        // Bind states, build output views, and execute
        try await runWithStates(
            function: function,
            inputs: inputs,
            primary: sessionState.kvCache,
            secondary: sessionState.additionalStates,
            outputArray: &logitsArray,
            outputName: logitsName
        )
        sessionState.logitsScratch = logitsArray  // retain the (same-storage) handle for reuse

        // Read logits from NDArray
        let totalLogits = batchSize * config.vocabSize
        let logitBuffer = readNDArray(logitsArray, as: LogitsScalarType.self, count: totalLogits)

        sessionState.processedTokenCount += batchSize

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIClean Batch",
            signpostID: batchSignpost
        )

        return logitBuffer
    }

    // MARK: - Lockstep batched decode (benchmark / smoke driver)

    /// Result of a batched decode: per-row generated tokens plus prefill/decode timing (for
    /// throughput reporting) and the batch size the graph was built for.
    public struct BatchedRunResult: Sendable {
        public let tokens: [[Int32]]
        public let batchSize: Int
        public let promptLen: Int
        public let prefillSeconds: Double
        public let decodeSeconds: Double
    }

    /// Decode `N` equal-length rows together through a fixed-batch (`batch=N`) graph, each row with
    /// its own ``SamplingConfiguration`` (e.g. row 0 greedy, row 1 temperature). Used by the
    /// benchmark/runner to exercise the batched graph + per-row ``BatchedSampler`` end-to-end on GPU.
    ///
    /// Constraints: lockstep (equal-length rows, one shared cache offset), whole-batch prefill, and
    /// finished rows are held (fed their last token, output ignored) until all drain. Pass an empty
    /// `eosTokenIds` to always generate `maxNewTokens` (stable throughput for benchmarking). A
    /// continuous/ragged serving path with elastic admission builds on `decodeBatch`/`prefillRow`.
    package func lockstepBatchedGenerate(
        promptRows: [[Int32]],
        configs: [SamplingConfiguration],
        maxNewTokens: Int,
        eosTokenIds: Set<Int32>
    ) async throws -> BatchedRunResult {
        precondition(!promptRows.isEmpty && promptRows.count == configs.count)
        let n = promptRows.count
        let promptLen = promptRows[0].count
        precondition(
            promptRows.allSatisfy { $0.count == promptLen }, "lockstep requires equal-length rows")

        let batchSession = try makeSessionState(batchSize: n)
        _ = try batchSession.kvCache.ensureCapacity(forContextLength: promptLen + maxNewTokens)

        var gen = Array(repeating: [Int32](), count: n)
        var live = Array(repeating: true, count: n)

        let prefillStart = SuspendingClock.now
        var perRowLast = try await decodeBatch(
            tokensPerRow: promptRows,
            startPositions: Array(repeating: 0, count: n),
            session: batchSession)
        let prefillSeconds = (SuspendingClock.now - prefillStart).inSeconds

        let decodeStart = SuspendingClock.now
        for s in 0..<maxNewTokens {
            let toks = BatchedSampler.sample(
                rows: perRowLast,
                configurations: configs,
                histories: gen.map { ArraySlice($0) },
                steps: (0..<n).map { gen[$0].count })
            for b in 0..<n where live[b] {
                gen[b].append(toks[b])
                if eosTokenIds.contains(toks[b]) { live[b] = false }
            }
            if !live.contains(true) { break }
            perRowLast = try await decodeBatch(
                tokensPerRow: toks.map { [$0] },
                startPositions: Array(repeating: promptLen + s, count: n),
                session: batchSession)
        }
        let decodeSeconds = (SuspendingClock.now - decodeStart).inSeconds
        return BatchedRunResult(
            tokens: gen, batchSize: n, promptLen: promptLen,
            prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds)
    }

    /// One batched forward step (lockstep): advance `tokensPerRow` (N rows × q columns) at the
    /// shared cache offset `startPositions` against `session`'s batch=N KV cache, returning each
    /// row's last-token logits. Sampling / EOS / admission live above this seam (the scheduler).
    ///
    /// Dense lockstep: `q` and `startPositions` must be equal across rows (one shared KV write
    /// offset); per-row offsets await a paged / software-gather cache.
    package func decodeBatch(
        tokensPerRow: [[Int32]],
        startPositions: [Int],
        session: GenerationSessionState
    ) async throws -> [[LogitsScalarType]] {
        precondition(!tokensPerRow.isEmpty, "decodeBatch requires at least one row")
        let n = tokensPerRow.count
        precondition(startPositions.count == n, "one start position per row")
        let q = tokensPerRow[0].count
        precondition(
            tokensPerRow.allSatisfy { $0.count == q }, "lockstep decode requires equal-length rows")
        // Ragged (per-row cursor) decode uses shared-write + per-row blit: all rows share the write slot `max(cursor)`,
        // each row's mask/positions are built per-cursor, then shorter rows are blitted down after the
        // forward. Only the decode step (q == 1) is ragged; batched prefill stays equal-length.
        let isRagged = !startPositions.allSatisfy { $0 == startPositions[0] }
        precondition(
            !isRagged || q == 1, "ragged (per-row) start positions require q == 1 (decode only)")

        let names = functionDescriptor.inputNames
        guard let inputIdsName = InputLayout.knownInputIdNames.first(where: names.contains),
            let positionIdsName = InputLayout.knownPositionIdNames.first(where: names.contains)
        else { throw InferenceRuntimeError.invalidState("missing input_ids/position_ids") }
        let maskName = InputLayout.knownAttnMaskNames.first(where: names.contains)
        guard case .ndArray(let idsDesc) = functionDescriptor.inputDescriptor(of: inputIdsName),
            case .ndArray(let posDesc) = functionDescriptor.inputDescriptor(of: positionIdsName)
        else { throw InferenceRuntimeError.invalidInputType("bad input descriptors") }
        var maskDesc: NDArrayDescriptor?
        if let maskName, case .ndArray(let d) = functionDescriptor.inputDescriptor(of: maskName) {
            maskDesc = d
        }
        if isRagged {
            guard maskName != nil, maskDesc != nil else {
                throw InferenceRuntimeError.invalidState(
                    "ragged decode needs an attn_mask graph input (export with --attention-mask batched)")
            }
            guard session.kvCache is RegionCopyable else {
                throw InferenceRuntimeError.invalidState(
                    "ragged decode requires a RegionCopyable KV cache")
            }
        }

        let vocab = config.vocabSize
        // All rows write their new token at the longest row's slot (shared write); shorter rows are
        // blitted down to their own cursor after the forward. Equal cursors ⇒ this is today's path.
        let writeSlot = startPositions.max() ?? 0
        let seqLen = writeSlot + q

        // Row-aware input fills (pure/testable). Uniform cursors reproduce the lockstep tensors
        // byte-for-byte; ragged cursors use the per-row builders.
        let idsFlat = BatchedInputBuilder.tokenIDs(rows: tokensPerRow)
        var ids = NDArray(descriptor: idsDesc.resolvingDynamicDimensions([n, q]))
        fillNDArray(&ids, as: Int32.self, count: n * q) { i in idsFlat[i] }

        let posFlat =
            isRagged
            ? BatchedInputBuilder.raggedDecodePositionIDs(cursors: startPositions)
            : BatchedInputBuilder.positionIDs(rowCount: n, startPos: writeSlot, queryLen: q)
        var pos = NDArray(descriptor: posDesc.resolvingDynamicDimensions([n, seqLen]))
        fillNDArray(&pos, as: Int32.self, count: n * seqLen) { i in posFlat[i] }

        var inputs: [String: NDArray] = [inputIdsName: ids, positionIdsName: pos]
        if let maskName, let maskDesc {
            let maskFlat =
                isRagged
                ? BatchedInputBuilder.raggedDecodeMask(cursors: startPositions)
                : BatchedInputBuilder.additiveCausalMask(
                    rowCount: n, startPos: writeSlot, queryLen: q)
            var mask = NDArray(descriptor: maskDesc.resolvingDynamicDimensions([n, 1, q, seqLen]))
            fillNDArray(&mask, as: Float16.self, count: n * q * seqLen) { i in maskFlat[i] }
            inputs[maskName] = mask
        }

        var out = NDArray(descriptor: logitsDescriptor.resolvingDynamicDimensions([n, q, vocab]))
        // Grow the KV cache to hold this step before writing into it. The batch cache starts at
        // min(256, maxContextLength), so a cohort crossing 256 total positions would otherwise drive
        // startPos off the buffer end.
        _ = try session.kvCache.ensureCapacity(forContextLength: seqLen)
        try await runWithStates(
            function: function, inputs: inputs,
            primary: session.kvCache, secondary: session.additionalStates,
            outputArray: &out, outputName: logitsName)

        // Per-row blit fix-up: each shorter row's new token was written at `writeSlot`; blit it down to
        // the row's real cursor so the dense cache matches a per-row write (bit-identical byte copy).
        if isRagged, let region = session.kvCache as? RegionCopyable {
            for b in 0..<n where startPositions[b] < writeSlot {
                region.copyRegion(row: b, fromSeq: writeSlot, toSeq: startPositions[b])
            }
        }

        let flat = readNDArray(out, as: LogitsScalarType.self, count: n * q * vocab)
        return (0..<n).map { b in
            let base = (b * q + (q - 1)) * vocab
            return Array(flat[base..<base + vocab])
        }
    }

    /// Prefill one row's prompt into batch slab `row` of `session`'s KV cache (continuous admission).
    /// Prefills the prompt alone in a fresh batch-1 scratch session (reusing `decodeBatch` at n=1),
    /// then blits that row's KV prefix `[0, promptLen)` into slab `row` via `copyRowPrefix` — so the
    /// other live rows' slabs are untouched. Returns the prompt's last-token logits.
    ///
    /// Attention-only: a model with additional (conv/sliding) states is not yet supported here (those
    /// states aren't per-row sliced); such a cohort must fall back to the single engine.
    package func prefillRow(
        row: Int,
        promptTokens: [Int32],
        session: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        precondition(!promptTokens.isEmpty, "prefillRow requires a non-empty prompt")
        guard let target = session.kvCache as? GrowingNDArrayState else {
            throw InferenceRuntimeError.invalidState(
                "prefillRow requires an NDArray-backed (GrowingNDArrayState) KV cache")
        }
        if session.additionalStates != nil {
            throw InferenceRuntimeError.invalidState(
                "prefillRow does not yet support models with additional (conv/sliding) states")
        }
        let promptLen = promptTokens.count
        // Populate the batch-1 scratch KV via the standard prefill path, then blit it into slab `row`.
        // The separate `prefill` executable (chunked, fixed shape) writes most of the prompt and the
        // held-back last token runs through `function` for the seed logits. This keeps the heavy, variable
        // prefill work OFF the decode executable's fused heap: a whole-batch decodeBatch(n=1,q=promptLen)
        // here shares that heap with decode (n=N,q=1), so every admission's size toggle strips + leaks the
        // fused heap (~8 MB/admission → OOM under sustained batched load). Routing prefill to its own
        // executable drops the leak ~14×. Assets without a prefill graph fall back to a whole-batch forward
        // (no separate heap available there anyway). This is the same path a standalone request takes, so
        // the cohort-vs-standalone parity is preserved.
        let scratch = try makeSessionState(batchSize: 1)
        let promptSlice = promptTokens[...]
        let logits: [LogitsScalarType]
        switch selectPrefillStrategy(newTokenCount: promptLen) {
        case .chunked(let chunkSize):
            logits = try await processChunkedPrompt(
                tokens: promptSlice, chunkSize: chunkSize, sessionState: scratch)
        case .wholeBatch:
            let allLogits = try await processTokenBatch(promptSlice, sessionState: scratch)
            logits = lastTokenLogits(from: allLogits, vocabSize: config.vocabSize)
        case .oneAtATime:
            var last: [LogitsScalarType] = []
            for j in promptSlice.indices {
                last = try await processTokenBatch(promptSlice[j...j], sessionState: scratch)
            }
            logits = last
        }
        guard let scratchCache = scratch.kvCache as? GrowingNDArrayState else {
            throw InferenceRuntimeError.invalidState("prefillRow scratch cache is not NDArray-backed")
        }
        _ = try target.ensureCapacity(forContextLength: promptLen)
        target.copyRowPrefix(from: scratchCache, toRow: row, count: promptLen)
        return logits
    }

    /// Run one prefill chunk on the prefill graph: KV cache writes only, no logits.
    private func encodePrefillChunk(
        _ tokens: ArraySlice<Int32>, using prefillFn: InferenceFunction,
        sessionState: GenerationSessionState
    ) async throws {
        let batchSize = tokens.count
        _ = try sessionState.kvCache.ensureCapacity(
            forContextLength: sessionState.processedTokenCount + batchSize)

        let context = InputContext.dynamic(
            tokens: tokens, processedTokenCount: sessionState.processedTokenCount)
        guard var handler = sessionState.inputHandler else {
            throw InferenceRuntimeError.invalidState("session has no input handler")
        }
        let inputs = try await handler.prepare(context)
        sessionState.inputHandler = handler  // retain the handler's mutated reusable buffers

        try await runWithStatesNoOutputs(
            function: prefillFn,
            inputs: inputs,
            primary: sessionState.kvCache,
            secondary: sessionState.additionalStates)

        sessionState.processedTokenCount += batchSize
    }

    // MARK: - Chunked Prefill

    private func processChunkedPrompt(
        tokens: ArraySlice<Int32>,
        chunkSize: Int,
        sessionState: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        // The prefill graph produces no logits, so hold the final token back for
        // `function`: it is the one whose logits seed sampling. Without one, nothing is
        // held back and the trailing chunk carries the logits.
        let heldBack = prefillHeldBackTokens(hasPrefillGraph: prefillFunction != nil)

        let chunkSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIClean Chunked Prefill",
            details: "\(tokens.count) tokens, chunkSize \(chunkSize)"
        )
        defer {
            InstrumentsProfiler.endCustomInterval(
                name: "CoreAIClean Chunked Prefill",
                signpostID: chunkSignpost
            )
        }

        return try await runChunkedPrefill(
            tokens: tokens,
            chunkSize: chunkSize,
            heldBack: heldBack,
            vocabSize: config.vocabSize
        ) { chunk, isHeldBack in
            // Held-back tail (and every chunk when there is no prefill graph) runs through
            // `main` for logits; earlier chunks fill the KV cache via the prefill graph.
            if !isHeldBack, let prefillFn = self.prefillFunction {
                try await self.encodePrefillChunk(chunk, using: prefillFn, sessionState: sessionState)
                return []
            }
            return try await self.processTokenBatch(chunk, sessionState: sessionState)
        }
    }

    /// Process tokens in chunks, returning ALL position logits (not just last token).
    /// Used for batched PPL evaluation where every position's logits are needed.
    func processChunkedPromptAllLogits(
        tokens: ArraySlice<Int32>,
        chunkSize: Int,
        sessionState: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        var allLogits: [LogitsScalarType] = []
        var remainingTokens = tokens

        while !remainingTokens.isEmpty {
            let currentChunkSize = min(chunkSize, remainingTokens.count)
            let chunkEnd = remainingTokens.startIndex + currentChunkSize
            let chunk = remainingTokens[remainingTokens.startIndex..<chunkEnd]
            let chunkLogits = try await processTokenBatch(chunk, sessionState: sessionState)
            allLogits.append(contentsOf: chunkLogits)
            remainingTokens = remainingTokens[chunkEnd...]
        }

        return allLogits
    }

    // MARK: - Generate (primary API)

    /// Shim over the idempotent generate path: drives generation on the engine's single
    /// internal session, preserving today's implicit prefix-cache reuse and reset(to:)
    /// semantics for existing callers.
    public func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        try await generate(
            with: input,
            sessionState: session,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions
        )
    }

    // MARK: - Lifecycle

    /// Wait for any in-flight generate() Task to finish.
    private func drain() {
        var attempts = 0
        while session.tokenBox.isBusy {
            attempts += 1
            if attempts > 5000 {
                fatalError("Sequential engine drain() timeout — generation Task stuck?")
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    public func cancel() async throws {
        session.tokenBox.cancelActive()
    }

    public func reset(to tokenIndex: Int) async throws {
        precondition(
            tokenIndex >= 0 && tokenIndex <= session.processedTokenCount,
            "reset(to: \(tokenIndex)) out of range [0, \(session.processedTokenCount)]")
        if tokenIndex != 0 && session.hasNonTruncatableStates {
            throw InferenceRuntimeError.invalidState(
                "Partial reset is not supported for hybrid models with recurrent state. "
                    + "Use reset(to: 0) and replay the prefix.")
        }
        session.tokenBox.cancelActive()
        internalReset(to: tokenIndex, sessionState: session)
    }

    /// Internal reset without cancelling the active generation token.
    /// Used by `reset(to:)` and the prefix-cache routing in `generate(with:sessionState:)`.
    func internalReset(to tokenIndex: Int, sessionState: GenerationSessionState) {
        let resetSpan = InstrumentsProfiler.beginReset(engine: "CoreAIClean")
        if tokenIndex == 0 {
            sessionState.processedTokenCount = 0
            sessionState.history.clear()
            sessionState.kvCache.reset()
            sessionState.additionalStates?.reset()
        } else {
            sessionState.processedTokenCount = tokenIndex
            sessionState.history.truncate(to: tokenIndex)
        }
        resetSpan.end()
    }

    public func cleanup() {
        let cleanupSpan = InstrumentsProfiler.beginCleanup(engine: "CoreAIClean")
        CLILogger.log("CoreAI clean engine cleanup complete")
        cleanupSpan.end()
    }

    // MARK: - Helpers
}

// MARK: - Idempotent Engine Conformance

extension CoreAISequentialEngine {
    /// Mint a fresh, independent per-request state (fresh KV cache + empty history).
    package func makeSessionState() throws -> GenerationSessionState {
        try makeSessionState(batchSize: 1)
    }

    /// Mint a fresh per-request state whose KV cache is sized for `batchSize` rows. `batchSize > 1`
    /// requires a batch-capable graph (static batch=N, or a dynamic-batch graph); the cache's batch
    /// dim resolves to `batchSize` instead of 1.
    package func makeSessionState(batchSize: Int) throws -> GenerationSessionState {
        let stateHandlers = try StateHandlerFactory.createSyncHandlers(
            descriptor: functionDescriptor,
            maxContextLength: config.maxContextLength,
            options: options,
            batchSize: batchSize)
        let session = GenerationSessionState(
            kvCache: stateHandlers.kvCache,
            additionalStates: stateHandlers.additionalStates,
            hasNonTruncatableStates: stateHandlers.hasNonTruncatableStates)
        // Each session owns its input handler (fresh reusable buffers) so the single lane on this
        // session never shares scratch with another session's generation.
        session.inputHandler = try makeInputHandler()
        return session
    }

    /// Drive generation over the caller-supplied state instead of engine-owned state.
    package func generate(
        with input: [TokenId],
        sessionState: GenerationSessionState,
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        // Cancel this session's prior generation so its Iterator stops on next poll. Per-session,
        // so a generation on another session is left untouched.
        sessionState.tokenBox.cancelActive()

        // Implicit prefix caching: resolve input against history.
        // For hybrid models with recurrent states, we must full-reset on any
        // rewind because recurrent state summarizes the whole prefix and cannot
        // be truncated by moving a KV cursor. Future: checkpoint/restore.
        // Default to 0 so a fresh session (no history) always reports a clean
        // value instead of leaking the previous session's hit count.
        sessionState.lastPrefixHitCount = 0
        if sessionState.history.count > 0 {
            let (commonPrefix, _) = sessionState.history.resolve(input: input)
            let plan = Self.prefixResetPlan(
                commonPrefix: commonPrefix,
                historyCount: sessionState.history.count,
                processedTokenCount: sessionState.processedTokenCount,
                inputCount: input.count,
                hasNonTruncatableStates: sessionState.hasNonTruncatableStates
            )
            if let resetTo = plan.resetTo {
                internalReset(to: resetTo, sessionState: sessionState)
            }
            sessionState.lastPrefixHitCount = plan.prefixHitCount
        }
        // Mirror the driven session's hit count for the single-valued protocol getter.
        lastPrefixHitCount = sessionState.lastPrefixHitCount

        let token = GenerationToken()
        sessionState.tokenBox.install(token)
        return GenerationSequence(
            engine: self,
            input: input,
            sessionState: sessionState,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions,
            generationToken: token
        )
    }

    /// Pure decision for the implicit prefix-cache routing in `generate(with:sessionState:)`.
    ///
    /// Given the common-prefix length resolved against the session history and the
    /// current cursor state, decides how far to rewind the KV cache and what to
    /// report as the prefix hit count. `resetTo == nil` means no reset is needed.
    ///
    /// Extracted so the four-way routing can be unit-tested without a compiled model;
    /// the logic is equivalent to the inline decision it replaced.
    static func prefixResetPlan(
        commonPrefix: Int,
        historyCount: Int,
        processedTokenCount: Int,
        inputCount: Int,
        hasNonTruncatableStates: Bool
    ) -> (resetTo: Int?, prefixHitCount: Int) {
        if hasNonTruncatableStates {
            // Hybrid model: recurrent state can't be partially rewound.
            // Full reset and replay the entire prompt.
            if commonPrefix < historyCount || processedTokenCount >= inputCount {
                return (0, 0)
            }
            return (nil, 0)
        } else if commonPrefix < inputCount && commonPrefix < historyCount {
            // Divergence: input differs from history. Full reset needed.
            return (0, commonPrefix)
        } else if processedTokenCount >= inputCount {
            // Pure extension: all input tokens match history. Rewind for seeding.
            return (Swift.max(0, commonPrefix - 1), commonPrefix)
        } else {
            return (nil, commonPrefix)
        }
    }
}

extension CoreAISequentialEngine {
    /// Async sequence of `InferenceOutput` produced by `generate()`.
    public struct GenerationSequence: InferenceOutputSequence {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        let engine: CoreAISequentialEngine
        let input: [CoreAISequentialEngine.TokenId]
        let sessionState: GenerationSessionState
        let samplingConfiguration: SamplingConfiguration
        let inferenceOptions: InferenceOptions
        let generationToken: GenerationToken

        /// Shared with the iterator so the caller can read why generation ended.
        let stopReasonStore = StopReasonStore()

        public var stopReason: StopReason? { stopReasonStore.stopReason }

        public func setStopReason(_ reason: StopReason) {
            stopReasonStore.set(reason)
        }

        public func makeAsyncIterator() -> Iterator {
            Iterator(
                engine: engine,
                input: input,
                sessionState: sessionState,
                samplingConfiguration: samplingConfiguration,
                inferenceOptions: inferenceOptions,
                stopReasonStore: stopReasonStore,
                generationToken: generationToken
            )
        }
    }
}

extension CoreAISequentialEngine.GenerationSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        private let engine: CoreAISequentialEngine
        private let sessionState: GenerationSessionState
        private let samplingConfiguration: SamplingConfiguration
        private let returnsLogits: Bool
        private let forcedContinuation: [CoreAISequentialEngine.TokenId]?
        private let maxTokens: Int
        private let stopReasonStore: StopReasonStore
        private let generationToken: GenerationToken

        private var inputTokens: [CoreAISequentialEngine.TokenId]
        private let generationStartOffset: Int
        private var step: Int = 0
        private var finished: Bool = false
        // Pre-computed logits for batched forcedContinuation evaluation.
        // When non-nil, next() yields from this buffer instead of running inference.
        private var batchedLogitsBuffer: [[LogitsScalarType]]?

        init(
            engine: CoreAISequentialEngine,
            input: [CoreAISequentialEngine.TokenId],
            sessionState: GenerationSessionState,
            samplingConfiguration: SamplingConfiguration,
            inferenceOptions: InferenceOptions,
            stopReasonStore: StopReasonStore,
            generationToken: GenerationToken
        ) {
            self.engine = engine
            self.sessionState = sessionState
            self.samplingConfiguration = samplingConfiguration.normalized()
            self.returnsLogits = inferenceOptions.includeLogits
            self.forcedContinuation = inferenceOptions.forcedContinuation
            self.stopReasonStore = stopReasonStore
            self.generationToken = generationToken
            self.inputTokens = input
            self.generationStartOffset = input.count
            self.maxTokens = SequentialIterator.clampMaxTokens(
                requested: inferenceOptions.maxTokens,
                forcedCount: inferenceOptions.forcedContinuation?.count,
                inputCount: input.count,
                maxContextLength: engine.config.maxContextLength
            )
        }

        deinit {
            sessionState.tokenBox.clearIfActive(generationToken)
        }

        public func next() async throws -> InferenceOutput? {
            if finished { return nil }

            if generationToken.isCancelled {
                stopReasonStore.set(.cancelled)
                finishAndRelease()
                return nil
            }

            guard step < maxTokens else {
                // Natural exhaustion. Don't clobber a reason a decoder set (e.g. `.eos`).
                stopReasonStore.setIfUnset(.maxTokens)
                finishAndRelease()
                return nil
            }

            // Fast path: batched forcedContinuation with logits.
            // All tokens were processed in one prefill; yield pre-computed logits.
            if let buffer = batchedLogitsBuffer {
                let logits = buffer[step]
                let token = forcedContinuation![step]
                step += 1
                if step >= maxTokens {
                    stopReasonStore.setIfUnset(.maxTokens)
                    finishAndRelease()
                }
                return InferenceOutput(tokenId: token, logits: logits)
            }

            // First call with forcedContinuation + logits: batch-process all tokens at once.
            if let forced = forcedContinuation, returnsLogits, step == 0 {
                let allTokens = inputTokens + forced.map { $0 }
                let vocabSize = engine.config.vocabSize

                let allLogits: [LogitsScalarType]
                let strategy = engine.selectPrefillStrategy(newTokenCount: allTokens.count)
                switch strategy {
                case .chunked(let chunkSize):
                    allLogits = try await engine.processChunkedPromptAllLogits(
                        tokens: allTokens[...], chunkSize: chunkSize, sessionState: sessionState)
                case .wholeBatch:
                    allLogits = try await engine.processTokenBatch(
                        allTokens[...], sessionState: sessionState)
                case .oneAtATime:
                    var collected: [LogitsScalarType] = []
                    for j in allTokens.indices {
                        collected.append(
                            contentsOf: try await engine.processTokenBatch(
                                allTokens[j...j], sessionState: sessionState))
                    }
                    allLogits = collected
                }

                // Split into per-position logit vectors.
                // Skip the prompt positions (inputTokens.count - 1 positions);
                // we want logits that predict each forced token.
                let promptLen = inputTokens.count
                var buffer: [[LogitsScalarType]] = []
                for i in 0..<forced.count {
                    let offset = (promptLen - 1 + i) * vocabSize
                    let endOffset = offset + vocabSize
                    guard endOffset <= allLogits.count else {
                        throw InferenceRuntimeError.invalidState(
                            "Batched logits underflow at position \(i): need \(endOffset), got \(allLogits.count)")
                    }
                    buffer.append(Array(allLogits[offset..<endOffset]))
                }
                batchedLogitsBuffer = buffer

                // Update engine state
                sessionState.history.append(contentsOf: allTokens[...])

                // Yield first result
                let logits = buffer[step]
                let token = forced[step]
                step += 1
                return InferenceOutput(tokenId: token, logits: logits)
            }

            do {
                try Task.checkCancellation()

                guard sessionState.processedTokenCount < inputTokens.count else {
                    throw InferenceRuntimeError.invalidState("No new tokens to process")
                }

                let oldProcessedCount = sessionState.processedTokenCount
                let newTokens = inputTokens[sessionState.processedTokenCount...]
                let strategy = engine.selectPrefillStrategy(newTokenCount: newTokens.count)

                let logitBuffer: [LogitsScalarType]
                switch strategy {
                case .chunked(let chunkSize):
                    logitBuffer = try await engine.processChunkedPrompt(
                        tokens: newTokens, chunkSize: chunkSize, sessionState: sessionState)
                case .wholeBatch:
                    let allLogits = try await engine.processTokenBatch(
                        newTokens, sessionState: sessionState)
                    logitBuffer = lastTokenLogits(from: allLogits, vocabSize: engine.config.vocabSize)
                case .oneAtATime:
                    var lastLogits: [LogitsScalarType] = []
                    for j in newTokens.indices {
                        lastLogits = try await engine.processTokenBatch(
                            newTokens[j...j], sessionState: sessionState)
                    }
                    logitBuffer = lastLogits
                }

                // Update history with newly processed tokens
                let processedSlice = inputTokens[oldProcessedCount..<sessionState.processedTokenCount]
                sessionState.history.append(contentsOf: processedSlice)

                // Check cancellation after inference step
                if generationToken.isCancelled {
                    stopReasonStore.set(.cancelled)
                    finishAndRelease()
                    return nil
                }

                let nextToken = SequentialIterator.nextToken(
                    fromLogits: logitBuffer,
                    forced: forcedContinuation,
                    step: step,
                    sampling: samplingConfiguration,
                    tokenHistory: inputTokens[generationStartOffset...]
                )

                inputTokens.append(nextToken)
                step += 1

                return InferenceOutput(
                    tokenId: nextToken,
                    logits: returnsLogits ? logitBuffer : nil
                )
            } catch is CancellationError {
                stopReasonStore.set(.cancelled)
                finishAndRelease()
                throw CancellationError()
            } catch {
                stopReasonStore.set(.error)
                finishAndRelease()
                throw error
            }
        }

        private func finishAndRelease() {
            guard !finished else {
                return
            }
            finished = true
            sessionState.tokenBox.clearIfActive(generationToken)
        }
    }
}
