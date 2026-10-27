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

    // Ring capacity when a sliding-window (ring) KV state is present. `function` writes the
    // ring before attending, so a multi-token step past the ring would silently evict history
    // its earlier queries still need; `processTokenBatch` refuses it. Prefill chunks also
    // break at ring boundaries so no single ring write runs past the last slot.
    private let slidingRingCapacity: Int?

    // Whether the single-token fallback past the ring has been logged (see `runMainSteps`).
    private var loggedRingSingleSteps = false

    // I/O names from descriptor
    private let logitsName: String

    // Builds a new input handler for each session, so sessions never share its reusable buffers.
    // When the graph has an `attn_mask` input, the handler fills it as a composite extra.
    private let makeInputHandler: () throws -> CompositeInputHandler<TokenInputHandler>

    /// Retained to build per-session state handlers in `makeSessionState()`.
    private let options: EngineOptions

    // Per-request mutable state. The `generate()` shim owns one internal session and
    // reuses it across calls, preserving implicit prefix-cache reuse and reset(to:)
    // semantics. Idempotent callers pass their own session instead.
    private let session: GenerationSessionState

    /// Bytes held by the engine's own session: its KV cache and persistent states, and its logits
    /// buffer, estimated from their shapes and element sizes.
    package var estimatedSessionBytes: (states: Int, logitsScratch: Int) {
        (session.estimatedStateBytes, session.estimatedLogitsScratchBytes)
    }

    // The single-row logits buffer lives on each session (`GenerationSessionState.logitsScratch`).
    private let logitsDescriptor: NDArrayDescriptor

    // Ring buffer mode: handled by TokenInputHandler.useCompactPositionIds

    // Track processed tokens for incremental inference (delegated to the shim session).
    public var processedTokenCount: Int { session.processedTokenCount }

    /// The static batch size of the graph, read from the leading dimension of `input_ids`.
    ///
    /// A dynamic-batch graph reports 1 here, while `maxBatchCapacity` reports `Int.max` for it.
    package var declaredBatchSize: Int {
        guard let name = InputLayout.knownInputIdNames.first(where: functionDescriptor.inputNames.contains),
            case .ndArray(let desc) = functionDescriptor.inputDescriptor(of: name),
            let dim0 = desc.shape.first, dim0 > 0
        else { return 1 }
        return dim0
    }

    var maxBatchCapacity: Int {
        guard let name = InputLayout.knownInputIdNames.first(where: functionDescriptor.inputNames.contains),
            case .ndArray(let desc) = functionDescriptor.inputDescriptor(of: name),
            let dim0 = desc.shape.first
        else { return 1 }
        if dim0 > 1 { return dim0 }  // Static batch=N graph.
        if dim0 < 0 { return Int.max }  // Dynamic batch. The runtime enforces the exported maximum.
        return 1  // Single-batch graph.
    }

    /// True when the graph has an attention-mask input and a batch dimension above 1, static or
    /// dynamic.
    package var supportsBatching: Bool {
        functionDescriptor.inputNames.contains(where: InputLayout.knownAttnMaskNames.contains)
            && maxBatchCapacity > 1
    }

    /// The prefix hit count of the most recent generation. With several sessions in use, this is
    /// the count of whichever session ran last.
    public private(set) var lastPrefixHitCount: Int = 0

    /// Whether a generation started by `generate(with:samplingConfiguration:inferenceOptions:)` is
    /// in flight. Generations on caller-supplied sessions are not counted.
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

        // Validate model architecture: 2 inputs (3 with `attn_mask`), 1+ output, at least KV cache pair.
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
            stateKinds: config.states,
            options: options
        )
        self.options = options
        self.slidingRingCapacity = stateHandlers.slidingRingCapacity
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

        // With an `attn_mask` input, the handler fills a single-row causal mask each step.
        // `decodeBatch` builds its own per-row masks and does not use this handler.
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

        // A session without a handler throws in `processTokenBatch`, so the default session needs one.
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

        try StateHandlerFactory.checkWriteFirstStep(
            offset: sessionState.processedTokenCount, queryLength: batchSize,
            ringCapacity: slidingRingCapacity)

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
        sessionState.inputHandler = handler  // Value type: store it back to keep its reusable buffers.

        // Reuse this session's logits buffer, and reallocate only when the step's token count changes.
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
        sessionState.logitsScratch = logitsArray  // Store the handle back so the next step reuses it.

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

    // MARK: - Batched decode

    /// Output of `lockstepBatchedGenerate`. `tokens` is `[N][generated]`, and a row that hit EOS is
    /// shorter and ends with the EOS token. Times are wall-clock seconds.
    package struct BatchedRunResult: Sendable {
        package let tokens: [[Int32]]
        package let prefillSeconds: Double
        package let decodeSeconds: Double
    }

    /// Prefills and decodes `N` equal-length prompts in lockstep on a new batch=N session, sampling
    /// each row with its own entry in `configs`.
    ///
    /// A row that samples an EOS token stops collecting tokens but stays in the batch until every
    /// row is done. Pass an empty `eosTokenIds` to always generate `maxNewTokens` per row.
    ///
    /// The prompts are prefilled as one `decodeBatch` step on the decode graph, not through the
    /// chunked prefill path that `prefillRow` uses. For rows at different cursors, use `prefillRow`
    /// and `decodeBatch` directly.
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
            tokens: gen, prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds)
    }

    /// See ``BatchedDecodeStep`` for the contract and the admission pattern.
    func decodeBatch(
        tokensPerRow: [[Int32]],
        startPositions: [Int],
        session: GenerationSessionState
    ) async throws -> [[LogitsScalarType]] {
        let out = try await runBatchedForward(
            tokensPerRow: tokensPerRow, startPositions: startPositions, session: session)
        let n = tokensPerRow.count
        let q = tokensPerRow[0].count
        let vocab = config.vocabSize
        return Self.logitsRows(
            out, count: n, rowLength: vocab, firstOffset: (q - 1) * vocab, rowStride: q * vocab)
    }

    func allPositionLogits(
        tokensPerRow: [[Int32]],
        startPosition: Int,
        session: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        let n = tokensPerRow.count
        let out = try await runBatchedForward(
            tokensPerRow: tokensPerRow, startPositions: Array(repeating: startPosition, count: n),
            session: session)
        return readNDArray(out, as: LogitsScalarType.self, count: n * tokensPerRow[0].count * config.vocabSize)
    }

    /// Copies `count` rows of `rowLength` logits out of `array`. Row `b` starts at element
    /// `firstOffset + b * rowStride` in row-major order.
    private static func logitsRows(
        _ array: NDArray, count: Int, rowLength: Int, firstOffset: Int, rowStride: Int
    ) -> [[LogitsScalarType]] {
        let view = array.view(as: LogitsScalarType.self)
        let contiguous = view.withUnsafePointer { _, shape, strides in
            isContiguousRowMajor(shape: shape, strides: strides)
        }
        guard contiguous else {
            let total = firstOffset + (count - 1) * rowStride + rowLength
            let flat = readNDArray(array, as: LogitsScalarType.self, count: total)
            return (0..<count).map { b in
                let start = firstOffset + b * rowStride
                return Array(flat[start..<start + rowLength])
            }
        }
        return view.withUnsafePointer { pointer, _, _ in
            (0..<count).map { b in
                Array(UnsafeBufferPointer(start: pointer + firstOffset + b * rowStride, count: rowLength))
            }
        }
    }

    /// Runs one batched forward step and returns the logits output, `[N, q, vocab]`.
    private func runBatchedForward(
        tokensPerRow: [[Int32]],
        startPositions: [Int],
        session: GenerationSessionState
    ) async throws -> NDArray {
        precondition(!tokensPerRow.isEmpty, "a batched forward requires at least one row")
        let n = tokensPerRow.count
        precondition(startPositions.count == n, "one start position per row")
        let q = tokensPerRow[0].count
        precondition(
            tokensPerRow.allSatisfy { $0.count == q }, "lockstep decode requires equal-length rows")
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
        // Every row writes its new token at `writeSlot = max(startPositions)`. After the forward pass,
        // shorter rows are copied down to their own cursors. Equal cursors need no copy.
        let writeSlot = startPositions.max() ?? 0
        let seqLen = writeSlot + q

        // `BatchedInputBuilder` documents the flat buffer layouts. position_ids and the mask span the
        // whole key axis `seqLen`, not just the `q` new tokens.
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
        // The batch cache starts small, so grow it before the graph writes up to `seqLen`.
        _ = try session.kvCache.ensureCapacity(forContextLength: seqLen)
        try await runWithStates(
            function: function, inputs: inputs,
            primary: session.kvCache, secondary: session.additionalStates,
            outputArray: &out, outputName: logitsName)

        // Move each shorter row's new KV entry from `writeSlot` down to its own cursor.
        if isRagged, let region = session.kvCache as? RegionCopyable {
            for b in 0..<n where startPositions[b] < writeSlot {
                region.copyRegion(row: b, fromSeq: writeSlot, toSeq: startPositions[b])
            }
        }

        return out
    }

    /// See ``BatchedDecodeStep`` for the contract.
    ///
    /// The prompt runs alone on a batch-1 scratch session through the normal prefill path, so the
    /// row matches a standalone request. Its `[0, promptLen)` KV prefix is then copied into the row.
    /// Throws for models with additional states (conv or sliding window), which are not per row.
    func prefillRow(
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
        // Use the normal prefill path, which runs chunks on the separate `prefill` graph when the asset
        // has one. Running the prompt through `decodeBatch(q: promptLen)` would change the decode
        // executable's input shapes on every admission, and each shape change leaks its heap.
        // Assets without a prefill graph still run the prompt on the decode graph.
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
        sessionState.inputHandler = handler  // Value type: store it back to keep its reusable buffers.

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

        // Without a prefill graph every chunk runs on write-first `main`; past a sliding ring
        // only single-token steps are correct there.
        if prefillFunction == nil, slidingRingCapacity != nil {
            let logits = try await runMainSteps(
                tokens, chunkSize: chunkSize, keepAllLogits: false, sessionState: sessionState)
            return lastTokenLogits(from: logits, vocabSize: config.vocabSize)
        }

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
            vocabSize: config.vocabSize,
            startOffset: sessionState.processedTokenCount,
            ringCapacity: slidingRingCapacity
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
        try await runMainSteps(tokens, chunkSize: chunkSize, keepAllLogits: true, sessionState: sessionState)
    }

    /// Run `tokens` through `main` in steps from `mainGraphStepSizes`, returning every
    /// position's logits (`keepAllLogits`) or just the last step's.
    private func runMainSteps(
        _ tokens: ArraySlice<Int32>,
        chunkSize: Int,
        keepAllLogits: Bool,
        sessionState: GenerationSessionState
    ) async throws -> [LogitsScalarType] {
        let startOffset = sessionState.processedTokenCount
        if let ring = slidingRingCapacity, chunkSize > 1, startOffset + tokens.count > ring,
            !loggedRingSingleSteps
        {
            loggedRingSingleSteps = true
            CLILogger.log(
                "Sliding-window model: positions past the \(ring)-slot ring run one token per "
                    + "step on 'main' to keep logits correct (slower)")
        }

        var logits: [LogitsScalarType] = []
        var remaining = tokens
        for size in mainGraphStepSizes(
            tokenCount: tokens.count, chunkSize: chunkSize, startOffset: startOffset,
            ringCapacity: slidingRingCapacity)
        {
            let end = remaining.startIndex + size
            let stepLogits = try await processTokenBatch(
                remaining[remaining.startIndex..<end], sessionState: sessionState)
            if keepAllLogits {
                logits.append(contentsOf: stepLogits)
            } else {
                logits = stepLogits
            }
            remaining = remaining[end...]
        }
        return logits
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
        if !StateHandlerFactory.ringRewindIsSafe(
            from: session.processedTokenCount, to: tokenIndex, ringCapacity: slidingRingCapacity)
        {
            throw InferenceRuntimeError.invalidState(
                "Partial reset to \(tokenIndex) would read sliding-ring slots that have been "
                    + "overwritten. Use reset(to: 0) and replay the prefix.")
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

    /// Creates a session whose KV cache has `batchSize` rows. A `batchSize` above 1 needs a graph
    /// with `maxBatchCapacity >= batchSize`.
    func makeSessionState(batchSize: Int) throws -> GenerationSessionState {
        let stateHandlers = try StateHandlerFactory.createSyncHandlers(
            descriptor: functionDescriptor,
            maxContextLength: config.maxContextLength,
            stateKinds: config.states,
            options: options,
            batchSize: batchSize)
        let session = GenerationSessionState(
            kvCache: stateHandlers.kvCache,
            additionalStates: stateHandlers.additionalStates,
            hasNonTruncatableStates: stateHandlers.hasNonTruncatableStates)
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
        // Cancel an earlier generation on this session only. Other sessions keep running.
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
                hasNonTruncatableStates: sessionState.hasNonTruncatableStates,
                ringCapacity: slidingRingCapacity
            )
            if let resetTo = plan.resetTo {
                internalReset(to: resetTo, sessionState: sessionState)
            }
            sessionState.lastPrefixHitCount = plan.prefixHitCount
        }
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
        hasNonTruncatableStates: Bool,
        ringCapacity: Int? = nil
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
            let target = Swift.max(0, commonPrefix - 1)
            guard
                StateHandlerFactory.ringRewindIsSafe(
                    from: processedTokenCount, to: target, ringCapacity: ringCapacity)
            else {
                // The ring has overwritten slots the rewound position reads: replay.
                return (0, 0)
            }
            return (target, commonPrefix)
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
            self.returnsLogits = inferenceOptions.returnsLogits
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
                    allLogits = try await engine.processChunkedPromptAllLogits(
                        tokens: allTokens[...], chunkSize: allTokens.count, sessionState: sessionState)
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
                    let allLogits = try await engine.runMainSteps(
                        newTokens, chunkSize: newTokens.count, keepAllLogits: false,
                        sessionState: sessionState)
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
