// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIShared
import Foundation
import Synchronization

/// Static-shape inference engine using Core AI models.
///
/// A static-shape asset is a ladder of statically-shaped programs named
/// `extend_<ctx>_<qlen>` / `prompt_opt_<ctx>_<qlen>`. This engine picks a rung
/// per step, fills that rung's inputs, binds the asset's persistent states, and
/// runs it.
///
/// Inputs come from ``StaticInputHandler``s chosen at init from what the graph
/// declares, and states from a ``StaticStateSet`` classified from the asset.
public final class StaticShapeEngine: InferenceEngine, @unchecked Sendable {
    public typealias ConfigType = ModelConfig

    public var supportsLogits: Bool { true }

    // MARK: I/O name contracts — models must use these exact names

    private static let logitsOutputName = "out_logits"
    private static let keyCacheName = "key_cache"
    private static let valueCacheName = "value_cache"
    private static let embeddingTableName = "embedding_table"
    private static let tokenIDsInputName = "in_new_token_ids"
    private static let gatheredOutputName = "out_transformer_input"

    // MARK: Function name parsing

    private struct FunctionDimensions {
        let contextLength: Int
        let queryLength: Int
    }

    /// Parses the `<ctx>_<qlen>` suffix shared by `extend_*` and `prompt_opt_*`.
    private static func parseFunctionDimensions(_ name: String) -> FunctionDimensions? {
        let parts = Array(name.split(separator: "_").suffix(2))
        guard parts.count == 2, let ctx = Int(parts[0]), let ql = Int(parts[1]) else { return nil }
        return FunctionDimensions(contextLength: ctx, queryLength: ql)
    }

    // MARK: Input name resolution

    private static let knownStepNames = ["in_step", "step"]
    private static let knownTransformerInputNames = ["transformer_input"]

    private static func resolveInputName(
        from inputNames: [String], candidates: [String]
    ) -> String? {
        candidates.first(where: inputNames.contains)
    }

    public var vocabSize: Int { config.vocabSize }

    public let config: ModelConfig
    private let model: AIModel

    // MARK: Properties

    // Lazily loaded inference functions, keyed by name.
    private var functions: [String: InferenceFunction]

    // Available function names by category.
    private let extendFunctionNames: [String]
    private let gatherFunctionNames: Set<String>

    // The ladder's rungs by role: `extend_*` decodes, `prompt_opt_*` prefills. An asset
    // need not ship both at the same query lengths.
    private static let decodeFunctionPrefix = "extend"
    private static let prefillFunctionPrefix = "prompt_opt"
    private let decodeRungs: [FunctionDimensions]
    private let prefillRungs: [FunctionDimensions]

    // Embedding table loaded once at init.
    private let embeddingTable: NDArray

    // Largest query length across all extend functions — used as prefill threshold.
    private let maxQueryLength: Int

    // Persistent model states, classified and sized from the asset.
    private let states: StaticStateSet

    // Per-step inputs. Handlers fill pre-allocated buffers in place.
    private let inputHandlers: [any StaticInputHandler]
    private var inputBuffers: InputBuffers

    // Number of tokens already processed in the current sequence.
    public private(set) var processedTokenCount: Int = 0

    // Token history for implicit prefix caching
    private var history = TokenHistory()
    public private(set) var lastPrefixHitCount: Int = 0

    // Track in-flight generation via token
    private let _activeToken = Mutex<GenerationToken?>(nil)

    public var isBusy: Bool { _activeToken.withLock { $0 != nil } }

    // MARK: Multimodal (vision) — nil for text-only bundles

    let visionModel: AIModel?
    let visionConfig: VisionConfig?

    // Per-generation image state, set by `generate(with: InputEmbeddings, ...)` and
    // consumed during `buildInputs`. Cleared by the text-only `generate` entry so a
    // subsequent text turn never scatters stale embeddings.
    var pendingImageEmbeddings: NDArray?
    var imageTokenPositions: [Int] = []

    /// Clear the engine's active token if it matches the given token.
    func clearTokenIfActive(_ token: GenerationToken) {
        let wasActive = _activeToken.withLock { current -> Bool in
            guard current === token else { return false }
            current = nil
            return true
        }
        // Only drop image state if this token was still the active one — a
        // superseded iterator finishing late (after a newer generate() call already
        // replaced both the token and any image state) must not clobber the new turn.
        if wasActive {
            pendingImageEmbeddings = nil
            imageTokenPositions = []
        }
    }

    // MARK: - Initialization

    /// - Parameter tensorData: Tensor data the graph reads alongside the model, keyed by
    ///   `auxiliary_assets` role (see ``EngineOptions/TensorDataKey``). Only the roles the graph's
    ///   inputs need are looked up.
    public init(
        configuration: ModelConfig,
        preparedModel: PreparedModel,
        tensorData: [String: URL] = [:],
        visionModel: AIModel? = nil,
        visionConfig: VisionConfig? = nil
    ) async throws {
        self.config = configuration
        self.model = preparedModel.model
        self.functions = [:]
        self.visionModel = visionModel
        self.visionConfig = visionConfig

        let allNames = model.functionNames
        CLILogger.log("Model loaded: \(allNames.count) functions: \(allNames.sorted())")

        self.extendFunctionNames =
            allNames
            .filter { $0.hasPrefix("extend") || $0.hasPrefix("prompt") }
            .sorted()
        self.gatherFunctionNames = Set(allNames.filter { $0.hasPrefix("gather_embeddings") })
        self.decodeRungs = extendFunctionNames.filter { $0.hasPrefix(Self.decodeFunctionPrefix + "_") }
            .compactMap(Self.parseFunctionDimensions)
        self.prefillRungs = extendFunctionNames.filter { $0.hasPrefix(Self.prefillFunctionPrefix + "_") }
            .compactMap(Self.parseFunctionDimensions)

        CLILogger.log(
            "Parsed \(extendFunctionNames.count) decoder functions, \(gatherFunctionNames.count) gather functions")

        // Index every rung of the ladder by its (query length, context) pair, and
        // every context bucket by one representative descriptor. Both `extend_*`
        // and `prompt_opt_*` at the same pair declare identical input shapes, so
        // either may stand in; states come from `extend_*` only.
        var functionsByKey: [(key: StaticBucketKey, descriptor: InferenceFunctionDescriptor)] = []
        var descriptorsByContext: [Int: InferenceFunctionDescriptor] = [:]
        for name in extendFunctionNames {
            guard let dims = Self.parseFunctionDimensions(name),
                let descriptor = model.functionDescriptor(for: name)
            else { continue }
            functionsByKey.append(
                (
                    StaticBucketKey(batchSize: dims.queryLength, contextBucket: dims.contextLength),
                    descriptor
                ))
            if name.hasPrefix("extend"), descriptorsByContext[dims.contextLength] == nil {
                descriptorsByContext[dims.contextLength] = descriptor
            }
        }

        self.maxQueryLength = functionsByKey.map(\.key.batchSize).max() ?? 64

        // Reference rung: the largest context bucket. Used to enumerate states, to
        // size fixed ones, and to decide which optional handlers this asset needs.
        guard let largestContext = descriptorsByContext.keys.max(),
            let referenceDescriptor = descriptorsByContext[largestContext]
        else {
            throw InferenceRuntimeError.invalidState(
                "No `extend_<ctx>_<qlen>` functions found — cannot drive a static-shape asset")
        }
        if largestContext != configuration.maxContextLength {
            CLILogger.log(
                "⚠️ Largest context bucket is \(largestContext) but config declares "
                    + "max_context_length \(configuration.maxContextLength)")
        }

        try Self.validateIOContract(descriptor: referenceDescriptor, contextBucket: largestContext)

        self.embeddingTable = try await Self.loadEmbeddingTable(from: model)

        // States: classified fixed vs per-bucket from the asset itself.
        self.states = try StaticStateFactory.makeStateSet(
            descriptorsByContext: descriptorsByContext,
            referenceDescriptor: referenceDescriptor,
            slidingWindow: configuration.overrides?.slidingWindow)

        // Inputs: the standard filler, plus whichever optional handlers the graph asks for.
        var handlers: [any StaticInputHandler] = []

        let positionIdsName = Self.resolveInputName(
            from: referenceDescriptor.inputNames, candidates: InputLayout.knownPositionIdNames)
        let causalMaskName =
            referenceDescriptor.inputNames.contains("causal_mask") ? "causal_mask" : nil
        let stepName = Self.resolveInputName(
            from: referenceDescriptor.inputNames, candidates: Self.knownStepNames)
        handlers.append(
            StaticBucketInputFiller(
                positionIdsName: positionIdsName,
                causalMaskName: causalMaskName,
                stepName: stepName,
                positionIds: positionIdsName.map {
                    BucketedInputDescriptors.collect($0, from: functionsByKey)
                } ?? .init([:]),
                causalMask: causalMaskName.map {
                    BucketedInputDescriptors.collect($0, from: functionsByKey)
                } ?? .init([:]),
                step: stepName.map {
                    BucketedInputDescriptors.collect($0, from: functionsByKey)
                } ?? .init([:])
            ))

        if referenceDescriptor.inputNames.contains(DualRoPEInputHandler.cosInputName) {
            guard let rope = configuration.overrides?.rope else {
                throw InferenceRuntimeError.invalidState(
                    "Graph declares '\(DualRoPEInputHandler.cosInputName)' but the bundle config "
                        + "has no `overrides.rope` block to build the table from")
            }
            CLILogger.log("Input handler: precomputed dual-RoPE rows")
            handlers.append(
                try DualRoPEInputHandler(
                    rope: rope,
                    cosDescriptors: .collect(DualRoPEInputHandler.cosInputName, from: functionsByKey),
                    sinDescriptors: .collect(DualRoPEInputHandler.sinInputName, from: functionsByKey)))
        }

        let slidingMask = BucketedInputDescriptors.collect(
            SlidingWindowInputHandler.maskInputName, from: functionsByKey)
        let slidingStep = BucketedInputDescriptors.collect(
            SlidingWindowInputHandler.stepInputName, from: functionsByKey)
        if !slidingMask.isEmpty || !slidingStep.isEmpty {
            guard let window = configuration.overrides?.slidingWindow else {
                throw InferenceRuntimeError.invalidState(
                    "Graph declares sliding-window inputs but the bundle config has no "
                        + "`overrides.sliding_window`")
            }
            // 0 when the asset has no sliding key cache, which the handler rejects.
            let ringDepth = states.slidingRing?.depth ?? 0
            CLILogger.log("Input handler: sliding window \(window), ring depth \(ringDepth)")
            handlers.append(
                try SlidingWindowInputHandler(
                    window: window, ringDepth: ringDepth,
                    maskDescriptors: slidingMask, stepDescriptors: slidingStep))
        }

        if referenceDescriptor.inputNames.contains(PerLayerEmbeddingsInputHandler.inputName) {
            guard let url = tensorData[EngineOptions.TensorDataKey.perLayerEmbeddings] else {
                throw InferenceRuntimeError.invalidState(
                    "Graph declares '\(PerLayerEmbeddingsInputHandler.inputName)' but no per-layer "
                        + "embeddings artifact was supplied. The bundle must declare it as "
                        + "`auxiliary_assets.\(EngineOptions.TensorDataKey.perLayerEmbeddings)` in metadata.json; "
                        + "EngineFactory.createEngine(bundle:) passes it through.")
            }
            let table = try PerLayerEmbeddings(contentsOf: url)
            // Token ids outside the table are skipped and gather zero rows, so a table
            // smaller than the vocabulary would degrade output silently.
            guard table.vocabSize >= configuration.vocabSize else {
                throw InferenceRuntimeError.invalidState(
                    "Per-layer embeddings table has \(table.vocabSize) rows, but the model's "
                        + "vocabulary is \(configuration.vocabSize)")
            }
            CLILogger.log(
                "Input handler: per-layer embeddings (vocab=\(table.vocabSize), "
                    + "rowWidth=\(table.rowWidth)) from \(url.lastPathComponent)")
            handlers.append(
                try PerLayerEmbeddingsInputHandler(
                    table: table,
                    descriptors: .collect(
                        PerLayerEmbeddingsInputHandler.inputName, from: functionsByKey)))
        }

        // Fail at load time if anything the graph declares has no handler, rather
        // than feeding it an unwritten buffer and producing NaN at runtime.
        var engineSupplied: Set<String> = [Self.embeddingTableName]
        if let transformerInput = Self.resolveInputName(
            from: referenceDescriptor.inputNames, candidates: Self.knownTransformerInputNames)
        {
            engineSupplied.insert(transformerInput)
        }
        for (_, descriptor) in functionsByKey {
            try StaticInputCoverage.verify(
                handlers: handlers, descriptor: descriptor, ignoring: engineSupplied)
        }

        self.inputHandlers = handlers
        var buffers = InputBuffers()
        for handler in handlers { handler.registerBuffers(into: &buffers) }
        self.inputBuffers = buffers

        CLILogger.log("Engine initialized")
    }

    public convenience init(
        configuration: ModelConfig, modelURL: URL, tensorData: [String: URL] = [:]
    ) async throws {
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        try await self.init(
            configuration: configuration,
            preparedModel: preparedModel,
            tensorData: tensorData)
    }

    /// Multimodal convenience init: main ladder at `modelURL` (+ its `tensorData` sidecars)
    /// plus a separate vision encoder asset. Used for `kind:vlm` bundles whose decoder
    /// is a Gemma4-style ladder (no separate `embedding` asset).
    public convenience init(
        configuration: ModelConfig,
        modelURL: URL,
        tensorData: [String: URL] = [:],
        visionModel: AIModel,
        visionConfig: VisionConfig
    ) async throws {
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        try await self.init(
            configuration: configuration,
            preparedModel: preparedModel,
            tensorData: tensorData,
            visionModel: visionModel,
            visionConfig: visionConfig)
    }

    /// Multimodal convenience init from URLs (prepares both assets internally). Lets
    /// callers wire vision without importing `PreparedModel`/`AIModel` (e.g. `CoreAIRunner`).
    public convenience init(
        configuration: ModelConfig,
        modelURL: URL,
        tensorData: [String: URL] = [:],
        visionModelURL: URL,
        visionConfig: VisionConfig
    ) async throws {
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        let preparedVision = try await PreparedModel.prepare(at: visionModelURL)
        try await self.init(
            configuration: configuration,
            preparedModel: preparedModel,
            tensorData: tensorData,
            visionModel: preparedVision.model,
            visionConfig: visionConfig)
    }

    // MARK: - Initialization Helpers

    /// Whether the states can be rewound to `target`, given the batch width `alignment` of the
    /// forward that restarts there (see ``SlidingRing/allowsRewind(processed:to:alignment:)``).
    private func canRewind(to target: Int, alignment: Int) -> Bool {
        states.canTruncate(processed: processedTokenCount, to: target, alignment: alignment)
    }

    private static func requireFunction(
        model: AIModel, functionName: String
    ) throws -> InferenceFunction {
        guard let fn = try model.loadFunction(named: functionName) else {
            throw InferenceRuntimeError.invalidState("Cannot load function '\(functionName)'")
        }
        return fn
    }

    private static func validateIOContract(
        descriptor: InferenceFunctionDescriptor, contextBucket: Int
    ) throws {
        guard descriptor.outputNames.contains(logitsOutputName) else {
            throw InferenceRuntimeError.invalidState(
                "The ctx \(contextBucket) function is missing required output '\(logitsOutputName)'. "
                    + "Available outputs: \(descriptor.outputNames)")
        }
        if descriptor.stateNames.count == 1 {
            throw InferenceRuntimeError.invalidState(
                "The ctx \(contextBucket) function has exactly 1 state (\(descriptor.stateNames)) "
                    + "— expected 0 (internal to model) or at least 2 (\(keyCacheName), \(valueCacheName))")
        }
        if descriptor.stateNames.count >= 2 {
            guard descriptor.stateNames.contains(keyCacheName),
                descriptor.stateNames.contains(valueCacheName)
            else {
                throw InferenceRuntimeError.invalidState(
                    "The ctx \(contextBucket) function has states \(descriptor.stateNames) "
                        + "but missing required '\(keyCacheName)' and/or '\(valueCacheName)'")
            }
        }
    }

    private static func loadEmbeddingTable(from model: AIModel) async throws -> NDArray {
        CLILogger.log("Loading embeddings...")
        guard let embeddingFunction = try model.loadFunction(named: "load_embeddings") else {
            throw InferenceRuntimeError.invalidState("Cannot load 'load_embeddings'")
        }

        guard case .ndArray(let embeddingDesc) = embeddingFunction.descriptor.outputDescriptor(of: embeddingTableName)
        else {
            throw InferenceRuntimeError.invalidState(
                "load_embeddings has no '\(embeddingTableName)' ndArray output descriptor")
        }
        var embeddingArray = NDArray(descriptor: embeddingDesc)

        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&embeddingArray, for: embeddingTableName)

        _ = try await embeddingFunction.run(
            inputs: [:],
            outputViews: consume outputViews
        )

        CLILogger.log("Embeddings loaded: shape=\(embeddingArray.shape)")
        return embeddingArray
    }

    // MARK: - Function Loading (lazy)

    private func loadFunction(named name: String) throws -> InferenceFunction {
        if let fn = functions[name] { return fn }
        guard let fn = try model.loadFunction(named: name) else {
            throw InferenceRuntimeError.invalidState("Cannot load function '\(name)'")
        }
        functions[name] = fn
        return fn
    }

    private func functionDescriptor(for name: String) throws -> InferenceFunctionDescriptor {
        if let fn = functions[name] { return fn.descriptor }
        guard let desc = model.functionDescriptor(for: name) else {
            throw InferenceRuntimeError.invalidState("Cannot find descriptor for '\(name)'")
        }
        return desc
    }

    /// Query length for a function: the sequence dimension of its
    /// `transformer_input`, falling back to the `_<qlen>` name suffix.
    private func queryLength(of functionName: String) throws -> Int {
        let desc = try functionDescriptor(for: functionName)
        if let txName = Self.resolveInputName(
            from: desc.inputNames, candidates: Self.knownTransformerInputNames),
            case .ndArray(let nd) = desc.inputDescriptor(of: txName), nd.shape.count >= 2
        {
            return nd.shape[1]
        }
        if let dims = Self.parseFunctionDimensions(functionName) { return dims.queryLength }
        return 1
    }

    /// Context bucket for a function, from its `<ctx>` name component. The name is
    /// authoritative: it is what selects the rung, and a state's shape need not
    /// have the context as its largest dimension (a merged dual-head-dim cache has
    /// more channels than context at the small buckets).
    private func contextLength(of functionName: String) throws -> Int {
        guard let dims = Self.parseFunctionDimensions(functionName) else {
            throw InferenceRuntimeError.invalidState(
                "Cannot parse a context bucket from function name '\(functionName)'")
        }
        return dims.contextLength
    }

    // MARK: - Graph Selection

    private func forwardGraph(numInputTokens: Int, currentPosition: Int, isPrefill: Bool) throws -> String {
        let (prefix, rungs) =
            isPrefill
            ? (Self.prefillFunctionPrefix, prefillRungs) : (Self.decodeFunctionPrefix, decodeRungs)
        guard let widest = rungs.map(\.queryLength).max() else {
            throw InferenceRuntimeError.invalidState(
                "No \(prefix)_<ctx>_<qlen> functions found in static-shape engine")
        }
        let selectedSeq = rungs.map(\.queryLength).filter { $0 >= numInputTokens }.min() ?? widest

        guard
            let selected =
                rungs
                .filter({ $0.queryLength == selectedSeq && $0.contextLength > currentPosition })
                .min(by: { $0.contextLength < $1.contextLength })
        else {
            throw InferenceRuntimeError.invalidState(
                "No \(prefix) graph with cache_len > \(currentPosition) and seq_len = \(selectedSeq)")
        }
        return "\(prefix)_\(selected.contextLength)_\(selected.queryLength)"
    }

    // MARK: - Generate (primary API)

    public func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        // Text-only entry: drop any image state left over from a prior multimodal
        // turn so a stale scatter can't corrupt this sequence.
        pendingImageEmbeddings = nil
        imageTokenPositions = []
        return try startGeneration(
            with: input,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions)
    }

    /// Number of leading tokens of `matchedPrefix` whose cached KV can be reused: the token-id
    /// match, cut at the first position that an image will be scattered into.
    static func reusablePrefixLength(matchedPrefix: Int, firstImagePosition: Int?) -> Int {
        guard let firstImagePosition else { return matchedPrefix }
        return Swift.min(matchedPrefix, firstImagePosition)
    }

    /// Shared generation setup (prefix-cache resolution + token/sequence creation), used
    /// by both the text-only `generate` above and the multimodal
    /// `generate(with: InputEmbeddings, ...)` (see `CoreAIStaticShapeEngine+Multimodal.swift`),
    /// which sets image state first and so must not go through the clearing entry point.
    func startGeneration(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> GenerationSequence {
        // Cancel any prior generation so its Iterator stops on next poll.
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }

        // Implicit prefix caching: resolve input against history.
        if history.count > 0 {
            let (matchedPrefix, _) = history.resolve(input: input)
            // Every image position carries the same placeholder id, so a token-id match says
            // nothing about image content: a pending image invalidates the cached prefix from
            // its first position on.
            let commonPrefix = Self.reusablePrefixLength(
                matchedPrefix: matchedPrefix,
                firstImagePosition: pendingImageEmbeddings == nil ? nil : imageTokenPositions.min())
            if let position = Self.resumePosition(
                commonPrefix: commonPrefix, inputCount: input.count, historyCount: history.count,
                processed: processedTokenCount,
                // The restart may run in any graph, the prefill graphs included.
                canRewind: { canRewind(to: $0, alignment: maxQueryLength) })
            {
                rewind(to: position)
            }
            lastPrefixHitCount = commonPrefix
        }

        let token = GenerationToken()
        _activeToken.withLock { $0 = token }
        return GenerationSequence(
            engine: self,
            input: input,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: inferenceOptions,
            generationToken: token
        )
    }

    // MARK: - Inference

    public func inference(
        inputTokens: [Int32], samplingConfig: SamplingConfiguration, returnsLogits: Bool,
        generationStartOffset: Int = 0, step: Int = 0
    ) async throws -> (logits: [LogitsScalarType]?, token: Int32) {
        CLILogger.log("Inference: \(inputTokens.count) tokens, processed: \(processedTokenCount)")

        let totalTokenCount = inputTokens.count
        guard processedTokenCount < totalTokenCount else {
            throw InferenceRuntimeError.invalidState("No new tokens to process")
        }

        var logitBuffer = [LogitsScalarType](repeating: 0, count: config.vocabSize)
        var currentPosition = processedTokenCount

        while currentPosition < totalTokenCount {
            let remaining = totalTokenCount - currentPosition
            let usePrefill = remaining > maxQueryLength
            let graphName = try forwardGraph(
                numInputTokens: remaining, currentPosition: currentPosition, isPrefill: usePrefill)
            let contextBucket = try contextLength(of: graphName)

            // Lay out per-bucket states for this rung before binding.
            try states.prepare(
                contextBucket: contextBucket, writtenTokenCount: processedTokenCount)

            let batchSize = try queryLength(of: graphName)
            let batchStartToken = (currentPosition / batchSize) * batchSize
            let batchEndToken = min(batchStartToken + batchSize - 1, totalTokenCount - 1)
            let tokensInBatch = batchEndToken - batchStartToken + 1

            CLILogger.log("Graph: \(graphName), batch=\(batchSize), step=\(batchStartToken), tokens=\(tokensInBatch)")

            let prepareSpan = InstrumentsProfiler.beginPrepareStep(
                operation: "buildInputs", engine: "StaticShape")
            let inputs = try await buildInputs(
                graphName: graphName,
                batchTokens: inputTokens[batchStartToken...batchEndToken],
                batchSize: batchSize,
                alignedStep: batchStartToken,
                contextBucket: contextBucket
            )
            prepareSpan.end()

            let logitsSpan = InstrumentsProfiler.beginLogitsInference(
                step: batchStartToken, tokens: tokensInBatch, engine: "StaticShape")

            let fn = try loadFunction(named: graphName)
            let desc = try functionDescriptor(for: graphName)

            var outputs = try await runStaticStep(
                function: fn, descriptor: desc, inputs: inputs, states: states)

            let logitsArray = outputs.remove(Self.logitsOutputName)?.ndArray
            logitsSpan.end()

            // Extract logits from the last token position.
            if !usePrefill, let logitsArray {
                let logitsView = logitsArray.view(as: LogitsScalarType.self)
                guard let logits = logitsView.contiguousElements else {
                    throw InferenceRuntimeError.invalidState(
                        "Logits array has non-contiguous (interleaved) layout — cannot extract values safely")
                }
                let copySpan = InstrumentsProfiler.beginLogitsCopy()
                // One bulk copy of the last token's row rather than a per-element loop.
                let vocabSize = config.vocabSize
                let offset = (tokensInBatch - 1) * vocabSize
                logits.withUnsafeBufferPointer { source in
                    precondition(source.count >= offset + vocabSize, "Logits output shorter than one vocab row")
                    logitBuffer.withUnsafeMutableBufferPointer { destination in
                        destination.baseAddress!.update(from: source.baseAddress! + offset, count: vocabSize)
                    }
                }
                copySpan.end()
            }

            currentPosition = batchEndToken + 1
            processedTokenCount = currentPosition
        }

        // Final-logit soft cap. The iOS export leaves `c · tanh(logits / c)` out of the
        // graph (tanh is best run on the CPU rather than in the graph), so apply it here
        // — to the returned logits, and through the sampling pipeline before the sampler, so
        // parity dumps and the sampled token see the same capped values the reference
        // implementation produces.
        let softcap = config.overrides?.finalLogitSoftcapping
        var actualLogits = returnsLogits ? logitBuffer : nil
        if let softcap, actualLogits != nil {
            LogitSoftcapProcessor.apply(to: &actualLogits!, cap: Float(softcap))
        }
        let sampleSpan = InstrumentsProfiler.beginSample(strategy: "cpu-fallback")
        let nextToken = samplingConfig.fallbackSampler(
            from: &logitBuffer, tokenHistory: inputTokens[generationStartOffset...], step: step,
            logitSoftcap: softcap)
        sampleSpan.end()
        CLILogger.log("Token: \(nextToken), processed: \(processedTokenCount)")
        return (logits: actualLogits, token: nextToken)
    }

    // MARK: - Speculative decoding support (per-position results for verification)

    /// Full token sequence seen by the speculative primitives, so batch inputs can be
    /// sliced at batch-aligned boundaries the way `inference` does.
    private var specSequence: [Int32] = []

    /// The speculative buffer holds exactly the tokens the cache has processed. A mismatch means
    /// tokens were processed or rewound outside the speculative primitives, and slicing batches
    /// from the buffer would skip or replay positions.
    static func validateSpeculativeSequence(count: Int, processed: Int) throws {
        guard count == processed else {
            throw InferenceRuntimeError.invalidState(
                "speculative sequence holds \(count) tokens but \(processed) are processed; "
                    + "reset(to: 0) and replay the prefix before speculative decoding")
        }
    }

    /// Prefill `tokens` at the current position, discarding logits.
    package func advance(_ tokens: [Int32]) async throws {
        _ = try await processSpeculativeBlock(tokens, capture: .none)
    }

    /// Forward `tokens` in one pass; per-position `[vocabSize]` logits predicting the
    /// following token. This is the verify primitive the speculative decoder needs: the
    /// static-shape graph already computes every position's logits (see `inference`), so
    /// verification of a draft block runs on the same engine.
    package func forwardWithPerPositionLogits(_ tokens: [Int32]) async throws -> [[LogitsScalarType]] {
        try await processSpeculativeBlock(tokens, capture: .fullLogits).logits
    }

    /// Forward `tokens` in one pass; per-position argmax (greedy) token.
    ///
    /// Computes the argmax directly from the contiguous logits buffer instead of
    /// materializing a `[vocabSize]` row per position: with a 262K vocab, that row copy
    /// (times every position times the softcap pass) dominated the verify cost. The result is
    /// the token plain decoding picks, softcap included: the cap rounds near-equal logits to the
    /// same value, and a tie goes to the lowest index. Only entries that can reach the maximum
    /// after capping are capped.
    package func forwardPerPositionArgmax(_ tokens: [Int32]) async throws -> [Int32] {
        try await processSpeculativeBlock(tokens, capture: .argmax).argmax
    }

    /// The `count` highest entries of one position's logits, highest first, as plain decoding
    /// ranks them: the final-logit softcap is applied the way `inference` applies it, rounding to
    /// `LogitsScalarType` included, but only to the entries that can tie or beat the maximum
    /// (see ``LogitRowScan``). The row is widened into `scratch` with a vectorized conversion.
    static func scanRow(
        _ row: UnsafePointer<LogitsScalarType>, vocab: Int, count: Int, softcap: Float?,
        scratch: inout [Float]
    ) -> (indices: [Int32], values: [Float]) {
        if scratch.count != vocab { scratch = [Float](repeating: 0, count: vocab) }
        return scratch.withUnsafeMutableBufferPointer { dst in
            var src = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: row),
                height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * 2)
            var dstBuf = vImage_Buffer(
                data: dst.baseAddress, height: 1,
                width: vImagePixelCount(vocab), rowBytes: vocab * 4)
            _ = vImageConvert_Planar16FtoPlanarF(&src, &dstBuf, 0)
            return LogitRowScan.topK(dst, k: count, cap: softcap)
        }
    }

    /// What `processSpeculativeBlock` extracts from each verified position.
    private enum SpecCapture: Equatable {
        case none  // advance state only (no logits read back)
        case argmax  // greedy: per-position argmax token (no full-vocab copy)
        case fullLogits  // sampling: per-position `[vocabSize]` distribution
    }

    private struct SpecCaptureResult {
        var logits: [[LogitsScalarType]] = []
        var argmax: [Int32] = []
    }

    /// Runs the delta `tokens` through the static-shape graph at the current position and
    /// returns per-position results (when `capture`). Mirrors `inference`'s batch loop but
    /// keeps every position and does no sampling or prefix-cache resolution: the decoder
    /// owns rollback via `reset(to:)`.
    @discardableResult
    private func processSpeculativeBlock(
        _ tokens: [Int32], capture: SpecCapture
    ) async throws -> SpecCaptureResult {
        guard !tokens.isEmpty else { return SpecCaptureResult() }
        // A block longer than the widest graph starts in a prefill graph, which returns no logits,
        // so it would yield fewer rows than tokens with the first rows missing.
        if capture != .none, tokens.count > maxQueryLength {
            throw InferenceRuntimeError.invalidArgument(
                "a verify block of \(tokens.count) tokens exceeds the widest graph "
                    + "(\(maxQueryLength) tokens)")
        }
        try Self.validateSpeculativeSequence(count: specSequence.count, processed: processedTokenCount)
        let captureFrom = processedTokenCount
        specSequence.append(contentsOf: tokens)
        let total = specSequence.count
        var result = SpecCaptureResult()
        var currentPosition = processedTokenCount

        while currentPosition < total {
            let remaining = total - currentPosition
            let usePrefill = remaining > maxQueryLength
            let graphName = try forwardGraph(
                numInputTokens: remaining, currentPosition: currentPosition, isPrefill: usePrefill)
            let contextBucket = try contextLength(of: graphName)
            try states.prepare(contextBucket: contextBucket, writtenTokenCount: processedTokenCount)

            let batchSize = try queryLength(of: graphName)
            // Batches must start on a batchSize boundary: the static graph's KV write
            // is compiled for a batchSize-aligned write offset, and a non-aligned
            // offset is rejected by the palettized (int4) program at runtime
            // ("Request cancelled"). So a verify block may span two batches (a second
            // forward) when it straddles a boundary: correct, and the dominant verify
            // cost is the forward, not the re-scanned committed positions.
            let batchStartToken = (currentPosition / batchSize) * batchSize
            let batchEndToken = min(batchStartToken + batchSize - 1, total - 1)
            let tokensInBatch = batchEndToken - batchStartToken + 1

            let inputs = try await buildInputs(
                graphName: graphName,
                batchTokens: specSequence[batchStartToken...batchEndToken],
                batchSize: batchSize,
                alignedStep: batchStartToken,
                contextBucket: contextBucket)

            let fn = try loadFunction(named: graphName)
            let desc = try functionDescriptor(for: graphName)
            var outputs = try await runStaticStep(
                function: fn, descriptor: desc, inputs: inputs, states: states)
            let logitsArray = outputs.remove(Self.logitsOutputName)?.ndArray

            if case .none = capture {
                // advance only: nothing to read back
            } else if !usePrefill, let logitsArray {
                let vocab = config.vocabSize
                // Read via the raw pointer: `contiguousElements` materializes an Array copy of the
                // whole batch (batchSize x vocab), which for a 262K vocab dominates verify cost.
                // The pointer path scans only the rows we keep. The argmax is vectorized through
                // Accelerate: a scalar Swift loop over 262K Float16 was ~15 ms per position and, at
                // k+1 positions per verify, cancelled the acceptance speedup.
                var scratch: [Float] = capture == .argmax ? [Float](repeating: 0, count: vocab) : []
                let softcap = config.overrides?.finalLogitSoftcapping.map { Float($0) }
                logitsArray.view(as: LogitsScalarType.self).withUnsafePointer { ptr, _, _ in
                    for local in 0..<tokensInBatch {
                        let absolutePosition = batchStartToken + local
                        guard absolutePosition >= captureFrom else { continue }
                        let offset = local * vocab
                        switch capture {
                        case .none:
                            break
                        case .argmax:
                            let top = Self.scanRow(
                                ptr + offset, vocab: vocab, count: 1, softcap: softcap, scratch: &scratch)
                            result.argmax.append(top.indices.first ?? 0)
                        case .fullLogits:
                            var row = [LogitsScalarType](repeating: 0, count: vocab)
                            for i in 0..<vocab { row[i] = ptr[offset + i] }
                            if let cap = config.overrides?.finalLogitSoftcapping {
                                LogitSoftcapProcessor.apply(to: &row, cap: Float(cap))
                            }
                            result.logits.append(row)
                        }
                    }
                }
            }

            currentPosition = batchEndToken + 1
            processedTokenCount = currentPosition
        }
        return result
    }

    /// Longest block the per-position forwards accept and the decoder can roll back from. The
    /// widest graph bounds it: a longer block would run through prefill graphs that capture no
    /// logits. The sliding-window ring bounds it further: rejecting every draft must rewind to
    /// the anchor, which the ring allows only for blocks up to `SlidingRing.maxVerifyBlockLength`.
    package var maxVerifyBlockLength: Int {
        min(
            maxQueryLength,
            states.slidingRing?.maxVerifyBlockLength(alignment: verifyTileWidth ?? 1) ?? Int.max)
    }

    /// Verify batches start on a multiple of the decode width (see `processSpeculativeBlock`): the
    /// widest `extend_*` rung, not `maxQueryLength`, which includes the prefill rungs.
    package var verifyTileWidth: Int? {
        extendFunctionNames.filter { $0.hasPrefix("extend") }
            .compactMap { Self.parseFunctionDimensions($0)?.queryLength }.max()
    }

    /// Query length of the decode graph that runs a block starting at the current position,
    /// which is the batch window a verify block must fit to take one forward. This is not the
    /// largest query length across all graphs, which includes the prefill graphs.
    package var decodeBlockSize: Int? {
        try? queryLength(
            of: forwardGraph(numInputTokens: 1, currentPosition: processedTokenCount, isPrefill: false))
    }

    /// Proposal tokens that fit with the anchor in one forward of the decode graph: the
    /// anchor sits at the current position, and a block that crosses the next batch-size
    /// boundary needs a second forward.
    package var oneForwardDraftBudget: Int {
        Self.batchAlignedDraftBudget(
            batchSize: decodeBlockSize ?? maxQueryLength, anchorPosition: processedTokenCount)
    }

    /// `batchSize - 1 - (anchorPosition mod batchSize)`: the slots left in the anchor's batch
    /// window after the anchor itself.
    static func batchAlignedDraftBudget(batchSize: Int, anchorPosition: Int) -> Int {
        guard batchSize > 0 else { return 0 }
        return batchSize - 1 - (anchorPosition % batchSize)
    }

    // MARK: - Inference Helpers

    private func buildInputs<Tokens: Collection<Int32>>(
        graphName: String,
        batchTokens: Tokens,
        batchSize: Int,
        alignedStep: Int,
        contextBucket: Int
    ) async throws -> [String: NDArray] {
        let desc = try functionDescriptor(for: graphName)

        let context = InputContext.static(
            tokens: ArraySlice(batchTokens),
            alignedStep: alignedStep,
            batchSize: batchSize,
            slidingWindow: nil,
            contextBucket: contextBucket)

        // Multimodal: HF masks image-placeholder positions to the pad token BEFORE the
        // per-layer-embedding gather (`llm_input_ids = where(image_mask, pad_token_id,
        // input_ids)`). The image rows carry vision soft tokens, not a real token, so
        // gathering the placeholder id's own per-layer embedding would inject a spurious
        // per-layer signal at every layer and corrupt the image representation. Only the
        // PLE handler needs this correction — the raw embedding-table gather below is
        // harmless waste for those rows since the scatter overwrites them afterward.
        let pleContext: InputContext
        if let imgId = visionConfig?.imageTokenId, pendingImageEmbeddings != nil {
            let padId = visionConfig?.padTokenId ?? 0
            let pleTokens = batchTokens.map { $0 == imgId ? padId : $0 }
            pleContext = InputContext.static(
                tokens: ArraySlice(pleTokens),
                alignedStep: alignedStep,
                batchSize: batchSize,
                slidingWindow: config.overrides?.slidingWindow,
                contextBucket: contextBucket)
        } else {
            pleContext = context
        }

        for handler in inputHandlers {
            if handler is PerLayerEmbeddingsInputHandler {
                try handler.fill(pleContext, into: &inputBuffers)
            } else {
                try handler.fill(context, into: &inputBuffers)
            }
        }
        var inputs = inputBuffers.borrowedInputs()

        // Pass-through constant embedding table
        if desc.inputNames.contains(Self.embeddingTableName) {
            inputs[Self.embeddingTableName] = embeddingTable
        }

        // Gather embeddings for this batch's tokens
        if let txName = Self.resolveInputName(from: desc.inputNames, candidates: Self.knownTransformerInputNames) {
            let gatherName = "gather_embeddings_\(batchSize)"
            guard gatherFunctionNames.contains(gatherName) else {
                throw InferenceRuntimeError.invalidState(
                    "No gather function '\(gatherName)' for batch size \(batchSize)")
            }
            let gatherSpan = InstrumentsProfiler.beginGatherEmbeddings()
            let gatherResult = try await runGather(tokenIDs: Array(batchTokens), batchSize: batchSize)
            gatherSpan.end()
            guard var gathered = gatherResult else {
                throw InferenceRuntimeError.invalidState("Gather '\(gatherName)' returned no output")
            }
            // Multimodal: overwrite image-placeholder rows in this batch with the encoded
            // image soft tokens before they enter the decoder.
            try scatterImageEmbeddings(into: &gathered, alignedStep: alignedStep, batchSize: batchSize)
            inputs[txName] = gathered
        }

        return inputs
    }

    // MARK: - Gather Embeddings

    private func runGather(tokenIDs: [Int32], batchSize: Int) async throws -> NDArray? {
        let name = "gather_embeddings_\(batchSize)"
        let fn = try loadFunction(named: name)
        let desc = try functionDescriptor(for: name)

        guard let tokenDesc = desc.inputDescriptor(of: Self.tokenIDsInputName),
            case .ndArray(let tokenNDDesc) = tokenDesc
        else {
            throw InferenceRuntimeError.invalidState("No descriptor for '\(Self.tokenIDsInputName)'")
        }

        var tokenArray = NDArray(descriptor: tokenNDDesc)
        let tokenView = tokenArray.mutableView(as: Int32.self)
        guard var tokenSpan = tokenView.contiguousElements else {
            throw InferenceRuntimeError.invalidState("tokenArray has non-contiguous layout")
        }
        // Zero unused (padding) query slots first: a partial final batch leaves
        // slots [tokensInBatch..<batchSize] otherwise uninitialized, so they would
        // gather a garbage token id → garbage query embedding. Padding with token 0
        // keeps the discarded columns finite (garbage could feed NaN into shared
        // reductions).
        for i in 0..<tokenSpan.count { tokenSpan[i] = 0 }
        if tokenNDDesc.shape.count == 2 {
            for i in 0..<min(batchSize, tokenIDs.count) {
                tokenSpan[i] = tokenIDs[i]
            }
        } else {
            tokenSpan[0] = tokenIDs[0]
        }

        var inputs: [String: NDArray] = [Self.tokenIDsInputName: tokenArray]
        inputs[Self.embeddingTableName] = embeddingTable

        var outputs = try await fn.run(
            inputs: inputs,
            outputViews: InferenceFunction.MutableViews()
        )

        return outputs.remove(Self.gatheredOutputName)?.ndArray
            ?? outputs.remove(desc.outputNames.first ?? "")?.ndArray
    }

    // MARK: - Lifecycle

    public func cancel() async throws {
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }
        pendingImageEmbeddings = nil
        imageTokenPositions = []
    }

    /// Throws unless `tokenIndex` names a position the engine has processed (`0...processed`).
    static func validateResetTarget(_ tokenIndex: Int, processed: Int) throws {
        guard tokenIndex >= 0, tokenIndex <= processed else {
            throw InferenceRuntimeError.invalidArgument(
                "reset(to: \(tokenIndex)) out of range [0, \(processed)]")
        }
    }

    /// Rewinds to `tokenIndex` processed tokens. Throws, leaving the engine unchanged, when
    /// the index is out of range or the sliding-window ring no longer holds the keys the
    /// restarted forward needs.
    public func reset(to tokenIndex: Int) async throws {
        try Self.validateResetTarget(tokenIndex, processed: processedTokenCount)
        guard canRewind(to: tokenIndex, alignment: verifyTileWidth ?? 1) else {
            throw InferenceRuntimeError.invalidState(
                "reset(to: \(tokenIndex)) needs keys the sliding-window ring has overwritten "
                    + "(\(processedTokenCount) tokens processed). Use reset(to: 0) and replay "
                    + "the prefix.")
        }
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }
        pendingImageEmbeddings = nil
        imageTokenPositions = []
        let resetSpan = InstrumentsProfiler.beginReset(engine: "StaticShape")
        rewind(to: tokenIndex)
        resetSpan.end()
    }

    /// Where a request resumes the cached sequence, or nil to continue from
    /// `processed` as is.
    ///
    /// - A request that diverges from the history restarts at 0.
    /// - One that the history already covers rewinds one token before the common
    ///   prefix, to re-run it for the next token's logits; or restarts at 0 when a
    ///   sliding-window ring no longer holds the keys that rewind needs.
    static func resumePosition(
        commonPrefix: Int, inputCount: Int, historyCount: Int, processed: Int,
        canRewind: (Int) -> Bool
    ) -> Int? {
        if commonPrefix < inputCount && commonPrefix < historyCount {
            return 0
        }
        guard processed >= inputCount else { return nil }
        let target = Swift.max(0, commonPrefix - 1)
        return canRewind(target) ? target : 0
    }

    /// Moves the cursor back to `position`. The one path every restart and rewind goes
    /// through, so a restart always zeroes the states.
    private func rewind(to position: Int) {
        if position == 0 {
            processedTokenCount = 0
            history.clear()
            // Same-bucket restarts reuse storage (prepare() early-returns), so
            // zero it; see StaticStateStorage.reset().
            states.reset()
        } else {
            processedTokenCount = position
            history.truncate(to: position)
        }
        if specSequence.count > position {
            specSequence.removeLast(specSequence.count - position)
        }
    }

    public func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {
        for fnName in extendFunctionNames {
            self.functions[fnName] = try Self.requireFunction(model: model, functionName: fnName)
        }
        try await reset()
    }
}

extension StaticShapeEngine {
    /// Async sequence of `InferenceOutput` produced by `generate()`.
    ///
    /// Iteration is structured: state lives on the iterator and releases naturally
    /// when iteration ends or the iterator is dropped (covering early break / task
    /// cancellation).
    public struct GenerationSequence: InferenceOutputSequence {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        let engine: StaticShapeEngine
        let input: [TokenId]
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
                samplingConfiguration: samplingConfiguration,
                inferenceOptions: inferenceOptions,
                stopReasonStore: stopReasonStore,
                generationToken: generationToken
            )
        }
    }
}

extension StaticShapeEngine.GenerationSequence {
    public struct Iterator: AsyncIteratorProtocol {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        private let engine: StaticShapeEngine
        private let samplingConfiguration: SamplingConfiguration
        private let returnsLogits: Bool
        private let forcedContinuation: [StaticShapeEngine.TokenId]?
        private let maxTokens: Int
        private let stopReasonStore: StopReasonStore
        private let generationToken: GenerationToken

        private var inputTokens: [StaticShapeEngine.TokenId]
        private let generationStartOffset: Int
        private var step: Int = 0
        private var finished: Bool = false

        init(
            engine: StaticShapeEngine,
            input: [StaticShapeEngine.TokenId],
            samplingConfiguration: SamplingConfiguration,
            inferenceOptions: InferenceOptions,
            stopReasonStore: StopReasonStore,
            generationToken: GenerationToken
        ) {
            self.engine = engine
            self.samplingConfiguration = samplingConfiguration
            self.returnsLogits = inferenceOptions.includeLogits
            self.forcedContinuation = inferenceOptions.forcedContinuation
            self.stopReasonStore = stopReasonStore
            self.generationToken = generationToken
            self.inputTokens = input
            self.generationStartOffset = input.count
            if let forced = inferenceOptions.forcedContinuation {
                self.maxTokens = forced.count
            } else {
                self.maxTokens = Swift.min(
                    inferenceOptions.maxTokens ?? Int.max,
                    Swift.max(0, engine.config.maxContextLength - input.count)
                )
            }
        }

        public mutating func next() async throws -> InferenceOutput? {
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

            do {
                try Task.checkCancellation()

                let oldProcessedCount = engine.processedTokenCount

                // When forced, we still need the forward pass (for logits + KV cache update)
                // but skip the sampler — the next token is predetermined.
                let (logits, sampledToken) = try await engine.inference(
                    inputTokens: inputTokens,
                    samplingConfig: samplingConfiguration,
                    returnsLogits: returnsLogits || forcedContinuation != nil,
                    generationStartOffset: generationStartOffset,
                    step: step
                )

                // Update history with newly processed tokens
                let processedSlice = inputTokens[oldProcessedCount..<engine.processedTokenCount]
                engine.history.append(contentsOf: processedSlice)

                // Check cancellation after inference step
                if generationToken.isCancelled {
                    stopReasonStore.set(.cancelled)
                    finishAndRelease()
                    return nil
                }

                let nextToken = forcedContinuation?[step] ?? sampledToken
                inputTokens.append(nextToken)
                step += 1

                return InferenceOutput(
                    tokenId: nextToken,
                    logits: returnsLogits ? logits : nil
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

        private mutating func finishAndRelease() {
            guard !finished else { return }
            finished = true
            engine.clearTokenIfActive(generationToken)
        }
    }
}

// The speculative primitives above are what the decoder verifies with on this engine.
extension StaticShapeEngine: SpeculativeEngine {}
