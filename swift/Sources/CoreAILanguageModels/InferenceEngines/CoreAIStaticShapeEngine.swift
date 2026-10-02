// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

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

    /// Clear the engine's active token if it matches the given token.
    func clearTokenIfActive(_ token: GenerationToken) {
        _activeToken.withLock { if $0 === token { $0 = nil } }
    }

    // MARK: - Initialization

    /// - Parameter tensorData: Tensor data the graph reads alongside the model, keyed by
    ///   `assets` role (see ``EngineOptions/TensorDataKey``). Only the roles the graph's
    ///   inputs need are looked up.
    public init(
        configuration: ModelConfig,
        preparedModel: PreparedModel,
        tensorData: [String: URL] = [:]
    ) async throws {
        self.config = configuration
        self.model = preparedModel.model
        self.functions = [:]

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
                        + "`assets.\(EngineOptions.TensorDataKey.perLayerEmbeddings)` in metadata.json; "
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

    // MARK: - Initialization Helpers

    private func canRewind(to target: Int) -> Bool {
        states.canTruncate(processed: processedTokenCount, to: target)
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
        // Cancel any prior generation so its Iterator stops on next poll.
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }

        // Implicit prefix caching: resolve input against history.
        if history.count > 0 {
            let (commonPrefix, _) = history.resolve(input: input)
            if let position = Self.resumePosition(
                commonPrefix: commonPrefix, inputCount: input.count, historyCount: history.count,
                processed: processedTokenCount, canRewind: canRewind(to:))
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
        for handler in inputHandlers {
            try handler.fill(context, into: &inputBuffers)
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
            let gathered = try await runGather(tokenIDs: Array(batchTokens), batchSize: batchSize)
            gatherSpan.end()
            guard let gathered else {
                throw InferenceRuntimeError.invalidState("Gather '\(gatherName)' returned no output")
            }
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
    }

    public func reset(to tokenIndex: Int) async throws {
        precondition(
            tokenIndex >= 0 && tokenIndex <= processedTokenCount,
            "reset(to: \(tokenIndex)) out of range [0, \(processedTokenCount)]")
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }
        guard canRewind(to: tokenIndex) else {
            throw InferenceRuntimeError.invalidState(
                "reset(to: \(tokenIndex)) needs keys the sliding-window ring has overwritten "
                    + "(\(processedTokenCount) tokens processed). Use reset(to: 0) and replay "
                    + "the prefix.")
        }
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
