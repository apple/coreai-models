// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Synchronization

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
/// - **2 states**: `keyCache`, `valueCache` — persistent across steps, updated in-place
///
/// KV cache NDArrays start small (256 tokens) and grow dynamically with 2× expansion.
/// Passed as `states` on every forward pass; the model graph updates them in-place.
public final class CoreAISequentialEngine: InferenceEngine, @unchecked Sendable {
    public typealias ConfigType = ModelConfig

    public var supportsLogits: Bool { true }
    public var vocabSize: Int { config.vocabSize }
    public let config: ModelConfig

    // Core AI function handle
    private let function: InferenceFunction
    private let functionDescriptor: InferenceFunctionDescriptor

    // I/O names from descriptor
    private let inputIdsName: String
    private let positionIdsName: String
    private let keyCacheName: String
    private let valueCacheName: String
    private let logitsName: String

    // Descriptors for dynamic shape resolution
    private let inputIdsDescriptor: NDArrayDescriptor
    private let positionIdsDescriptor: NDArrayDescriptor
    private let logitsDescriptor: NDArrayDescriptor

    // Persistent state — reused across steps
    private var keyCache: NDArray
    private var valueCache: NDArray
    private var logitsArray: NDArray
    // Pre-allocated input_ids reused across decode steps. Only reallocated when
    // batch size changes (i.e., once when transitioning from prefill to decode).
    // Saves ~50-100 µs/step worth of `NDArray(descriptor:)` + descriptor resolve
    // work in the steady state.
    private var inputIdsArray: NDArray
    private var cachedInputBatchSize: Int
    private var cachedLogitsBatchSize: Int
    private var currentKVCapacity: Int
    private let keyCacheDescriptor: NDArrayDescriptor
    private let valueCacheDescriptor: NDArrayDescriptor

    // Track processed tokens for incremental inference
    public private(set) var processedTokenCount: Int = 0

    // Token history for implicit prefix caching
    private var history = TokenHistory()
    public private(set) var lastPrefixHitCount: Int = 0

    // Track in-flight generation via token (replaces simple bool lock)
    private let _activeToken = Mutex<GenerationToken?>(nil)

    public var isBusy: Bool { _activeToken.withLock { $0 != nil } }

    /// Clear the engine's active token if it matches the given token.
    /// Called by the iterator when generation finishes or is cancelled.
    func clearTokenIfActive(_ token: GenerationToken) {
        _activeToken.withLock { if $0 === token { $0 = nil } }
    }

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

        // Validate model architecture: 2 inputs, 1+ output, 2 states
        guard descriptor.inputNames.count == 2 else {
            throw InferenceRuntimeError.invalidInputType(
                "Expected 2 inputs, got \(descriptor.inputNames.count): \(descriptor.inputNames)")
        }
        guard descriptor.outputNames.count >= 1 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 1 output, got \(descriptor.outputNames.count): \(descriptor.outputNames)")
        }
        guard descriptor.stateNames.count == 2 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected 2 states (KV cache), got \(descriptor.stateNames.count): "
                    + "states=\(descriptor.stateNames), outputs=\(descriptor.outputNames)")
        }

        // Extract names
        self.inputIdsName = descriptor.inputNames[0]
        self.positionIdsName = descriptor.inputNames[1]
        self.keyCacheName = descriptor.stateNames[0]
        self.valueCacheName = descriptor.stateNames[1]
        self.logitsName = descriptor.outputNames[0]

        // Extract and validate input descriptors
        guard case .ndArray(let inputIdsDesc) = descriptor.inputDescriptor(of: inputIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(inputIdsName)'")
        }
        self.inputIdsDescriptor = inputIdsDesc

        guard case .ndArray(let posIdsDesc) = descriptor.inputDescriptor(of: positionIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(positionIdsName)'")
        }
        self.positionIdsDescriptor = posIdsDesc

        // Extract and validate logits descriptor
        guard case .ndArray(let logitsDesc) = descriptor.outputDescriptor(of: logitsName) else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get descriptor for '\(logitsName)'")
        }
        guard logitsDesc.scalarType == .float16 else {
            throw InferenceRuntimeError.unsupportedLogitsType(
                "Only float16 logits supported, got \(logitsDesc.scalarType)")
        }
        self.logitsDescriptor = logitsDesc

        // Extract KV cache state descriptors
        guard case .ndArray(let keyCacheDesc) = descriptor.stateDescriptor(of: keyCacheName),
            case .ndArray(let valueCacheDesc) = descriptor.stateDescriptor(of: valueCacheName)
        else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get KV cache state descriptors")
        }

        // Store unresolved descriptors for dynamic reallocation
        self.keyCacheDescriptor = keyCacheDesc
        self.valueCacheDescriptor = valueCacheDesc

        let isDynamic = keyCacheDesc.shape.contains(where: { $0 < 0 })

        // Allocate KV cache at initial size (grow on demand unless fixedSize requested)
        let initialCapacity: Int
        if options.kvCacheStrategy == .fixedSize || !isDynamic {
            initialCapacity = config.maxContextLength
        } else {
            initialCapacity = min(256, config.maxContextLength)
        }
        self.currentKVCapacity = initialCapacity
        let resolvedKeyDesc = keyCacheDesc.resolvingDynamicDimensions(
            keyCacheDesc.shape.map { $0 < 0 ? initialCapacity : $0 })
        let resolvedValueDesc = valueCacheDesc.resolvingDynamicDimensions(
            valueCacheDesc.shape.map { $0 < 0 ? initialCapacity : $0 })
        self.keyCache = NDArray(descriptor: resolvedKeyDesc)
        self.valueCache = NDArray(descriptor: resolvedValueDesc)

        CLILogger.log(
            "KV cache: dynamic=\(isDynamic), initial=\(initialCapacity), key=\(keyCacheDesc.shape) → \(resolvedKeyDesc.shape)"
        )

        // Allocate initial logits (1 token — will be reallocated per batch)
        let initLogitsDesc = logitsDesc.resolvingDynamicDimensions([1, 1, config.vocabSize])
        self.logitsArray = NDArray(descriptor: initLogitsDesc)
        self.cachedLogitsBatchSize = 1

        // Allocate initial input_ids ([1, 1] — decode steady state). Will be
        // reallocated on first prefill if batch != 1.
        let initInputDesc = inputIdsDesc.resolvingDynamicDimensions([1, 1])
        self.inputIdsArray = NDArray(descriptor: initInputDesc)
        self.cachedInputBatchSize = 1

        // Load inference function
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
        if newTokenCount > config.chunkThreshold {
            return .chunked(chunkSize: config.prefillChunkSize)
        }
        return .wholeBatch
    }

    // MARK: - Token Batch Processing

    /// Process a batch of tokens in a single forward pass.
    private func processTokenBatch(_ tokens: ArraySlice<Int32>) async throws -> [LogitsScalarType] {
        let batchSize = tokens.count
        guard batchSize > 0 else {
            throw InferenceRuntimeError.invalidState("Cannot process empty token batch")
        }

        try ensureKVCapacity(forContextLength: processedTokenCount + batchSize)

        let batchSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIClean Batch",
            details: "\(batchSize) tokens at position \(processedTokenCount)"
        )

        // Reuse pre-allocated input_ids when the batch size is unchanged.
        // Steady-state decode keeps batchSize=1 forever, so this avoids the
        // `NDArray(descriptor:)` + `resolvingDynamicDimensions` work on every
        // step — small per call, but compounds over long generations.
        if cachedInputBatchSize != batchSize {
            let resolvedInputDesc = inputIdsDescriptor.resolvingDynamicDimensions([1, batchSize])
            inputIdsArray = NDArray(descriptor: resolvedInputDesc)
            cachedInputBatchSize = batchSize
        }
        fillNDArray(&inputIdsArray, as: Int32.self, with: tokens)

        // Build position_ids: [0, 1, ..., processedTokenCount + batchSize - 1]
        // Shape grows by 1 each step, so we can't easily pre-allocate this one.
        let totalPositions = processedTokenCount + batchSize
        let resolvedPosDesc = positionIdsDescriptor.resolvingDynamicDimensions([1, totalPositions])
        var positionIds = NDArray(descriptor: resolvedPosDesc)
        fillNDArray(&positionIds, as: Int32.self, count: totalPositions) { Int32($0) }

        // Reuse pre-allocated logits when the batch size is unchanged.
        if cachedLogitsBatchSize != batchSize {
            let resolvedLogitsDesc = logitsDescriptor.resolvingDynamicDimensions([1, batchSize, config.vocabSize])
            logitsArray = NDArray(descriptor: resolvedLogitsDesc)
            cachedLogitsBatchSize = batchSize
        }

        // Build states (KV cache — persistent, inout)
        var states = InferenceFunction.MutableViews()
        states.insert(&keyCache, for: keyCacheName)
        states.insert(&valueCache, for: valueCacheName)

        // Build output backings (logits — written in-place)
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&logitsArray, for: logitsName)

        // Execute
        _ = try await function.run(
            inputs: [inputIdsName: inputIdsArray, positionIdsName: positionIds],
            states: consume states,
            outputViews: consume outputViews
        )

        // Read logits from NDArray
        let totalLogits = batchSize * config.vocabSize
        let logitBuffer = readNDArray(logitsArray, as: LogitsScalarType.self, count: totalLogits)

        processedTokenCount += batchSize

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIClean Batch",
            signpostID: batchSignpost
        )

        return logitBuffer
    }

    // MARK: - Chunked Prefill

    private func processChunkedPrompt(
        tokens: ArraySlice<Int32>,
        chunkSize: Int
    ) async throws -> [LogitsScalarType] {
        let totalChunks = (tokens.count + chunkSize - 1) / chunkSize

        let chunkSignpost = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIClean Chunked Prefill",
            details: "\(tokens.count) tokens in \(totalChunks) chunks of \(chunkSize)"
        )

        var lastLogits: [LogitsScalarType] = []
        var remainingTokens = tokens
        var chunkIndex = 0

        while !remainingTokens.isEmpty {
            let currentChunkSize = min(chunkSize, remainingTokens.count)
            let chunkEnd = remainingTokens.startIndex + currentChunkSize
            let chunk = remainingTokens[remainingTokens.startIndex..<chunkEnd]

            CLILogger.log(
                "Chunk \(chunkIndex + 1)/\(totalChunks): \(chunk.count) tokens at position \(processedTokenCount)"
            )

            lastLogits = try await processTokenBatch(chunk)
            remainingTokens = remainingTokens[chunkEnd...]
            chunkIndex += 1
        }

        InstrumentsProfiler.endCustomInterval(
            name: "CoreAIClean Chunked Prefill",
            signpostID: chunkSignpost
        )

        return lastTokenLogits(from: lastLogits, vocabSize: config.vocabSize)
    }

    // MARK: - Generate (primary API)

    public func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> GenerationSequence {
        // Implicit prefix caching: resolve before creating Iterator.
        // Implicit prefix caching: resolve input against history.
        if history.count > 0 {
            let (commonPrefix, _) = history.resolve(input: input)
            if commonPrefix < input.count && commonPrefix < history.count {
                // Divergence: input differs from history. Full reset needed.
                internalReset(to: 0)
            } else if processedTokenCount >= input.count {
                // Pure extension: all input tokens match history. Rewind for seeding.
                let resetTo = Swift.max(0, commonPrefix - 1)
                internalReset(to: resetTo)
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

    // MARK: - Lifecycle

    /// Wait for any in-flight generate() Task to finish.
    private func drain() {
        var attempts = 0
        while _activeToken.withLock({ $0 != nil }) {
            attempts += 1
            if attempts > 5000 {
                fatalError("Sequential engine drain() timeout — generation Task stuck?")
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

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
        internalReset(to: tokenIndex)
    }

    /// Internal reset without cancelling the active generation token.
    /// Used by the Iterator when it detects a prefix mismatch mid-generation.
    func internalReset(to tokenIndex: Int) {
        let resetSpan = InstrumentsProfiler.beginReset(engine: "CoreAIClean")
        if tokenIndex == 0 {
            processedTokenCount = 0
            history.clear()
            zeroFill(&keyCache)
            zeroFill(&valueCache)
        } else {
            processedTokenCount = tokenIndex
            history.truncate(to: tokenIndex)
        }
        resetSpan.end()
    }

    public func cleanup() {
        let cleanupSpan = InstrumentsProfiler.beginCleanup(engine: "CoreAIClean")
        CLILogger.log("CoreAI clean engine cleanup complete")
        cleanupSpan.end()
    }

    // MARK: - KV Cache (dynamic growth)

    private func ensureKVCapacity(forContextLength needed: Int) throws {
        guard needed > currentKVCapacity else { return }
        guard needed <= config.maxContextLength else {
            throw InferenceRuntimeError.invalidState(
                "Context length \(needed) exceeds maximum \(config.maxContextLength)")
        }

        var newCapacity = currentKVCapacity
        while newCapacity < needed { newCapacity *= 2 }
        newCapacity = min(newCapacity, config.maxContextLength)

        let resolvedKeyDesc = keyCacheDescriptor.resolvingDynamicDimensions(
            keyCacheDescriptor.shape.map { $0 < 0 ? newCapacity : $0 })
        let resolvedValueDesc = valueCacheDescriptor.resolvingDynamicDimensions(
            valueCacheDescriptor.shape.map { $0 < 0 ? newCapacity : $0 })

        var newKeyCache = NDArray(descriptor: resolvedKeyDesc)
        var newValueCache = NDArray(descriptor: resolvedValueDesc)
        _ = newKeyCache.mutableRawView()
        _ = newValueCache.mutableRawView()

        try Self.copyCache(from: keyCache, to: &newKeyCache)
        try Self.copyCache(from: valueCache, to: &newValueCache)

        CLILogger.log("KV cache grew: \(currentKVCapacity) → \(newCapacity)")
        keyCache = newKeyCache
        valueCache = newValueCache
        currentKVCapacity = newCapacity
    }

    private static func copyCache(from source: NDArray, to destination: inout NDArray) throws {
        let srcShape = source.shape
        let dstShape = destination.shape
        guard let headDim = srcShape.last else {
            throw InferenceRuntimeError.invalidState("KV cache has empty shape — cannot copy")
        }
        let seqDim = KVCacheFactory.detectSequenceDim(shape: srcShape)

        // Number of independent blocks before the sequence dimension (L * B * H or B * H)
        let numBlocks = srcShape[..<seqDim].reduce(1, *)
        let oldSeqLen = srcShape[seqDim]
        let copySize = oldSeqLen * headDim

        // Strides in elements for the sequence block
        let srcBlockStride = srcShape[seqDim...].reduce(1, *)  // S_old * D
        let dstBlockStride = dstShape[seqDim...].reduce(1, *)  // S_new * D

        source.view(as: LogitsScalarType.self).withUnsafePointer { srcPtr, _, _ in
            var dstView = destination.mutableView(as: LogitsScalarType.self)
            dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                for block in 0..<numBlocks {
                    let srcOff = block * srcBlockStride
                    let dstOff = block * dstBlockStride
                    dstPtr.advanced(by: dstOff).update(
                        from: srcPtr.advanced(by: srcOff), count: copySize)
                }
            }
        }
    }

    // MARK: - Helpers

    private func zeroFill(_ array: inout NDArray) {
        let count = array.shape.reduce(1, *)
        var view = array.mutableView(as: LogitsScalarType.self)
        // Inlined constant write — under -Onone, fillNDArray's
        // `(Int) -> LogitsScalarType` closure is invoked per element (no inlining),
        // which made zeroing the KV cache (~14.7M elements for a 32K-context
        // Qwen3) take ~6 seconds per `reset()`. Direct loop keeps this in
        // the few-ms range even unoptimized; under -O it lowers to memset.
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count {
                ptr[i] = 0
            }
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

extension CoreAISequentialEngine.GenerationSequence {
    public final class Iterator: AsyncIteratorProtocol {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        private let engine: CoreAISequentialEngine
        private let samplingConfiguration: SamplingConfiguration
        private let returnsLogits: Bool
        private let forcedContinuation: [CoreAISequentialEngine.TokenId]?
        private let maxTokens: Int
        private let stopReasonStore: StopReasonStore
        private let generationToken: GenerationToken

        private var inputTokens: [CoreAISequentialEngine.TokenId]
        private var step: Int = 0
        private var finished: Bool = false

        init(
            engine: CoreAISequentialEngine,
            input: [CoreAISequentialEngine.TokenId],
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
            if let forced = inferenceOptions.forcedContinuation {
                self.maxTokens = forced.count
            } else {
                self.maxTokens = Swift.min(
                    inferenceOptions.maxTokens ?? Int.max,
                    Swift.max(0, engine.config.maxContextLength - input.count)
                )
            }
        }

        deinit {
            engine.clearTokenIfActive(generationToken)
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

            do {
                try Task.checkCancellation()

                guard engine.processedTokenCount < inputTokens.count else {
                    throw InferenceRuntimeError.invalidState("No new tokens to process")
                }

                let oldProcessedCount = engine.processedTokenCount
                let newTokens = inputTokens[engine.processedTokenCount...]
                let strategy = engine.selectPrefillStrategy(newTokenCount: newTokens.count)

                let logitBuffer: [LogitsScalarType]
                switch strategy {
                case .chunked(let chunkSize):
                    logitBuffer = try await engine.processChunkedPrompt(tokens: newTokens, chunkSize: chunkSize)
                case .wholeBatch:
                    let allLogits = try await engine.processTokenBatch(newTokens)
                    logitBuffer = lastTokenLogits(from: allLogits, vocabSize: engine.config.vocabSize)
                case .oneAtATime:
                    var lastLogits: [LogitsScalarType] = []
                    for j in newTokens.indices {
                        lastLogits = try await engine.processTokenBatch(newTokens[j...j])
                    }
                    logitBuffer = lastLogits
                }

                // Update history with newly processed tokens
                let processedSlice = inputTokens[oldProcessedCount..<engine.processedTokenCount]
                engine.history.append(contentsOf: processedSlice)

                // Check cancellation after inference step
                if generationToken.isCancelled {
                    stopReasonStore.set(.cancelled)
                    finishAndRelease()
                    return nil
                }

                let nextToken: Int32
                if let forced = forcedContinuation {
                    nextToken = forced[step]
                } else {
                    var mutableLogits = logitBuffer
                    nextToken = samplingConfiguration.fallbackSampler(from: &mutableLogits)
                }

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
            engine.clearTokenIfActive(generationToken)
        }
    }
}
