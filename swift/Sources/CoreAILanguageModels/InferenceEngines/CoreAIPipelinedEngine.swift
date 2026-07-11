// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Metal
import MetalPerformanceShaders
import Synchronization
import os

// MARK: - Timing

private func milliseconds(since start: ContinuousClock.Instant) -> Double {
    let duration = ContinuousClock.now - start
    let (secs, attoseconds) = duration.components
    return (Double(secs) + Double(attoseconds) / 1e18) * 1000.0
}

// MARK: - Constants

/// Maximum number of in-flight pipeline stages. Shared by the backpressure gate
/// and all buffer rotation logic to guarantee no two concurrent stages alias
/// the same memory.
private let pipelineDepth = 3
private let averageExpectedPromptSize = 256
private let temperatureTolerance: Double = 0.001

/// MPSNDArray enforces 64-byte row-stride alignment on backing buffers.
private let minimumMPSNDArrayBufferSize = 64

// MARK: - Core AI Pipelined Engine (Public Wrapper)

/// GPU-pipelined inference engine using Core AI's encode API.
///
/// Key features:
/// - Non-blocking GPU encoding via `InferenceFunction.encode`
/// - GPU-direct token sampling (argmax/topK) via MPSGraph compute shaders
/// - Pipeline-depth-matched buffer rotation for CPU/GPU overlap
/// - Growing KV cache with pipelined expansion
/// - All tensors are owned MTLBuffers — Core AI never allocates/frees them
final class CoreAIPipelinedEngine: InferenceEngine, Sendable {
    typealias ConfigType = ModelConfig

    nonisolated(unsafe) private var engine: EngineImpl
    nonisolated(unsafe) private var history = TokenHistory()
    nonisolated(unsafe) private(set) var lastPrefixHitCount: Int = 0
    private let engineInUse = Atomic<Bool>(false)
    let config: ModelConfig

    // Generation lifecycle
    private let _activeToken = Mutex<GenerationToken?>(nil)
    private let _generationTask = Mutex<Task<Void, Never>?>(nil)

    var isBusy: Bool { _activeToken.withLock { $0 != nil } }

    var processedTokenCount: Int { engine.processedTokenCount }

    init(
        config: ModelConfig,
        preparedModel: PreparedModel,
        options: EngineOptions = EngineOptions()
    ) async throws {
        let engine = try await EngineImpl(
            config: config, preparedModel: preparedModel, options: options)
        self.engine = engine
        self.config = config
    }

    /// Atomically claim exclusive use of `engine`.
    ///
    /// Traps on contention. Callers must guarantee single-ownership.
    private func acquireEngine() {
        let (exchanged, _) = engineInUse.compareExchange(
            expected: false,
            desired: true,
            ordering: .acquiring
        )
        guard exchanged else {
            fatalError("Trying to acquire engine when it's already in use")
        }
    }

    /// Try to claim exclusive use of `engine` without trapping.
    ///
    /// Returns `true` if the caller now holds it (and must call `releaseEngine`), `false` if
    /// another caller holds it.
    private func tryAcquireEngine() -> Bool {
        let (exchanged, _) = engineInUse.compareExchange(
            expected: false,
            desired: true,
            ordering: .acquiring
        )
        return exchanged
    }

    private func releaseEngine() {
        engineInUse.store(false, ordering: .releasing)
    }

    func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        if inferenceOptions.includeLogits {
            throw InferenceRuntimeError.invalidArgument(
                "CoreAI pipelined engine does not support logits (GPU-side sampling). "
                    + "Use a sequential engine for constrained generation or evaluation."
            )
        }
        if inferenceOptions.forcedContinuation != nil {
            throw InferenceRuntimeError.invalidArgument(
                "CoreAI pipelined engine does not support forcedContinuation (GPU-side sampling). "
                    + "Use a sequential engine for evaluation."
            )
        }

        // Serialize: if a prior generation is still winding down (GPU drain),
        // cancel it and wait for the engine slot to be released.
        if let priorTask = _generationTask.withLock({ $0 }) {
            _activeToken.withLock { $0?.cancel() }
            await priorTask.value
        }

        let maxTokens = inferenceOptions.maxTokens
        let stopReasonStore = StopReasonStore()
        let (base, outputContinuation) =
            AsyncThrowingStream<InferenceOutput, any Error>.makeStream()

        let token = GenerationToken()
        _activeToken.withLock { $0 = token }

        let task = Task {
            self.acquireEngine()
            defer {
                self.releaseEngine()
                // Only clear if this generation still owns both slots
                if self._activeToken.withLock({ $0 === token }) {
                    self._activeToken.withLock { $0 = nil }
                    self._generationTask.withLock { $0 = nil }
                }
            }
            do {
                let (tokenStream, tokenContinuation) =
                    AsyncThrowingStream<InferenceEngine.TokenId, any Error>.makeStream()

                // Implicit prefix caching: resolve input against history
                var (commonPrefix, resolvedNewTokens) = self.history.resolve(input: input)
                self.lastPrefixHitCount = commonPrefix

                // Detect TRUE divergence before backup (tokens actually differ)
                let isDivergence = commonPrefix < input.count && commonPrefix < self.history.count

                // Ensure at least 1 token for prefill (seeds the decode loop).
                // Back up by 1 if the entire input is cached.
                if resolvedNewTokens.isEmpty && commonPrefix > 0 {
                    commonPrefix -= 1
                    resolvedNewTokens = input[commonPrefix...]
                }

                if isDivergence {
                    // Tokens differ — full reset (partial rewind corrupts buffer rotation)
                    await self.engine.computeStream.currentWorkCompleted()
                    self.engine.reset()
                    self.history.clear()
                    resolvedNewTokens = input[...]
                    commonPrefix = 0
                } else if commonPrefix < self.engine.processedTokenCount {
                    // Pure extension — partial rewind (buffer phase preserved)
                    await self.engine.computeStream.currentWorkCompleted()
                    self.engine.processedTokenCount = commonPrefix
                    self.engine.step = commonPrefix
                    self.history.truncate(to: commonPrefix)
                }

                let newTokens = Array(resolvedNewTokens)

                async let forwarding: Void = {
                    do {
                        for try await token in tokenStream {
                            // Track generated tokens in history
                            self.history.append(token)
                            outputContinuation.yield(InferenceOutput(tokenId: token))
                        }
                    } catch {
                        outputContinuation.finish(throwing: error)
                    }
                }()

                // Track prefill tokens BEFORE runCompletion — the forwarding loop
                // concurrently appends generated tokens, so prefill must come first.
                if !newTokens.isEmpty {
                    self.history.append(contentsOf: newTokens[...])
                }

                try await self.engine.runCompletion(
                    prompt: newTokens,
                    sampler: samplingConfiguration,
                    maxTokens: maxTokens,
                    yieldingTo: tokenContinuation
                )

                tokenContinuation.finish()
                await forwarding
                stopReasonStore.setIfUnset(.maxTokens)
                outputContinuation.finish()
            } catch is CancellationError {
                stopReasonStore.set(.cancelled)
                outputContinuation.finish()
            } catch {
                stopReasonStore.set(.error)
                outputContinuation.finish(throwing: error)
            }
        }
        _generationTask.withLock { $0 = task }
        return GenerationSequence(base: base, stopReasonStore: stopReasonStore)
    }

    /// Wait for any in-flight generate() Task to return the engine.
    private func drain() {
        var attempts = 0
        while engineInUse.load(ordering: .acquiring) {
            attempts += 1
            if attempts > 5000 {
                fatalError("Engine not returned after drain() — tokenSequence Task stuck?")
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    func cancel() async throws {
        let task: Task<Void, Never>? = _generationTask.withLock { task in
            task?.cancel()
            defer { task = nil }
            return task
        }
        _activeToken.withLock {
            $0?.cancel()
            $0 = nil
        }
        await task?.value
    }

    func reset(to tokenIndex: Int) async throws {
        precondition(
            tokenIndex >= 0 && tokenIndex <= processedTokenCount,
            "reset(to: \(tokenIndex)) out of range [0, \(processedTokenCount)]")
        if tokenIndex == 0 {
            // Full reset: cancel + drain + clear everything
            _activeToken.withLock {
                $0?.cancel()
                $0 = nil
            }
            _generationTask.withLock {
                $0?.cancel()
                $0 = nil
            }
            drain()
            await engine.computeStream.currentWorkCompleted()
            guard tryAcquireEngine() else { return }
            defer { releaseEngine() }
            engine.reset()
            history.clear()
        } else {
            // Partial reset: wait for generation to finish naturally, then rewind counter.
            // Do NOT cancel — cancelling corrupts the pipeline's double-buffer state.
            // The KV cache is valid up to processedTokenCount after natural completion.
            drain()
            await engine.computeStream.currentWorkCompleted()
            guard tryAcquireEngine() else { return }
            defer { releaseEngine() }
            engine.processedTokenCount = tokenIndex
            engine.step = tokenIndex
            history.truncate(to: tokenIndex)
        }
    }

    func cleanup() async throws {
        let cleanupSpan = InstrumentsProfiler.beginCleanup(engine: "CoreAI-Pipelined")
        if tryAcquireEngine() {
            let stream = engine.computeStream
            releaseEngine()
            await stream.currentWorkCompleted()
        }
        cleanupSpan.end()
    }

    func validateSamplingStrategy(_ config: SamplingConfiguration) throws {
        // All sampling configurations are now supported by the GPU sampler:
        // greedy, temperature, topK, topP, and minP.
    }

    func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {
        acquireEngine()
        defer { releaseEngine() }
        try await engine.performWarmup(queryLength: queryLength, samplingConfig: sampling)
    }
}

// MARK: - Pipeline Depth Gate

/// Bounds in-flight encode calls so MPSGraph's per-encode scratch
/// (sized by the graph's max shape — multiple GB on large models) can't accumulate.
///
/// Without this, the decode loop submits encodes (~220/s) faster than the
/// sampler callback drains them (~70/s); depth grows until
/// `MPSCommandBufferImageCache` fails to allocate another private MTLBuffer.
///
/// Capacity matches `pipelineDepth` — covers {logits encode + sampler commit + optional KV-cache grow};
/// deeper queues only cost memory.
///
/// Class, not actor: `release()` runs synchronously from the Metal callback —
/// an actor would force `Task { await release() }` with ordering ambiguity.
/// `internal` (not `private`) so `PipelineGateTests` can reach it.
final class PipelineGate: Sendable {
    private struct State: Sendable {
        var inFlight: Int = 0
        var waiters: [CheckedContinuation<Void, Never>] = []
    }

    private let capacity: Int
    private let state = OSAllocatedUnfairLock<State>(initialState: State())

    init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    /// Take a slot; suspend if all slots are busy.
    func acquire() async {
        // Fast path: take a slot without suspending.
        let takenImmediately = state.withLock { state -> Bool in
            guard state.inFlight < capacity else { return false }
            state.inFlight += 1
            return true
        }
        if takenImmediately { return }

        // Slow path: enqueue a waiter. Re-check under the lock in case a slot
        // opened between the fast path and now.
        await withCheckedContinuation { cont in
            let runImmediately = state.withLock { state -> Bool in
                if state.inFlight < capacity {
                    state.inFlight += 1
                    return true
                }
                state.waiters.append(cont)
                return false
            }
            if runImmediately { cont.resume() }
        }
    }

    /// Give back a slot. Called from the sampler's GPU-completion callback on a
    /// Metal callback thread; resumes any pending waiter (slot transferred
    /// directly without decrementing `inFlight`) or decrements the count.
    ///
    /// The waiter is resumed *outside* the lock so a rescheduled task can't
    /// re-enter `acquire` while we still hold it.
    func release() {
        let waiter = state.withLock { state -> CheckedContinuation<Void, Never>? in
            if !state.waiters.isEmpty {
                // Slot transferred to the woken waiter — inFlight count unchanged.
                return state.waiters.removeFirst()
            }
            state.inFlight -= 1
            return nil
        }
        waiter?.resume()
    }

    // Test-only introspection. Kept as underscored names to discourage
    // production use; exercised by PipelineGateTests.

    var _inFlightForTesting: Int {
        state.withLock { $0.inFlight }
    }

    var _waitersForTesting: Int {
        state.withLock { $0.waiters.count }
    }
}

// MARK: - Engine Implementation

private struct EngineImpl: ~Copyable {
    var vocabSize: Int { config.vocabSize }

    let config: ModelConfig
    let options: EngineOptions
    let function: InferenceFunction
    let pipelineQueue: MTLCommandQueue
    let computeStream: ComputeStream
    let device: MTLDevice

    // Descriptor metadata
    let inputIdsName: String
    let positionIdsName: String
    let keyCacheName: String
    let valueCacheName: String
    let logitsOutputName: String
    let keyCacheScalarType: NDArray.ScalarType
    let valueCacheScalarType: NDArray.ScalarType

    // Base descriptors for shape resolution (preferredStrides, not contiguous)
    let inputIdsBaseDesc: NDArrayDescriptor
    let positionIdsBaseDesc: NDArrayDescriptor
    let logitsBaseDesc: NDArrayDescriptor

    // Owned MTLBuffers
    var inputTokensBuffer: MTLBuffer
    var cachePositionBuffers: [MTLBuffer]
    var decodeOutputBuffers: [MTLBuffer]
    var decodeLogitsBuffers: [MTLBuffer]

    // KV cache — reuses CoreAIKVCache protocol from KVCache+CoreAI.swift
    var kvCache: any CoreAIKVCache

    // Logits — reuses GrowingLogitsBuffer from TensorStorage+CoreAI.swift
    var logits: GrowingLogitsBuffer

    // GPU sampler — reuses MPSGraphSampler from MPSGraphSamplers.swift
    var cachedSampler: (any MPSGraphSampler)?
    var cachedSamplerTemperature: Double?

    // State
    var processedTokenCount: Int = 0
    var step: Int = 0

    // Backpressure gate — see PipelineGate doc-comment for the failure mode it prevents.
    // Capacity matches pipeline depth: {encode logits + sampler commit + optional KV-cache grow} in flight.
    let inFlightGate = PipelineGate(capacity: pipelineDepth)

    // MARK: - Init

    init(
        config: ModelConfig,
        preparedModel: PreparedModel,
        options: EngineOptions = EngineOptions()
    ) async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw InferenceRuntimeError.genericError("Failed to create Metal device")
        }

        let model = preparedModel.model

        // Get function descriptor
        guard let descriptor = model.functionDescriptor(for: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot find function '\(config.function)' in model")
        }

        // Validate: 2 inputs, 1+ output, 2 states
        guard descriptor.inputNames.count == 2 else {
            throw InferenceRuntimeError.invalidInputType(
                "Expected 2 inputs, got \(descriptor.inputNames.count): \(descriptor.inputNames)")
        }
        guard descriptor.outputNames.count >= 1 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 1 output, got \(descriptor.outputNames.count)")
        }
        guard descriptor.stateNames.count == 2 else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected 2 states (KV cache), got \(descriptor.stateNames.count): \(descriptor.stateNames)")
        }

        // Extract names
        let inputIdsName = descriptor.inputNames[0]
        let positionIdsName = descriptor.inputNames[1]
        let keyCacheName = descriptor.stateNames[0]
        let valueCacheName = descriptor.stateNames[1]
        let logitsOutputName = descriptor.outputNames[0]

        // Extract state descriptors for KV cache shape/type
        guard case .ndArray(let keyCacheDesc) = descriptor.stateDescriptor(of: keyCacheName),
            case .ndArray(let valueCacheDesc) = descriptor.stateDescriptor(of: valueCacheName)
        else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get KV cache state descriptors")
        }

        // Extract input descriptors
        guard case .ndArray(let inputIdsDesc) = descriptor.inputDescriptor(of: inputIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(inputIdsName)'")
        }
        guard case .ndArray(let posIdsDesc) = descriptor.inputDescriptor(of: positionIdsName) else {
            throw InferenceRuntimeError.invalidInputType("Cannot get descriptor for '\(positionIdsName)'")
        }

        // Extract logits descriptor
        guard case .ndArray(let logitsDesc) = descriptor.outputDescriptor(of: logitsOutputName) else {
            throw InferenceRuntimeError.invalidOutputType("Cannot get descriptor for '\(logitsOutputName)'")
        }
        guard logitsDesc.scalarType == .float16 else {
            throw InferenceRuntimeError.unsupportedLogitsType(
                "Only float16 logits supported, got \(logitsDesc.scalarType)")
        }

        // Allocate inputTokens MTLBuffer
        let inputTokensByteCount = config.maxContextLength * inputIdsDesc.scalarType.byteSize
        guard let inputTokensBuf = device.makeBuffer(length: inputTokensByteCount, options: .storageModeShared) else {
            throw InferenceRuntimeError.bufferAllocationFailed("inputTokens (\(inputTokensByteCount) bytes)")
        }

        // Allocate pipeline-depth-matched cache position buffers
        let cachePosSize = config.maxContextLength * posIdsDesc.scalarType.byteSize
        var cachePosBuffers: [MTLBuffer] = []
        for _ in 0..<pipelineDepth {
            guard let buf = device.makeBuffer(length: cachePosSize, options: .storageModeShared) else {
                throw InferenceRuntimeError.bufferAllocationFailed("cachePositions (\(cachePosSize) bytes)")
            }
            cachePosBuffers.append(buf)
        }

        // Pre-populate cache positions with [0, 1, ..., maxCtx-1]
        for buf in cachePosBuffers {
            let ptr = buf.contents().bindMemory(to: Int32.self, capacity: config.maxContextLength)
            for i in 0..<config.maxContextLength {
                ptr[i] = Int32(i)
            }
        }

        // Allocate pipeline-depth-matched decode output buffers (sampler writes next token)
        var decodeOutBuffers: [MTLBuffer] = []
        for _ in 0..<pipelineDepth {
            let decodeOutSize = max(minimumMPSNDArrayBufferSize, MemoryLayout<Int32>.size)
            guard let buf = device.makeBuffer(length: decodeOutSize, options: .storageModeShared) else {
                throw InferenceRuntimeError.bufferAllocationFailed(
                    "decodeOutputBuffer (\(decodeOutSize) bytes)")
            }
            decodeOutBuffers.append(buf)
        }

        // Allocate pipeline-depth-matched decode logits buffers (inference writes logits for decode)
        let decodeLogitsSize = config.vocabSize * MemoryLayout<UInt16>.size
        var decodeLogBufs: [MTLBuffer] = []
        for _ in 0..<pipelineDepth {
            guard let buf = device.makeBuffer(length: decodeLogitsSize, options: .storageModeShared) else {
                throw InferenceRuntimeError.bufferAllocationFailed("decodeLogitsBuffer (\(decodeLogitsSize) bytes)")
            }
            decodeLogBufs.append(buf)
        }

        // Create KV cache using factory — pass original descriptors (with -1 dynamic dims intact)
        // so the factory can correctly detect growing vs static support via isDynamicKVCache().
        let kvCacheLocal = try KVCacheFactory.make(
            options: options,
            device: device,
            keyReqs: keyCacheDesc,
            valueReqs: valueCacheDesc,
            maxContextLength: config.maxContextLength
        )

        let resolvedSize = options.resolvedKVCacheSize(maxContextLength: config.maxContextLength)
        CLILogger.log("Created \(options.kvCacheStrategy) KV cache with size \(resolvedSize, default: "nil")")

        // Create growing logits buffer (reuses TensorStorage+CoreAI.swift)
        let logitsRef = try GrowingLogitsBuffer(
            device: device,
            descriptor: descriptor,
            name: logitsOutputName,
            vocabSize: config.vocabSize,
            maxCapacity: config.maxContextLength,
            initialCapacity: averageExpectedPromptSize
        )

        // Load inference function
        guard let fn = try model.loadFunction(named: config.function) else {
            throw InferenceRuntimeError.genericError(
                "Cannot load function '\(config.function)'")
        }

        guard let pipelineQueue = device.makeCommandQueue() else {
            throw InferenceRuntimeError.invalidState(
                "Failed to allocate MTLCommandQueue for CoreAIPipelinedEngine")
        }
        pipelineQueue.label = "CoreAIPipelinedEngine.queue"
        let computeStream = ComputeStream(commandQueue: pipelineQueue)

        // Assign
        self.config = config
        self.options = options
        self.function = fn
        self.pipelineQueue = pipelineQueue
        self.computeStream = computeStream
        self.device = device
        self.inputIdsName = inputIdsName
        self.positionIdsName = positionIdsName
        self.keyCacheName = keyCacheName
        self.valueCacheName = valueCacheName
        self.logitsOutputName = logitsOutputName
        self.keyCacheScalarType = keyCacheDesc.scalarType
        self.valueCacheScalarType = valueCacheDesc.scalarType
        self.inputIdsBaseDesc = inputIdsDesc
        self.positionIdsBaseDesc = posIdsDesc
        self.logitsBaseDesc = logitsDesc
        self.inputTokensBuffer = inputTokensBuf
        self.cachePositionBuffers = cachePosBuffers
        self.decodeOutputBuffers = decodeOutBuffers
        self.decodeLogitsBuffers = decodeLogBufs
        self.kvCache = kvCacheLocal
        self.logits = logitsRef
        self.cachedSampler = nil
        self.cachedSamplerTemperature = nil

        CLILogger.log("CoreAI pipelined engine initialized — Vocab: \(config.vocabSize)")
    }

    // MARK: - Sampler

    private mutating func getOrCreateSampler(for config: SamplingConfiguration) throws -> any MPSGraphSampler {
        let temperature = config.temperature

        if let existingSampler = cachedSampler, let existingTemp = cachedSamplerTemperature {
            let existingIsGreedy = existingTemp == 0
            let requestedIsGreedy = temperature == 0

            if existingIsGreedy != requestedIsGreedy {
                throw InferenceRuntimeError.genericError(
                    "Sampling configuration changed mid-generation. Call reset() first.")
            }
            if !existingIsGreedy && !requestedIsGreedy
                && abs(existingTemp - temperature) > temperatureTolerance
            {
                throw InferenceRuntimeError.genericError(
                    "Temperature changed mid-generation (\(existingTemp) -> \(temperature)). Call reset() first.")
            }
            return existingSampler
        }

        let newSampler = try MPSGraphSamplerFactory.makeSampler(
            device: device,
            vocabSize: self.config.vocabSize,
            config: config
        )
        cachedSampler = newSampler
        cachedSamplerTemperature = temperature
        return newSampler
    }

    // MARK: - Core Encode Step

    /// Encodes inference + GPU sampling for one step.
    ///
    /// 1. Construct RawView/MutableRawView from MTLBuffers with current shapes
    /// 2. Encode to ComputeStream (non-blocking)
    /// 3. withMetal3Queue: encode GPU argmax/topK (writes to rotating decodeOutputBuffers)
    /// 4. Callback yields token
    private mutating func _encodeNextStepGPU(
        tokens: some Collection<Int32>,
        gpuSampler: any MPSGraphSampler,
        yieldingTo continuation: AsyncThrowingStream<InferenceEngine.TokenId, Error>.Continuation
    ) async throws {
        let currentStep = processedTokenCount

        let actualTokenCount = tokens.isEmpty ? 1 : tokens.count
        let queryLength = actualTokenCount

        defer {
            processedTokenCount += actualTokenCount
            step += 1
        }

        let encodeStepID = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIPipelinedEncodeNextStep",
            details: "step=\(currentStep) qLen=\(queryLength)"
        )

        // PrepareStep: write tokens + build views
        let prepareSpan = InstrumentsProfiler.beginPrepareStep(
            step: currentStep, operation: "write+build", engine: "CoreAI-Pipelined")

        // Prefill: write tokens at their natural position so this step's region is disjoint
        // from any prior chunk's region still in-flight on the GPU (encode holds a live
        // MTLBuffer reference; no encodeWriteOperands serialization available in Core AI).
        // Decode: token is in the previous step's decodeOutputBuffer — no CPU write needed.
        let tokenByteOffset = processedTokenCount * MemoryLayout<Int32>.size
        if !tokens.isEmpty {
            let ptr = inputTokensBuffer.contents().bindMemory(
                to: Int32.self, capacity: processedTokenCount + queryLength)
            for (i, token) in tokens.enumerated() {
                ptr[processedTokenCount + i] = token
            }
        }

        // Select cache position buffer for this step (pipeline-depth-matched rotation)
        let cachePosBuffer = cachePositionBuffers[step % pipelineDepth]
        let posLength = processedTokenCount + queryLength

        // Build Inputs as AsyncValue (from MTLBuffers)
        let tokenShape = [1, queryLength]
        let tokenStrides = try resolvedStrides(descriptor: inputIdsBaseDesc, shape: tokenShape)
        let tokenValue: InferenceFunction.AsyncValue
        if tokens.isEmpty {
            // Decode: read input token from previous step's decode output buffer
            tokenValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: decodeOutputBuffers[(step + pipelineDepth - 1) % pipelineDepth],
                byteOffset: 0,
                scalarType: .int32,
                shape: tokenShape,
                strides: tokenStrides
            )
        } else {
            // Prefill: read from inputTokensBuffer at natural position
            tokenValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: inputTokensBuffer,
                byteOffset: tokenByteOffset,
                scalarType: .int32,
                shape: tokenShape,
                strides: tokenStrides
            )
        }
        let posShape = [1, posLength]
        let posStrides = try resolvedStrides(descriptor: positionIdsBaseDesc, shape: posShape)
        let posValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: cachePosBuffer,
            byteOffset: 0,
            scalarType: .int32,
            shape: posShape,
            strides: posStrides
        )

        let asyncInputs: [String: InferenceFunction.AsyncValue] = [
            inputIdsName: tokenValue,
            positionIdsName: posValue,
        ]

        // Build States as AsyncMutableValue (KV cache, in-place update)
        let keyBuffer = kvCache.keyBinding.metalBuffer
        let keyShape = kvCache.keyBinding.layout.shape
        let keyStrides = kvCache.keyBinding.layout.strides
        var keyState = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: keyBuffer,
            byteOffset: 0,
            scalarType: keyCacheScalarType,
            shape: keyShape,
            strides: keyStrides
        )
        let valBuffer = kvCache.valueBinding.metalBuffer
        let valShape = kvCache.valueBinding.layout.shape
        let valStrides = kvCache.valueBinding.layout.strides
        var valState = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: valBuffer,
            byteOffset: 0,
            scalarType: valueCacheScalarType,
            shape: valShape,
            strides: valStrides
        )

        var asyncStates = InferenceFunction.AsyncMutableViews()
        asyncStates.insert(&keyState, for: keyCacheName)
        asyncStates.insert(&valState, for: valueCacheName)

        // Build Output as AsyncMutableValue (logits)
        // Decode uses per-step rotating buffer; prefill uses the shared growing buffer.
        let logitsOutputBuffer = tokens.isEmpty ? decodeLogitsBuffers[step % pipelineDepth] : logits.metalBuffer
        let logitsShape = [1, queryLength, vocabSize]
        let logitsStrides = try resolvedStrides(descriptor: logitsBaseDesc, shape: logitsShape)
        var logitsOutput = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: logitsOutputBuffer,
            byteOffset: 0,
            scalarType: .float16,
            shape: logitsShape,
            strides: logitsStrides
        )

        var asyncOutputs = InferenceFunction.AsyncMutableViews()
        asyncOutputs.insert(&logitsOutput, for: logitsOutputName)

        prepareSpan.end()

        // Backpressure: cap outstanding encode calls
        await inFlightGate.acquire()

        // Encode inference using the public encode() API.
        // This commits + uses runAfterSyncPoint (no stream wait) — enables true pipelining.
        let logitsSpan = InstrumentsProfiler.beginLogitsInference(
            step: currentStep, tokens: queryLength, engine: "CoreAI-Pipelined")
        let _ = try function.encode(
            inputs: asyncInputs,
            states: consume asyncStates,
            outputViews: consume asyncOutputs,
            to: computeStream
        )
        logitsSpan.end()

        // GPU sampling via Metal queue
        let localGPUSampler = gpuSampler
        let outputBuffer = decodeOutputBuffers[step % pipelineDepth]
        let samplerLogitsBuffer = tokens.isEmpty ? decodeLogitsBuffers[step % pipelineDepth] : logits.metalBuffer
        let logitsOffset = (actualTokenCount - 1) * vocabSize * MemoryLayout<UInt16>.size
        let samplerStrategy = gpuSampler is MPSGraphArgmaxSampler ? "GPU-argmax" : "GPU-composite"
        let samplerTemperature = cachedSamplerTemperature ?? 0.0

        let sampleSpan = InstrumentsProfiler.beginSampleEncoding(
            step: currentStep, strategy: samplerStrategy, temperature: samplerTemperature)

        do {
            let queue = pipelineQueue
            let localInFlightGate = inFlightGate
            let completionCallback: (Int32) -> Void = { nextToken in
                // Release the pipeline slot acquired before encode. Happens on
                // Metal's callback thread — PipelineGate.release() is thread-safe.
                localInFlightGate.release()
                InstrumentsProfiler.endCustomInterval(
                    name: "CoreAIPipelinedEncodeNextStep",
                    signpostID: encodeStepID,
                    details: "token=\(nextToken)"
                )
                continuation.yield(nextToken)
            }

            if queryLength == 1 {
                localGPUSampler.encode(
                    to: queue,
                    logitsBuffer: samplerLogitsBuffer,
                    logitsOffset: logitsOffset,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: completionCallback
                )
            } else {
                localGPUSampler.encodeWithSlice(
                    to: queue,
                    logitsBuffer: samplerLogitsBuffer,
                    queryLength: actualTokenCount,
                    outputBuffer: outputBuffer,
                    outputOffset: 0,
                    completion: completionCallback
                )
            }
        }

        sampleSpan.end()
    }

    // MARK: - Token Generation

    private mutating func generateTokenBatch(
        count: Int,
        gpuSampler: any MPSGraphSampler,
        yieldingTo continuation: AsyncThrowingStream<InferenceEngine.TokenId, Error>.Continuation,
        isCancelled: borrowing Atomic<Bool>
    ) async throws {
        for _ in 0..<count {
            guard !isCancelled.load(ordering: .relaxed) else { return }
            try await _encodeNextStepGPU(
                tokens: [],
                gpuSampler: gpuSampler,
                yieldingTo: continuation
            )
        }
    }

    // MARK: - KV Cache Growth

    private mutating func growKVCacheAndRebind(neededCapacity: Int) async throws {
        let cacheSpan = InstrumentsProfiler.beginCacheManagement(
            step: processedTokenCount, operation: "grow", engine: "CoreAI-Pipelined")

        do {
            do {
                let queue = pipelineQueue
                guard let cmdBuf = queue.makeCommandBuffer() else {
                    throw KVCacheError.allocationFailed(0)
                }

                if (try kvCache.encodePipelinedExpansion(
                    forContextLength: neededCapacity,
                    commandBuffer: cmdBuf)) != nil
                {
                    CLILogger.log("KV cache grew (pipelined) to \(kvCache.currentCapacity)")
                } else {
                    throw KVCacheError.capacityExceeded(
                        needed: neededCapacity, available: kvCache.currentCapacity)
                }
            }
        } catch {
            cacheSpan.end()
            throw error
        }
        cacheSpan.end()
    }

    // MARK: - Run Completion

    mutating func runCompletion(
        prompt: [InferenceEngine.TokenId],
        sampler: SamplingConfiguration,
        maxTokens: Int?,
        yieldingTo continuation: AsyncThrowingStream<InferenceEngine.TokenId, Error>.Continuation
    ) async throws {
        let gpuSampler = try getOrCreateSampler(for: sampler)

        let isCancelled = Atomic<Bool>(false)
        continuation.onTermination = { _ in
            isCancelled.store(true, ordering: .relaxed)
        }

        let contextLeftAfterPrompt = config.maxContextLength - processedTokenCount - prompt.count
        guard contextLeftAfterPrompt >= 1 else {
            throw InferenceRuntimeError.contextLengthExceeded(
                processedTokenCount, config.maxContextLength)
        }
        let totalMaxTokens = min(maxTokens ?? Int.max, contextLeftAfterPrompt)

        // Pre-grow KV cache for prompt
        let promptCapacityNeeded = min(
            processedTokenCount + prompt.count + totalMaxTokens, config.maxContextLength)
        if promptCapacityNeeded > kvCache.currentCapacity {
            do {
                let queue = pipelineQueue
                let grew = try kvCache.ensureCapacity(
                    forContextLength: promptCapacityNeeded, queue: queue)
                if grew {
                    CLILogger.log(
                        "KV cache grew to \(kvCache.currentCapacity) for prompt (\(prompt.count) tokens)"
                    )
                }
            }
        }

        // Split prompt into chunks when it exceeds the chunk threshold.
        // Skip prefill entirely if prompt is empty (prefix-cached continuation).
        if !prompt.isEmpty {
            let prefillTokens: ArraySlice<Int32>
            if prompt.count > config.chunkThreshold {
                prefillTokens = try await processChunkedInput(tokens: prompt)
            } else {
                let prefillCapacity = max(1, prompt.count)
                if try logits.ensureCapacity(forContextLength: prefillCapacity) {
                    let fmt = ByteCountFormatter()
                    fmt.countStyle = .memory
                    CLILogger.log(
                        "Logits buffer grew to capacity \(logits.currentCapacity) (\(fmt.string(fromByteCount: Int64(logits.currentByteCount))))"
                    )
                }
                prefillTokens = prompt[...]
            }

            // Process prompt with sampling
            try await _encodeNextStepGPU(
                tokens: prefillTokens,
                gpuSampler: gpuSampler,
                yieldingTo: continuation
            )
        }

        // Generate-Grow-Continue loop
        var remainingTokens = totalMaxTokens - 1

        while remainingTokens > 0 {
            guard !isCancelled.load(ordering: .relaxed) else { break }

            let availableSlots = kvCache.currentCapacity - processedTokenCount
            let tokensThisRound = min(remainingTokens, availableSlots)

            if tokensThisRound > 0 {
                try await generateTokenBatch(
                    count: tokensThisRound,
                    gpuSampler: gpuSampler,
                    yieldingTo: continuation,
                    isCancelled: isCancelled
                )
                remainingTokens -= tokensThisRound
            }

            if remainingTokens > 0 {
                let neededCapacity = processedTokenCount + remainingTokens
                try await growKVCacheAndRebind(neededCapacity: neededCapacity)
            }
        }

        // Sentinel: submit an empty command buffer on the same serial queue.
        // Its addCompletedHandler fires after all real sampler callbacks (serial
        // queue FIFO ordering via MTLDispatchListApply), guaranteeing every
        // continuation.yield has returned before the caller calls finish().
        // We use a bare command buffer instead of the sampler to avoid the shared
        // MPSGraphExecutableExecutionDescriptor issue in MPSGraphCompositeSampler.
        await withCheckedContinuation { (sentinelCont: CheckedContinuation<Void, Never>) in
            do {
                let queue = pipelineQueue
                guard let cmdBuf = queue.makeCommandBuffer() else {
                    sentinelCont.resume()
                    return
                }
                cmdBuf.addCompletedHandler { _ in sentinelCont.resume() }
                cmdBuf.commit()
            }
        }
    }

    // MARK: - Chunked Prefill

    mutating func processChunkedInput(tokens: [Int32]) async throws -> ArraySlice<Int32> {
        let chunkSize = config.prefillChunkSize
        var remainingTokens = tokens[...]

        try logits.ensureCapacity(forContextLength: chunkSize)

        while remainingTokens.count > chunkSize {
            let chunk = Array(remainingTokens.prefix(chunkSize))
            try await _encodeChunk(tokens: chunk)
            remainingTokens = remainingTokens.dropFirst(chunkSize)
        }

        return remainingTokens
    }

    private mutating func _encodeChunk(tokens: [Int32]) async throws {
        let queryLength = tokens.count
        let currentStep = processedTokenCount

        let chunkID = InstrumentsProfiler.beginCustomInterval(
            name: "CoreAIPipelinedChunk",
            details: "step=\(currentStep) qLen=\(queryLength)"
        )

        // Write at the chunk's natural position so each chunk occupies a disjoint
        // region of inputTokensBuffer. Encode holds a live MTLBuffer reference — writing
        // all chunks at offset 0 would race with the GPU reading the previous chunk.
        let ptr = inputTokensBuffer.contents().bindMemory(
            to: Int32.self, capacity: processedTokenCount + queryLength)
        for (i, token) in tokens.enumerated() {
            ptr[processedTokenCount + i] = token
        }

        let cachePosBuffer = cachePositionBuffers[step % pipelineDepth]
        let posLength = processedTokenCount + queryLength

        // Build async values and encode
        let tokenShape = [1, queryLength]
        let tokenStrides = try resolvedStrides(descriptor: inputIdsBaseDesc, shape: tokenShape)
        let posShape = [1, posLength]
        let posStrides = try resolvedStrides(descriptor: positionIdsBaseDesc, shape: posShape)

        let tokenValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: inputTokensBuffer,
            byteOffset: processedTokenCount * MemoryLayout<Int32>.size,
            scalarType: .int32, shape: tokenShape, strides: tokenStrides)
        let posValue = unsafe InferenceFunction.AsyncValue(
            unsafeBuffer: cachePosBuffer, byteOffset: 0,
            scalarType: .int32, shape: posShape, strides: posStrides)

        let asyncInputs: [String: InferenceFunction.AsyncValue] = [
            inputIdsName: tokenValue, positionIdsName: posValue,
        ]

        let keyBuffer = kvCache.keyBinding.metalBuffer
        let keyShape = kvCache.keyBinding.layout.shape
        let keyStrides = kvCache.keyBinding.layout.strides
        let valBuffer = kvCache.valueBinding.metalBuffer
        let valShape = kvCache.valueBinding.layout.shape
        let valStrides = kvCache.valueBinding.layout.strides
        var keyState = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: keyBuffer, byteOffset: 0,
            scalarType: keyCacheScalarType, shape: keyShape, strides: keyStrides)
        var valState = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: valBuffer, byteOffset: 0,
            scalarType: valueCacheScalarType, shape: valShape, strides: valStrides)
        var asyncStates = InferenceFunction.AsyncMutableViews()
        asyncStates.insert(&keyState, for: keyCacheName)
        asyncStates.insert(&valState, for: valueCacheName)

        let logitsShape = [1, queryLength, vocabSize]
        let logitsStrides = try resolvedStrides(descriptor: logitsBaseDesc, shape: logitsShape)
        var logitsOutput = unsafe InferenceFunction.AsyncMutableValue(
            unsafeBuffer: logits.metalBuffer, byteOffset: 0,
            scalarType: .float16, shape: logitsShape, strides: logitsStrides)
        var asyncOutputs = InferenceFunction.AsyncMutableViews()
        asyncOutputs.insert(&logitsOutput, for: logitsOutputName)

        let _ = try function.encode(
            inputs: asyncInputs,
            states: consume asyncStates,
            outputViews: consume asyncOutputs,
            to: computeStream
        )

        processedTokenCount += queryLength
        step += 1
        InstrumentsProfiler.endCustomInterval(name: "CoreAIPipelinedChunk", signpostID: chunkID)
    }

    mutating func reset() {
        let span = InstrumentsProfiler.beginReset(engine: "CoreAI-Pipelined")
        processedTokenCount = 0
        step = 0
        cachedSampler = nil
        cachedSamplerTemperature = nil
        span.end()
    }

    // MARK: - Warmup

    mutating func performWarmup(queryLength: Int, samplingConfig: SamplingConfiguration?) async throws {
        let warmupStart = ContinuousClock.now
        let warmupSpan = InstrumentsProfiler.beginWarmup()

        // A single warmup at any shape primes the framework's internal caches
        // (reshape, kernel compilation, state pool). Benchmarks show no benefit
        // from warming every bucket shape — the jump from none→any is what matters.
        let defaultWarmupLength = 256

        let shapesToWarm: [Int]
        if queryLength > 0 {
            shapesToWarm = [queryLength]
        } else {
            shapesToWarm = [1, defaultWarmupLength]
        }

        CLILogger.log("Running warmup for \(shapesToWarm.count) shape(s)")

        let maxShape = shapesToWarm.last ?? 1
        try logits.ensureCapacity(forContextLength: maxShape)

        do {
            let queue = pipelineQueue
            if try kvCache.ensureCapacity(forContextLength: maxShape, queue: queue) {
                CLILogger.log("KV cache grew to \(kvCache.currentCapacity) for warmup")
            }
        }

        let warmupSampler = try MPSGraphSamplerFactory.makeSampler(
            device: device,
            vocabSize: config.vocabSize,
            temperature: samplingConfig?.temperature ?? 0
        )

        for shape in shapesToWarm {
            // Write dummy tokens
            let ptr = inputTokensBuffer.contents().bindMemory(to: Int32.self, capacity: shape)
            for i in 0..<shape { ptr[i] = 1 }

            let cachePosBuffer = cachePositionBuffers[step % pipelineDepth]
            let posLength = processedTokenCount + shape

            let tShape = [1, shape]
            let tStrides = try resolvedStrides(descriptor: inputIdsBaseDesc, shape: tShape)
            let pShape = [1, posLength]
            let pStrides = try resolvedStrides(descriptor: positionIdsBaseDesc, shape: pShape)

            let tokenValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: inputTokensBuffer, byteOffset: 0,
                scalarType: .int32, shape: tShape, strides: tStrides)
            let posValue = unsafe InferenceFunction.AsyncValue(
                unsafeBuffer: cachePosBuffer, byteOffset: 0,
                scalarType: .int32, shape: pShape, strides: pStrides)
            let asyncInputs: [String: InferenceFunction.AsyncValue] = [
                inputIdsName: tokenValue, positionIdsName: posValue,
            ]

            let keyBuffer = kvCache.keyBinding.metalBuffer
            let kShape = kvCache.keyBinding.layout.shape
            let kStrides = kvCache.keyBinding.layout.strides
            let valBuffer = kvCache.valueBinding.metalBuffer
            let vShape = kvCache.valueBinding.layout.shape
            let vStrides = kvCache.valueBinding.layout.strides
            var keyState = unsafe InferenceFunction.AsyncMutableValue(
                unsafeBuffer: keyBuffer, byteOffset: 0,
                scalarType: keyCacheScalarType, shape: kShape, strides: kStrides)
            var valState = unsafe InferenceFunction.AsyncMutableValue(
                unsafeBuffer: valBuffer, byteOffset: 0,
                scalarType: valueCacheScalarType, shape: vShape, strides: vStrides)
            var asyncStates = InferenceFunction.AsyncMutableViews()
            asyncStates.insert(&keyState, for: keyCacheName)
            asyncStates.insert(&valState, for: valueCacheName)

            let lShape = [1, shape, vocabSize]
            let lStrides = try resolvedStrides(descriptor: logitsBaseDesc, shape: lShape)
            var logitsOutput = unsafe InferenceFunction.AsyncMutableValue(
                unsafeBuffer: logits.metalBuffer, byteOffset: 0,
                scalarType: .float16, shape: lShape, strides: lStrides)
            var asyncOutputs = InferenceFunction.AsyncMutableViews()
            asyncOutputs.insert(&logitsOutput, for: logitsOutputName)

            let _ = try function.encode(
                inputs: asyncInputs,
                states: consume asyncStates,
                outputViews: consume asyncOutputs,
                to: computeStream
            )

            // Warm up argmax kernel using pipeline-matched decode buffers
            let warmupLogitsBuffer = decodeLogitsBuffers[step % pipelineDepth]
            let warmupOutputBuffer = decodeOutputBuffers[step % pipelineDepth]
            let logitsOffset = (shape - 1) * vocabSize * MemoryLayout<UInt16>.size

            do {
                let queue = pipelineQueue
                warmupSampler.encode(
                    to: queue,
                    logitsBuffer: warmupLogitsBuffer,
                    logitsOffset: logitsOffset,
                    outputBuffer: warmupOutputBuffer,
                    outputOffset: 0,
                    completion: { _ in }
                )
            }

            step += 1
        }

        await computeStream.currentWorkCompleted()
        reset()

        warmupSpan.end()
        let warmupElapsed = milliseconds(since: warmupStart)
        CLILogger.log(
            "CoreAI pipelined warmup complete (\(shapesToWarm.count) shapes): \(String(format: "%.2f", warmupElapsed))ms"
        )
    }
}

extension CoreAIPipelinedEngine {
    /// Async sequence of `InferenceOutput` produced by `generate()`.
    ///
    /// Unlike the CPU engines, the pipelined engine samples on-device and drives
    /// output from a producer `Task`, so this sequence forwards an underlying
    /// `AsyncThrowingStream`. The producer records the `stopReason` directly.
    public struct GenerationSequence: InferenceOutputSequence {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        let base: AsyncThrowingStream<InferenceOutput, any Error>
        let stopReasonStore: StopReasonStore

        public var stopReason: StopReason? { stopReasonStore.stopReason }

        public func setStopReason(_ reason: StopReason) {
            stopReasonStore.set(reason)
        }

        public func makeAsyncIterator() -> Iterator {
            Iterator(base: base.makeAsyncIterator(), stopReasonStore: stopReasonStore)
        }
    }
}

extension CoreAIPipelinedEngine.GenerationSequence {
    public struct Iterator: AsyncIteratorProtocol {
        public typealias Element = InferenceOutput
        public typealias Failure = Error

        var base: AsyncThrowingStream<InferenceOutput, any Error>.AsyncIterator
        let stopReasonStore: StopReasonStore

        public mutating func next() async throws -> InferenceOutput? {
            do {
                return try await base.next()
            } catch is CancellationError {
                // The producer Task is independent and won't observe the
                // consumer's cancellation, so record it from the consumer side.
                stopReasonStore.set(.cancelled)
                throw CancellationError()
            } catch {
                stopReasonStore.set(.error)
                throw error
            }
        }
    }
}
