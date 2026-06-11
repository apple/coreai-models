// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Embeddings Inference Engine for Qwen3-VL.
// Accepts inputs_embeds (f16) instead of input_ids (i32) for vision-language prefill.

import CoreAI
import CoreAIShared
import Foundation
import Metal



// MARK: - Embeddings Inference Engine

/// CoreAI inference engine for models that accept `inputs_embeds` (f16) instead of `input_ids`.
/// Used for Qwen3-VL where vision features are fused into the embedding sequence externally.
#if canImport(CoreAI)
@available(macOS 27, iOS 27, *)
public final class EmbeddingsInferenceEngine: @unchecked Sendable {
    public var vocabSize: Int { config.vocabSize }
    public let config: ModelConfig

    private let function: InferenceFunction
    private let functionDescriptor: InferenceFunctionDescriptor

    private let inputEmbedsName: String
    private let positionIdsName: String
    private let keyCacheName: String
    private let valueCacheName: String
    private let logitsName: String
    // Explicit KV mode: these are the output names for updated KV caches
    private let keyCacheOutputName: String
    private let valueCacheOutputName: String
    private let explicitKV: Bool

    private let inputEmbedsDescriptor: NDArrayDescriptor
    private let positionIdsDescriptor: NDArrayDescriptor
    private let logitsDescriptor: NDArrayDescriptor
    private let keyCacheDescriptor: NDArrayDescriptor
    private let valueCacheDescriptor: NDArrayDescriptor

    private let hiddenSize: Int
    private let positionIdsRank: Int

    private var keyCache: NDArray
    private var valueCache: NDArray
    private var logitsArray: NDArray
    private var currentKVCapacity: Int
    private var processedTokenCount: Int = 0
    private var embedTable: UnsafeBufferPointer<LogitsScalarType>?
    private var embedTableData: Data?
    private var prefillDone: Bool = false

    public init(
        config: ModelConfig,
        modelURL: URL,
        embedTableURL: URL?,
        options: EngineOptions = EngineOptions()
    ) async throws {
        self.config = config

        let model = try await AIModel(contentsOf: modelURL, options: .default)

        guard let descriptor = model.functionDescriptor(for: config.function) else {
            throw InferenceRuntimeError.functionNotFound(config.function)
        }
        self.functionDescriptor = descriptor

        // Detect mode: stateful (2 inputs, 2 states) vs explicit KV (4 inputs, 0 states)
        let isExplicitKV = descriptor.stateNames.isEmpty && descriptor.inputNames.count == 4
        let isStateful = descriptor.stateNames.count == 2 && descriptor.inputNames.count == 2
        guard isStateful || isExplicitKV else {
            throw InferenceRuntimeError.genericError(
                "Unexpected signature: inputs=\(descriptor.inputNames.count) states=\(descriptor.stateNames.count). " +
                "Expected stateful(2in+2states) or explicit-KV(4in+0states).")
        }
        self.explicitKV = isExplicitKV
        CLILogger.log("EmbeddingsEngine: mode=\(isExplicitKV ? "explicit-KV" : "stateful"), inputs=\(descriptor.inputNames), states=\(descriptor.stateNames), outputs=\(descriptor.outputNames)")

        if isExplicitKV {
            self.inputEmbedsName = descriptor.inputNames[0]
            self.positionIdsName = descriptor.inputNames[1]
            self.keyCacheName = descriptor.inputNames[2]
            self.valueCacheName = descriptor.inputNames[3]
            // Outputs: [keyCache_out, valueCache_out, logits]
            self.logitsName = descriptor.outputNames[2]
            self.keyCacheOutputName = descriptor.outputNames[0]
            self.valueCacheOutputName = descriptor.outputNames[1]
        } else {
            self.inputEmbedsName = descriptor.inputNames[0]
            self.positionIdsName = descriptor.inputNames[1]
            self.keyCacheName = descriptor.stateNames[0]
            self.valueCacheName = descriptor.stateNames[1]
            self.logitsName = descriptor.outputNames[0]
            self.keyCacheOutputName = ""
            self.valueCacheOutputName = ""
        }

        guard case .ndArray(let embedsDesc) = descriptor.inputDescriptor(of: inputEmbedsName) else {
            throw InferenceRuntimeError.genericError("Cannot get descriptor for '\(inputEmbedsName)'")
        }
        self.inputEmbedsDescriptor = embedsDesc

        guard case .ndArray(let posDesc) = descriptor.inputDescriptor(of: positionIdsName) else {
            throw InferenceRuntimeError.genericError("Cannot get descriptor for '\(positionIdsName)'")
        }
        self.positionIdsDescriptor = posDesc

        guard case .ndArray(let logitsDesc) = descriptor.outputDescriptor(of: logitsName) else {
            throw InferenceRuntimeError.genericError("Cannot get descriptor for '\(logitsName)'")
        }
        self.logitsDescriptor = logitsDesc

        // Get KV cache descriptors — from states (stateful) or inputs (explicit KV)
        let keyCacheDesc: NDArrayDescriptor
        let valueCacheDesc: NDArrayDescriptor
        if isExplicitKV {
            guard case .ndArray(let kd) = descriptor.inputDescriptor(of: keyCacheName),
                  case .ndArray(let vd) = descriptor.inputDescriptor(of: valueCacheName)
            else { throw InferenceRuntimeError.genericError("Cannot get KV cache input descriptors") }
            keyCacheDesc = kd; valueCacheDesc = vd
        } else {
            guard case .ndArray(let kd) = descriptor.stateDescriptor(of: keyCacheName),
                  case .ndArray(let vd) = descriptor.stateDescriptor(of: valueCacheName)
            else { throw InferenceRuntimeError.genericError("Cannot get KV cache state descriptors") }
            keyCacheDesc = kd; valueCacheDesc = vd
        }
        self.keyCacheDescriptor = keyCacheDesc
        self.valueCacheDescriptor = valueCacheDesc

        self.hiddenSize = embedsDesc.shape[2]
        self.positionIdsRank = posDesc.shape.count

        let isDynamic = keyCacheDesc.shape.contains { $0 < 0 }
        let initialCapacity = isDynamic ? min(256, config.maxContextLength) : config.maxContextLength
        self.currentKVCapacity = initialCapacity

        let resolvedKeyDesc = isDynamic
            ? keyCacheDesc.resolvingDynamicDimensions(keyCacheDesc.shape.map { $0 < 0 ? initialCapacity : $0 })
            : keyCacheDesc
        let resolvedValueDesc = isDynamic
            ? valueCacheDesc.resolvingDynamicDimensions(valueCacheDesc.shape.map { $0 < 0 ? initialCapacity : $0 })
            : valueCacheDesc

        var keyCache = NDArray(descriptor: resolvedKeyDesc)
        var valueCache = NDArray(descriptor: resolvedValueDesc)
        let kcCount = resolvedKeyDesc.shape.reduce(1, *)
        let vcCount = resolvedValueDesc.shape.reduce(1, *)
        fillNDArray(&keyCache, as: LogitsScalarType.self, count: kcCount) { _ in LogitsScalarType(0) }
        fillNDArray(&valueCache, as: LogitsScalarType.self, count: vcCount) { _ in LogitsScalarType(0) }
        self.keyCache = keyCache
        self.valueCache = valueCache

        let initLogitsDesc = logitsDesc.resolvingDynamicDimensions([1, 1, config.vocabSize])
        self.logitsArray = NDArray(descriptor: initLogitsDesc)

        guard let fn = try model.loadFunction(named: config.function) else {
            throw InferenceRuntimeError.functionNotFound(config.function)
        }
        self.function = fn

        if let embedTableURL = embedTableURL {
            try loadEmbedTable(from: embedTableURL)
        }
    }

    // MARK: - Embedding Table

    private func loadEmbedTable(from url: URL) throws {
        let data = try Data(contentsOf: url)
        let expectedSize = config.vocabSize * hiddenSize * MemoryLayout<LogitsScalarType>.size
        guard data.count == expectedSize else {
            throw InferenceRuntimeError.genericError(
                "embed_tokens size mismatch: got \(data.count), expected \(expectedSize)")
        }
        self.embedTableData = data
        data.withUnsafeBytes { rawPtr in
            let ptr = rawPtr.bindMemory(to: LogitsScalarType.self)
            self.embedTable = UnsafeBufferPointer(start: ptr.baseAddress, count: ptr.count)
        }
    }

    // MARK: - KV Cache Growth

    private func ensureKVCapacity(forContextLength needed: Int) throws {
        guard needed > currentKVCapacity else { return }
        guard needed <= config.maxContextLength else {
            throw InferenceRuntimeError.contextLengthExceeded(needed, config.maxContextLength)
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
        copyCache(from: keyCache, to: &newKeyCache)
        copyCache(from: valueCache, to: &newValueCache)
        keyCache = newKeyCache
        valueCache = newValueCache
        currentKVCapacity = newCapacity
    }

    private func copyCache(from source: NDArray, to destination: inout NDArray) {
        let srcShape = source.shape
        let dstShape = destination.shape
        // Sequence dim is 3 for 5D KV cache [layers, batch, heads, seq, head_dim]
        let seqDim = srcShape.count == 5 ? 3 : 2
        let headDim = srcShape.last!
        let numBlocks = srcShape[..<seqDim].reduce(1, *)
        let oldSeqLen = srcShape[seqDim]
        let copySize = oldSeqLen * headDim
        let srcBlockStride = srcShape[seqDim...].reduce(1, *)
        let dstBlockStride = dstShape[seqDim...].reduce(1, *)
        source.view(as: LogitsScalarType.self).withUnsafePointer { srcPtr, _, _ in
            destination.view(as: LogitsScalarType.self).withUnsafePointer { dstPtr, _, _ in
                let dst = UnsafeMutablePointer(mutating: dstPtr)
                for block in 0..<numBlocks {
                    dst.advanced(by: block * dstBlockStride)
                        .update(from: srcPtr.advanced(by: block * srcBlockStride), count: copySize)
                }
            }
        }
    }

    // MARK: - Execute Forward Pass

    private func executeEmbeddingsBatch(
        embeddings: Data, byteOffset: Int, byteCount: Int, batchSize: Int
    ) async throws -> [LogitsScalarType] {
        try ensureKVCapacity(forContextLength: processedTokenCount + batchSize)

        let resolvedEmbedsDesc = inputEmbedsDescriptor.resolvingDynamicDimensions([1, batchSize, hiddenSize])
        var inputEmbedsArray = NDArray(descriptor: resolvedEmbedsDesc)
        embeddings.withUnsafeBytes { rawPtr in
            var view = inputEmbedsArray.mutableView(as: LogitsScalarType.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                let src = rawPtr.baseAddress! + byteOffset
                let count = byteCount / MemoryLayout<LogitsScalarType>.size
                src.assumingMemoryBound(to: LogitsScalarType.self)
                    .withMemoryRebound(to: LogitsScalarType.self, capacity: count) { srcPtr in
                        ptr.update(from: srcPtr, count: count)
                    }
            }
        }

        let totalPositions = processedTokenCount + batchSize
        let posShape: [Int] = positionIdsRank == 3 ? [1, 3, totalPositions] : [1, totalPositions]
        let resolvedPosDesc = positionIdsDescriptor.resolvingDynamicDimensions(posShape)
        var positionIdsArray = NDArray(descriptor: resolvedPosDesc)
        fillNDArray(&positionIdsArray, as: Int32.self, count: posShape.reduce(1, *)) { idx in
            Int32(idx % totalPositions)
        }

        let resolvedLogitsDesc = logitsDescriptor.resolvingDynamicDimensions([1, batchSize, config.vocabSize])
        logitsArray = NDArray(descriptor: resolvedLogitsDesc)

        if explicitKV {
            // Explicit KV mode: pass KV cache as inputs, read updated cache from outputs
            var keyCacheOut = NDArray(descriptor: keyCacheDescriptor)
            var valueCacheOut = NDArray(descriptor: valueCacheDescriptor)
            var outputBackings = InferenceFunction.MutableViews()
            outputBackings.insert(&keyCacheOut, for: keyCacheOutputName)
            outputBackings.insert(&valueCacheOut, for: valueCacheOutputName)
            outputBackings.insert(&logitsArray, for: logitsName)
            _ = try await function.run(
                inputs: [inputEmbedsName: inputEmbedsArray, positionIdsName: positionIdsArray,
                         keyCacheName: keyCache, valueCacheName: valueCache],
                states: InferenceFunction.MutableViews(),
                outputViews: consume outputBackings
            )
            keyCache = keyCacheOut
            valueCache = valueCacheOut
        } else {
            // Stateful mode: KV cache managed as states
            var states = InferenceFunction.MutableViews()
            states.insert(&keyCache, for: keyCacheName)
            states.insert(&valueCache, for: valueCacheName)
            var outputBackings = InferenceFunction.MutableViews()
            outputBackings.insert(&logitsArray, for: logitsName)
            _ = try await function.run(
                inputs: [inputEmbedsName: inputEmbedsArray, positionIdsName: positionIdsArray],
                states: consume states,
                outputViews: consume outputBackings
            )
        }

        let totalLogits = batchSize * config.vocabSize
        let logitBuffer = readNDArray(logitsArray, as: LogitsScalarType.self, count: totalLogits)
        processedTokenCount += batchSize
        return logitBuffer
    }

    private func executeTokenBatch(tokenId: Int32) async throws -> [LogitsScalarType] {
        guard let table = embedTable else {
            fatalError("EmbeddingsEngine: embed_tokens not loaded")
        }

        try ensureKVCapacity(forContextLength: processedTokenCount + 1)

        let resolvedEmbedsDesc = inputEmbedsDescriptor.resolvingDynamicDimensions([1, 1, hiddenSize])
        var inputEmbedsArray = NDArray(descriptor: resolvedEmbedsDesc)
        let offset = Int(tokenId) * hiddenSize
        var view = inputEmbedsArray.mutableView(as: LogitsScalarType.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(from: table.baseAddress! + offset, count: hiddenSize)
        }

        let totalPositions = processedTokenCount + 1
        let posShape: [Int] = positionIdsRank == 3 ? [1, 3, totalPositions] : [1, totalPositions]
        let resolvedPosDesc = positionIdsDescriptor.resolvingDynamicDimensions(posShape)
        var positionIdsArray = NDArray(descriptor: resolvedPosDesc)
        fillNDArray(&positionIdsArray, as: Int32.self, count: posShape.reduce(1, *)) { idx in
            Int32(idx % totalPositions)
        }

        let resolvedLogitsDesc = logitsDescriptor.resolvingDynamicDimensions([1, 1, config.vocabSize])
        logitsArray = NDArray(descriptor: resolvedLogitsDesc)

        if explicitKV {
            var keyCacheOut = NDArray(descriptor: keyCacheDescriptor)
            var valueCacheOut = NDArray(descriptor: valueCacheDescriptor)
            var outputBackings = InferenceFunction.MutableViews()
            outputBackings.insert(&keyCacheOut, for: keyCacheOutputName)
            outputBackings.insert(&valueCacheOut, for: valueCacheOutputName)
            outputBackings.insert(&logitsArray, for: logitsName)
            _ = try await function.run(
                inputs: [inputEmbedsName: inputEmbedsArray, positionIdsName: positionIdsArray,
                         keyCacheName: keyCache, valueCacheName: valueCache],
                states: InferenceFunction.MutableViews(),
                outputViews: consume outputBackings
            )
            keyCache = keyCacheOut
            valueCache = valueCacheOut
        } else {
            var states = InferenceFunction.MutableViews()
            states.insert(&keyCache, for: keyCacheName)
            states.insert(&valueCache, for: valueCacheName)
            var outputBackings = InferenceFunction.MutableViews()
            outputBackings.insert(&logitsArray, for: logitsName)
            _ = try await function.run(
                inputs: [inputEmbedsName: inputEmbedsArray, positionIdsName: positionIdsArray],
                states: consume states,
                outputViews: consume outputBackings
            )
        }

        let logitBuffer = readNDArray(logitsArray, as: LogitsScalarType.self, count: config.vocabSize)
        processedTokenCount += 1
        return logitBuffer
    }

    // MARK: - Public Prefill API

    /// Prefill with pre-computed combined embeddings (vision + text).
    public func prefillWithEmbeddings(
        _ embeddings: Data,
        seqLen: Int,
        progressHandler: ((Int, Int) -> Void)? = nil
    ) async throws -> [LogitsScalarType] {
        let chunkSize = min(seqLen, 32)

        var lastLogits: [LogitsScalarType] = []
        var offset = 0
        let reportInterval = max(1, seqLen / 10)

        while offset < seqLen {
            let currentChunk = min(chunkSize, seqLen - offset)
            let byteOffset = offset * hiddenSize * MemoryLayout<LogitsScalarType>.size
            let byteCount = currentChunk * hiddenSize * MemoryLayout<LogitsScalarType>.size
            lastLogits = try await executeEmbeddingsBatch(
                embeddings: embeddings, byteOffset: byteOffset, byteCount: byteCount, batchSize: currentChunk)
            offset += currentChunk
            if offset % reportInterval < currentChunk || offset == seqLen {
                progressHandler?(min(offset, seqLen), seqLen)
            }
        }

        // Extract last-token logits
        let tokensInLast = lastLogits.count / config.vocabSize
        let lastOffset = (tokensInLast - 1) * config.vocabSize
        prefillDone = true
        return Array(lastLogits[lastOffset..<(lastOffset + config.vocabSize)])
    }

    // MARK: - InferenceEngine Protocol

    public func inference(
        inputTokens: [Int32], samplingConfig: SamplingConfiguration, returnsLogits: Bool
    ) async throws -> (logits: [LogitsScalarType]?, token: Int32) {
        guard prefillDone else {
            throw InferenceRuntimeError.genericError("Must call prefillWithEmbeddings before inference()")
        }
        let logits = try await executeTokenBatch(tokenId: inputTokens.last!)
        var mutableLogits = logits
        let nextToken = samplingConfig.fallbackSampler(from: &mutableLogits)
        return (logits: returnsLogits ? logits : nil, token: nextToken)
    }

    public func reset() async throws {
        processedTokenCount = 0
        prefillDone = false
        zeroFill(&keyCache)
        zeroFill(&valueCache)
    }

    public func cleanup() {
        embedTable = nil
        embedTableData = nil
    }

    public func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {}
    public func validateSamplingStrategy(_ config: SamplingConfiguration) throws {}

    private func zeroFill(_ array: inout NDArray) {
        let count = array.shape.reduce(1, *)
        fillNDArray(&array, as: LogitsScalarType.self, count: count) { _ in LogitsScalarType(0) }
    }
}



#endif
