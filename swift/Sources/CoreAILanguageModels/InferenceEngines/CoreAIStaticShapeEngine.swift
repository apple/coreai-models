// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Static-shape inference engine using Core AI models.
public final class StaticShapeEngine: InferenceEngine, @unchecked Sendable {
    public typealias ConfigType = ModelConfig

    public var supportsLogits: Bool { true }

    // MARK: I/O name contracts — models must use these exact names

    private static let logitsOutputName = "out_logits"
    private static let keyCacheName = "key_cache"
    private static let valueCacheName = "value_cache"

    public var vocabSize: Int { config.vocabSize }

    public let config: ModelConfig
    private let model: AIModel

    // MARK: Properties

    // Lazily loaded inference functions, keyed by name.
    private var functions: [String: InferenceFunction]

    // Available function names by category.
    // Extend functions are sorted by query length (ascending) for graph selection.
    private let extendFunctionNames: [String]
    private let gatherFunctionNames: Set<String>

    // Embedding table loaded once at init.
    private let embeddingTable: NDArray

    // Largest query length across all extend functions — used as prefill threshold.
    private let maxQueryLength: Int

    // Fixed size caches shared across all decoding functions.
    private var keyCache: NDArray
    private var valueCache: NDArray

    // Number of tokens already processed in the current sequence.
    private var processedTokenCount: Int = 0

    // MARK: - Initialization

    public init(configuration: ModelConfig, preparedModel: PreparedModel) async throws {
        self.config = configuration
        self.model = preparedModel.model
        self.functions = [:]

        let allNames = model.functionNames
        CLILogger.log("Model loaded: \(allNames.count) functions: \(allNames.sorted())")

        // Categorize functions
        self.extendFunctionNames =
            allNames
            .filter { $0.hasPrefix("extend") || $0.hasPrefix("prompt") }
            .sorted()
        self.gatherFunctionNames = Set(allNames.filter { $0.hasPrefix("gather_embeddings") })

        CLILogger.log(
            "Parsed \(extendFunctionNames.count) decoder functions, \(gatherFunctionNames.count) gather functions")

        // Compute max query length from function names for prefill threshold
        self.maxQueryLength =
            extendFunctionNames.compactMap { name -> Int? in
                let parts = name.split(separator: "_")
                return parts.last.flatMap { Int($0) }
            }.max() ?? 64

        // Grab largest context length extend function to use the descriptors for allocating largest context length
        // key/value caches.
        var largestContextExtend: (name: String, descriptor: InferenceFunctionDescriptor)?
        for name in extendFunctionNames {
            let desc = try Self.requireDescriptor(model: model, functionName: name)
            if Self.contextLength(descriptor: desc, config: configuration) == configuration.maxContextLength {
                largestContextExtend = (name, desc)
                break
            }
        }
        guard let (largestExtendName, largestExtendDescriptor) = largestContextExtend else {
            throw InferenceRuntimeError.invalidState(
                "Failed to find an extend function with the max context length of \(configuration.maxContextLength)")
        }

        // Validate output/state contract against the max-context function
        try Self.validateIOContract(descriptor: largestExtendDescriptor, functionName: largestExtendName)

        // Load embeddings
        self.embeddingTable = try await Self.loadEmbeddingTable(from: model)

        // Allocate KV cache IOSurfaces sized to the max-context descriptor
        if case .ndArray(let keyCacheDescriptor) = largestExtendDescriptor.stateDescriptor(of: Self.keyCacheName),
            case .ndArray(let valueCacheDescriptor) = largestExtendDescriptor.stateDescriptor(of: Self.valueCacheName)
        {
            self.keyCache = NDArray(descriptor: keyCacheDescriptor)
            self.valueCache = NDArray(descriptor: valueCacheDescriptor)
            CLILogger.log(
                "KV cache allocated: key \(keyCacheDescriptor.minimumByteCount) bytes, value \(valueCacheDescriptor.minimumByteCount) bytes (IOSurface)"
            )
        } else {
            throw InferenceRuntimeError.invalidState(
                "No KV cache state descriptors found — cannot allocate cache buffers")
        }

        CLILogger.log("Engine initialized")
    }

    public convenience init(configuration: ModelConfig, modelURL: URL) async throws {
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        try await self.init(configuration: configuration, preparedModel: preparedModel)
    }

    // MARK: - Initialization Helpers

    private static func requireDescriptor(
        model: AIModel, functionName: String
    ) throws -> InferenceFunctionDescriptor {
        guard let desc = model.functionDescriptor(for: functionName) else {
            throw InferenceRuntimeError.invalidState("Cannot find descriptor for '\(functionName)'")
        }
        return desc
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
        descriptor: InferenceFunctionDescriptor, functionName: String
    ) throws {
        guard descriptor.outputNames.contains(logitsOutputName) else {
            throw InferenceRuntimeError.invalidState(
                "Function '\(functionName)' missing required output '\(logitsOutputName)'. "
                    + "Available outputs: \(descriptor.outputNames)")
        }
        if descriptor.stateNames.count == 1 {
            throw InferenceRuntimeError.invalidState(
                "Function '\(functionName)' has exactly 1 state (\(descriptor.stateNames)) "
                    + "— expected 0 (internal to model) or 2 (\(keyCacheName), \(valueCacheName))")
        }
        if descriptor.stateNames.count >= 2 {
            guard descriptor.stateNames.contains(keyCacheName),
                descriptor.stateNames.contains(valueCacheName)
            else {
                throw InferenceRuntimeError.invalidState(
                    "Function '\(functionName)' has states \(descriptor.stateNames) "
                        + "but missing required '\(keyCacheName)' and/or '\(valueCacheName)'")
            }
        }
    }

    private static func loadEmbeddingTable(from model: AIModel) async throws -> NDArray {
        CLILogger.log("Loading embeddings...")
        guard let embeddingFunction = try model.loadFunction(named: "load_embeddings") else {
            throw InferenceRuntimeError.invalidState("Cannot load 'load_embeddings'")
        }

        guard case .ndArray(let embeddingDesc) = embeddingFunction.descriptor.outputDescriptor(of: "embedding_table")
        else {
            throw InferenceRuntimeError.invalidState(
                "load_embeddings has no 'embedding_table' ndArray output descriptor")
        }
        var embeddingArray = NDArray(descriptor: embeddingDesc)

        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&embeddingArray, for: "embedding_table")

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

    /// Returns the query length for a given function by reading the
    /// `transformer_input` descriptor's sequence dimension.
    private func queryLength(of functionName: String) throws -> Int {
        let desc = try functionDescriptor(for: functionName)
        if let txName = desc.inputNames.first(where: { $0.contains("transformer_input") }),
            case .ndArray(let nd) = desc.inputDescriptor(of: txName), nd.shape.count >= 2
        {
            return nd.shape[1]
        }
        // Fallback: parse from function name (extend_<ctx>_<seq>)
        let parts = functionName.split(separator: "_")
        if let last = parts.last, let seq = Int(last) { return seq }
        return 1
    }

    /// Returns the context length for a given function by reading the
    /// key_cache state descriptor.
    private func contextLength(of functionName: String) throws -> Int {
        let desc = try functionDescriptor(for: functionName)
        return Self.contextLength(descriptor: desc, config: config)
    }

    private static func contextLength(
        model: AIModel, functionName: String, config: ModelConfig
    ) throws -> Int {
        guard let desc = model.functionDescriptor(for: functionName) else {
            return config.maxContextLength
        }
        return contextLength(descriptor: desc, config: config)
    }

    private static func contextLength(
        descriptor: InferenceFunctionDescriptor, config: ModelConfig
    ) -> Int {
        if case .ndArray(let keyDesc) = descriptor.stateDescriptor(of: keyCacheName) {
            if keyDesc.shape.contains(-1) {
                return config.maxContextLength
            }
            return keyDesc.shape.max() ?? config.maxContextLength
        }
        return config.maxContextLength
    }

    // MARK: - Graph Selection

    private func forwardGraph(numInputTokens: Int, currentPosition: Int, isPrefill: Bool) throws -> String {
        var pairs: [(contextLength: Int, queryLength: Int)] = []
        for name in extendFunctionNames {
            let parts = Array(name.split(separator: "_").suffix(2))
            guard parts.count == 2, let maxCtx = Int(parts[0]), let seqLen = Int(parts[1]) else { continue }
            pairs.append((maxCtx, seqLen))
        }

        let sorted = pairs.sorted { $0.queryLength < $1.queryLength }
        guard let maxPair = sorted.last else {
            throw InferenceRuntimeError.invalidState(
                "No extend functions found in static-shape engine")
        }
        let selectedSeq =
            sorted.first(where: { $0.queryLength >= numInputTokens })?.queryLength
            ?? maxPair.queryLength
        let candidates = pairs.filter { $0.queryLength == selectedSeq }

        guard
            let selected =
                candidates
                .sorted(by: { $0.contextLength < $1.contextLength })
                .first(where: { $0.contextLength > currentPosition })
        else {
            throw InferenceRuntimeError.invalidState(
                "No graph with cache_len > \(currentPosition) and seq_len = \(selectedSeq)")
        }
        return isPrefill
            ? "prompt_opt_\(selected.contextLength)_\(selected.queryLength)"
            : "extend_\(selected.contextLength)_\(selected.queryLength)"
    }

    // MARK: - Causal Mask

    private static func fillCausalMask(
        _ view: inout NDArray.MutableView<LogitsScalarType>,
        tokensInBatch: Int,
        alignedStep: Int
    ) {
        view.withUnsafeMutablePointer { ptr, shape, strides in
            // Stride-aware indexing for non-contiguous strides
            for context in 0..<shape[1] {
                for query in 0..<shape[3] {
                    let offset = context &* strides[1] &+ query &* strides[3]
                    ptr[offset] = LogitsScalarType(-40000.0)
                }
            }

            // Unmask positions where attention is allowed
            for query in 0..<tokensInBatch {
                let queryPos = alignedStep + query
                let upperBound = min(queryPos, shape[1] &- 1)
                for context in 0...upperBound {
                    let offset = context &* strides[1] &+ query &* strides[3]
                    ptr[offset] = 0
                }
            }
        }
    }

    // MARK: - Generate (primary API)

    public func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> InferenceStream {
        let (stream, continuation) = InferenceStream.makeStream()
        Task {
            do {
                let forced = inferenceOptions.forcedContinuation
                let maxTokens: Int
                if let forced {
                    maxTokens = forced.count
                } else {
                    maxTokens = min(
                        inferenceOptions.maxTokens ?? Int.max,
                        max(0, self.config.maxContextLength - input.count)
                    )
                }
                let returnsLogits = inferenceOptions.includeLogits
                var inputTokens = input

                for i in 0..<maxTokens {
                    try Task.checkCancellation()
                    let (logits, sampledToken) = try await self.inference(
                        inputTokens: inputTokens,
                        samplingConfig: samplingConfiguration,
                        returnsLogits: returnsLogits || forced != nil
                    )

                    let nextToken = forced?[i] ?? sampledToken

                    continuation.yield(
                        InferenceOutput(
                            tokenId: nextToken,
                            logits: returnsLogits ? logits : nil
                        ))
                    inputTokens.append(nextToken)
                }
                stream.setStopReason(.maxTokens)
                continuation.finish()
            } catch is CancellationError {
                stream.setStopReason(.cancelled)
                continuation.finish()
            } catch {
                stream.setStopReason(.error)
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    // MARK: - Inference

    public func inference(
        inputTokens: [Int32], samplingConfig: SamplingConfiguration, returnsLogits: Bool
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
                tokensInBatch: tokensInBatch
            )
            prepareSpan.end()

            let logitsSpan = InstrumentsProfiler.beginLogitsInference(
                step: batchStartToken, tokens: tokensInBatch, engine: "StaticShape")

            let fn = try loadFunction(named: graphName)
            let desc = try functionDescriptor(for: graphName)

            guard case .ndArray(let keyCacheDescriptor) = desc.stateDescriptor(of: Self.keyCacheName),
                case .ndArray(let valueCacheDescriptor) = desc.stateDescriptor(of: Self.valueCacheName)
            else {
                throw InferenceRuntimeError.invalidState("Missing KV cache state descriptors for '\(graphName)'")
            }

            // Create MutableRawView using this function's descriptor for shape metadata.
            // No copy is needed on graph switch because all extend functions share the
            // same KV cache shape, strides, and interleaveLayout
            let keyCacheView = keyCache.mutableRawView().slice(at: keyCacheDescriptor.shape.map { 0..<$0 })
            let valueCacheView = valueCache.mutableRawView().slice(at: valueCacheDescriptor.shape.map { 0..<$0 })

            var states = InferenceFunction.MutableViews()
            states.insert(keyCacheView, for: Self.keyCacheName)
            states.insert(valueCacheView, for: Self.valueCacheName)
            var outputs = try await fn.run(
                inputs: inputs,
                states: consume states,
                outputViews: InferenceFunction.MutableViews()
            )

            let logitsArray = outputs.remove(Self.logitsOutputName)?.ndArray
            logitsSpan.end()

            // Extract logits from the last token position.
            if !usePrefill, let logitsArray {
                let logitsView = logitsArray.view(as: LogitsScalarType.self)
                guard let logits = logitsView.contiguousElements else {
                    throw InferenceRuntimeError.invalidState(
                        "Logits array has non-contiguous (interleaved) layout — cannot extract values safely")
                }
                let offset = (tokensInBatch - 1) * config.vocabSize
                for i in 0..<config.vocabSize {
                    logitBuffer[i] = logits[offset + i]
                }
            }

            currentPosition = batchEndToken + 1
            processedTokenCount = currentPosition
        }

        let actualLogits = returnsLogits ? logitBuffer : nil
        let sampleSpan = InstrumentsProfiler.beginSample(strategy: "cpu-fallback")
        let nextToken = samplingConfig.fallbackSampler(from: &logitBuffer)
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
        tokensInBatch: Int
    ) async throws -> [String: NDArray] {
        let desc = try functionDescriptor(for: graphName)
        var inputs = [String: NDArray]()

        if desc.inputNames.contains("embedding_table") {
            inputs["embedding_table"] = embeddingTable
        }

        // Gather embeddings for this batch's tokens
        if let txName = desc.inputNames.first(where: { $0.contains("transformer_input") }) {
            let gatherName = "gather_embeddings_\(batchSize)"
            guard gatherFunctionNames.contains(gatherName) else {
                throw InferenceRuntimeError.invalidState(
                    "No gather function '\(gatherName)' for batch size \(batchSize)")
            }
            guard let gathered = try await runGather(tokenIDs: Array(batchTokens), batchSize: batchSize) else {
                throw InferenceRuntimeError.invalidState("Gather '\(gatherName)' returned no output")
            }
            inputs[txName] = gathered
        }

        // Position IDs
        guard let posName = desc.inputNames.first(where: { $0.contains("pos") }) else {
            throw InferenceRuntimeError.invalidState("Graph '\(graphName)' has no position_ids input")
        }
        if case .ndArray(let nd) = desc.inputDescriptor(of: posName) {
            var pos = NDArray(descriptor: nd)
            var posView = pos.mutableView(as: UInt16.self)
            guard var posSpan = posView.contiguousElements else {
                throw InferenceRuntimeError.invalidState("pos array has non-contiguous layout")
            }
            for i in 0..<batchSize {
                posSpan[i] = UInt16(alignedStep + i)
            }
            inputs[posName] = pos
        }

        // Causal mask
        if case .ndArray(let nd) = desc.inputDescriptor(of: "causal_mask") {
            var mask = NDArray(descriptor: nd)
            var maskView = mask.mutableView(as: LogitsScalarType.self)
            Self.fillCausalMask(&maskView, tokensInBatch: tokensInBatch, alignedStep: alignedStep)
            inputs["causal_mask"] = mask
        }

        // Step
        if let stepName = desc.inputNames.first(where: { $0.contains("step") && !$0.contains("pos") }),
            case .ndArray(let nd) = desc.inputDescriptor(of: stepName)
        {
            var step = NDArray(descriptor: nd)
            var stepView = step.mutableView(as: Int32.self)
            guard var stepSpan = stepView.contiguousElements else {
                throw InferenceRuntimeError.invalidState("step array has non-contiguous layout")
            }
            stepSpan[0] = Int32(alignedStep)
            inputs[stepName] = step
        }

        return inputs
    }

    // MARK: - Gather Embeddings

    private func runGather(tokenIDs: [Int32], batchSize: Int) async throws -> NDArray? {
        let name = "gather_embeddings_\(batchSize)"
        let fn = try loadFunction(named: name)
        let desc = try functionDescriptor(for: name)

        // Token IDs input
        let tokenInputName = "in_new_token_ids"
        guard let tokenDesc = desc.inputDescriptor(of: tokenInputName),
            case .ndArray(let tokenNDDesc) = tokenDesc
        else {
            throw InferenceRuntimeError.invalidState("No descriptor for '\(tokenInputName)'")
        }

        var tokenArray = NDArray(descriptor: tokenNDDesc)
        var tokenView = tokenArray.mutableView(as: Int32.self)
        guard var tokenSpan = tokenView.contiguousElements else {
            throw InferenceRuntimeError.invalidState("tokenArray has non-contiguous layout")
        }
        if tokenNDDesc.shape.count == 2 {
            for i in 0..<min(batchSize, tokenIDs.count) {
                tokenSpan[i] = tokenIDs[i]
            }
        } else {
            tokenSpan[0] = tokenIDs[0]
        }

        var inputs: [String: NDArray] = [tokenInputName: tokenArray]
        inputs["embedding_table"] = embeddingTable

        var outputs = try await fn.run(
            inputs: inputs,
            outputViews: InferenceFunction.MutableViews()
        )

        let expectedOutput = "out_transformer_input"
        return outputs.remove(expectedOutput)?.ndArray
            ?? outputs.remove(desc.outputNames.first ?? "")?.ndArray
    }

    // MARK: - Lifecycle

    public func reset() {
        let resetSpan = InstrumentsProfiler.beginReset(engine: "StaticShape")
        processedTokenCount = 0
        resetSpan.end()
    }

    public func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {
        for fnName in extendFunctionNames {
            self.functions[fnName] = try Self.requireFunction(model: model, functionName: fnName)
        }
        reset()
    }
}
