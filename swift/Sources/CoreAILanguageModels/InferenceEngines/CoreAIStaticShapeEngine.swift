// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Synchronization

/// Static-shape inference engine using Core AI models.
public final class StaticShapeEngine: InferenceEngine, @unchecked Sendable {
    public typealias ConfigType = ModelConfig

    public var supportsLogits: Bool { true }

    // MARK: I/O name contracts — models must use these exact names

    private static let logitsOutputName = "out_logits"
    private static let keyCacheName = "key_cache"
    private static let valueCacheName = "value_cache"
    // Engine-produced I/O names (embedding gather needs the model's gather graph, so these
    // aren't `SyncInputHandler`s). `transformerInputToken` is a substring: the declared name
    // may be prefixed (e.g. `out_transformer_input`).
    private static let embeddingTableName = "embedding_table"
    private static let transformerInputToken = "transformer_input"

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

    // Generic persistent state (KV + any hybrid/sliding states), discovered by name from the
    // descriptor and zero-initialized. No hardcoded KV contract, no per-model state properties.
    //
    // Each static graph is compiled with its own per-ctx sequence strides, so a state buffer
    // MUST be physically laid out for the exact bucket the running graph expects — a max-ctx
    // allocation viewed through a slice hands the smaller-bucket graph the wrong stride and
    // corrupts the cache. We therefore RIGHT-SIZE the ctx-scaled states (e.g. the global KV
    // cache) to the running bucket and grow+re-lay-out on a bucket crossing (`ensureStateCtx`).
    // States whose descriptor is constant across buckets (e.g. a fixed sliding-window ring)
    // are allocated once and never re-laid-out.
    private var stateHandler: FixedNDArrayState
    private let stateNames: [String]
    // ctx bucket -> that bucket's (name, descriptor) list, for re-allocation on crossing.
    private let stateDescriptorsByCtx: [Int: [(name: String, descriptor: NDArrayDescriptor)]]
    // States whose sequence extent scales with the ctx bucket (must be re-laid-out on grow).
    private let ctxScaledStateNames: Set<String>
    // The ctx bucket the state buffers are currently laid out for.
    private var currentStateCtx: Int

    // Model-specific input preparation (embedding gather, position ids, causal mask, step, and
    // profile extras like PLE / dual-RoPE / sliding-ring). The engine never learns an input name.
    private let providers: [any SyncInputHandler]
    private let profileName: String

    // Bundle location (for the model profile to find sidecar resources, e.g. the PLE table).
    private let bundleURL: URL

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

    public init(configuration: ModelConfig, preparedModel: PreparedModel, bundleURL: URL) async throws {
        self.config = configuration
        self.model = preparedModel.model
        self.functions = [:]
        self.bundleURL = bundleURL

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
            let ctx =
                Self.parseContextLength(functionName: name)
                ?? Self.contextLength(descriptor: desc, config: configuration)
            if ctx == configuration.maxContextLength {
                largestContextExtend = (name, desc)
                break
            }
        }
        guard let (largestExtendName, largestExtendDescriptor) = largestContextExtend else {
            throw InferenceRuntimeError.invalidState(
                "Failed to find an extend function with the max context length of \(configuration.maxContextLength)")
        }

        // Validate output contract against the max-context function
        try Self.validateIOContract(descriptor: largestExtendDescriptor, functionName: largestExtendName)

        // Load embeddings
        self.embeddingTable = try await Self.loadEmbeddingTable(from: model)

        // Build the per-ctx-bucket state descriptor table. Each extend bucket declares the
        // same state NAMES but with its own physical shape/strides; we allocate against the
        // exact bucket the running graph uses (see the field docs above).
        var descriptorsByCtx: [Int: [(name: String, descriptor: NDArrayDescriptor)]] = [:]
        for name in extendFunctionNames where name.hasPrefix("extend") {
            let desc = try Self.requireDescriptor(model: model, functionName: name)
            let ctx =
                Self.parseContextLength(functionName: name)
                ?? Self.contextLength(descriptor: desc, config: configuration)
            guard descriptorsByCtx[ctx] == nil else { continue }
            var descs: [(name: String, descriptor: NDArrayDescriptor)] = []
            for stateName in desc.stateNames {
                guard case .ndArray(let d) = desc.stateDescriptor(of: stateName) else {
                    throw InferenceRuntimeError.invalidState(
                        "State '\(stateName)' has no NDArray descriptor")
                }
                descs.append((stateName, d))
            }
            descriptorsByCtx[ctx] = descs
        }
        guard !descriptorsByCtx.isEmpty else {
            throw InferenceRuntimeError.invalidState("No extend function to size the state caches from")
        }
        self.stateDescriptorsByCtx = descriptorsByCtx

        // Classify each state as ctx-scaled (its sequence extent changes across buckets → must
        // be re-laid-out on a bucket crossing) or fixed (constant across buckets → allocated
        // once). Derived by comparing the smallest and largest bucket's descriptor for each state.
        let smallestCtx = descriptorsByCtx.keys.min()!
        let largestCtx = descriptorsByCtx.keys.max()!
        var ctxScaled: Set<String> = []
        let smallDescs = descriptorsByCtx[smallestCtx]!
        let largeDescs = Dictionary(
            uniqueKeysWithValues: descriptorsByCtx[largestCtx]!.map { ($0.name, $0.descriptor) })
        for (name, smallDesc) in smallDescs {
            if let largeDesc = largeDescs[name], largeDesc.shape != smallDesc.shape {
                ctxScaled.insert(name)
            }
        }
        self.ctxScaledStateNames = ctxScaled

        // Allocate at the smallest bucket; ctx-scaled states grow on demand via ensureStateCtx.
        self.stateHandler = FixedNDArrayState(states: smallDescs)
        self.stateNames = smallDescs.map(\.name)
        self.currentStateCtx = smallestCtx
        CLILogger.log(
            "States allocated at ctx=\(smallestCtx): \(stateNames) "
                + "(ctx-scaled: \(ctxScaled.sorted()))")

        // Assemble input providers (base + model profile) and verify — at load — that every
        // declared graph input is produced by some provider. Fail closed on any gap.
        let (providers, profileName) = try StaticModelProfile.assembleInputHandlers(
            descriptor: largestExtendDescriptor, bundleURL: bundleURL)
        self.providers = providers
        self.profileName = profileName

        try Self.checkInputCoverage(
            providers: providers, model: model, extendFunctionNames: extendFunctionNames)

        CLILogger.log("Engine initialized (profile: \(profileName), \(providers.count) input providers)")
    }

    /// Ensure the ctx-scaled state buffers are physically laid out for context bucket `ctx`,
    /// growing + re-laying-out the written prefix when a decode step crosses into a larger
    /// bucket. Fixed states (constant descriptor across buckets, e.g. a sliding-window ring)
    /// are carried over verbatim. No-op when already laid out for `ctx`.
    private func ensureStateCtx(_ ctx: Int) throws {
        guard ctx != currentStateCtx, ctxScaledStateNames.isEmpty == false else { return }
        guard let targetDescs = stateDescriptorsByCtx[ctx] else {
            throw InferenceRuntimeError.invalidState(
                "No state descriptors for ctx bucket \(ctx) (ladder: \(stateDescriptorsByCtx.keys.sorted()))")
        }
        let old = stateHandler
        var new = FixedNDArrayState(states: targetDescs)
        let copyPositions = min(processedTokenCount, currentStateCtx)
        for i in stateNames.indices {
            let name = stateNames[i]
            var dst = new[stateIndex: i]
            let src = old[stateIndex: i]
            if ctxScaledStateNames.contains(name) {
                // Grow: copy the written sequence prefix into the larger layout.
                Self.copyStatePrefix(from: src.array, to: &dst.array, copyPositions: copyPositions)
            } else {
                // Fixed: identical shape across buckets — carry the whole buffer over.
                Self.copyStatePrefix(from: src.array, to: &dst.array, copyPositions: nil)
            }
            new[stateIndex: i] = dst
        }
        stateHandler = new
        currentStateCtx = ctx
        CLILogger.log("State caches re-laid-out: ctx \(currentStateCtx) → \(ctx) (copied \(copyPositions) positions)")
    }

    /// Copy a state buffer's written sequence prefix from `src` into the (larger) `dst` layout.
    /// The sequence dim is the last dim; `copyPositions == nil` copies the whole buffer (fixed
    /// states, identical shape). Honors the cache's channel-interleave: physically the layout is
    /// `[groups, ctx, factor]`, so positions `[0, copyLen)` across the `factor` interleaved
    /// channels form one contiguous `copyLen·factor` run per group. src and dst share group order
    /// + interleave, differing only in ctx — so a per-group run copy is correct.
    private static func copyStatePrefix(from src: NDArray, to dst: inout NDArray, copyPositions: Int?) {
        let srcShape = src.shape
        guard srcShape.isEmpty == false else { return }
        let dstShape = dst.shape
        let seqDim = srcShape.count - 1
        let srcSeq = srcShape[seqDim]
        let dstSeq = dstShape[seqDim]
        let factor = src.interleaveLayout?.factor ?? 1
        let copySeq = min(copyPositions ?? srcSeq, min(srcSeq, dstSeq))
        guard copySeq > 0 else { return }
        let groupCount = srcShape.reduce(1, *) / srcSeq / factor
        let srcGroupStride = srcSeq * factor
        let dstGroupStride = dstSeq * factor
        let runElems = copySeq * factor

        func run<T: BitwiseCopyable>(_ type: T.Type) {
            let srcView = src.view(as: T.self)
            var dstView = dst.mutableView(as: T.self)
            dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                srcView.withUnsafePointer { srcPtr, _, _ in
                    for g in 0..<groupCount {
                        dstPtr.advanced(by: g &* dstGroupStride).update(
                            from: srcPtr.advanced(by: g &* srcGroupStride), count: runElems)
                    }
                }
            }
        }
        switch src.scalarType {
        case .float16, .bfloat16: run(Float16.self)
        case .float32: run(Float.self)
        case .int8: run(Int8.self)
        default:
            preconditionFailure(
                "State cache re-layout: unsupported scalar type \(src.scalarType). "
                    + "KV caches are fp16/bf16/fp32/int8; add the type here if a new one is introduced.")
        }
    }

    /// Every input any extend/prompt graph declares must be produced by exactly one handler,
    /// except `transformer_input` / `embedding_table`, which the engine produces directly.
    private static func checkInputCoverage(
        providers: [any SyncInputHandler], model: AIModel, extendFunctionNames: [String]
    ) throws {
        var declared = Set<String>()
        for name in extendFunctionNames {
            if let desc = model.functionDescriptor(for: name) {
                declared.formUnion(desc.inputNames)
            }
        }
        // Engine-produced inputs (embedding gather needs the model's gather graph).
        let engineProduced = declared.filter {
            $0.contains(Self.transformerInputToken) || $0 == Self.embeddingTableName
        }
        var produced = providers.reduce(into: Set<String>()) { $0.formUnion($1.inputNames) }
        produced.formUnion(engineProduced)
        let uncovered = declared.subtracting(produced)
        guard uncovered.isEmpty else {
            throw InferenceRuntimeError.invalidState(
                "No input handler produces required input(s): \(uncovered.sorted()). "
                    + "Add them to the model profile (StaticModelProfile.resolve).")
        }
    }

    public convenience init(configuration: ModelConfig, modelURL: URL) async throws {
        let preparedModel = try await PreparedModel.prepare(at: modelURL)
        try await self.init(configuration: configuration, preparedModel: preparedModel, bundleURL: modelURL)
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
        // States are discovered + allocated generically (FixedNDArrayState) and their inputs
        // covered by the provider registry — no hardcoded KV-only state contract here.
    }

    private static func loadEmbeddingTable(from model: AIModel) async throws -> NDArray {
        CLILogger.log("Loading embeddings...")
        guard let embeddingFunction = try model.loadFunction(named: "load_embeddings") else {
            throw InferenceRuntimeError.invalidState("Cannot load 'load_embeddings'")
        }

        guard
            case .ndArray(let embeddingDesc) = embeddingFunction.descriptor.outputDescriptor(
                of: Self.embeddingTableName)
        else {
            throw InferenceRuntimeError.invalidState(
                "load_embeddings has no 'embedding_table' ndArray output descriptor")
        }
        var embeddingArray = NDArray(descriptor: embeddingDesc)

        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&embeddingArray, for: Self.embeddingTableName)

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
        if let txName = desc.inputNames.first(where: { $0.contains(Self.transformerInputToken) }),
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
        if let ctx = Self.parseContextLength(functionName: functionName) { return ctx }
        let desc = try functionDescriptor(for: functionName)
        return Self.contextLength(descriptor: desc, config: config)
    }

    private static func contextLength(
        model: AIModel, functionName: String, config: ModelConfig
    ) throws -> Int {
        if let ctx = parseContextLength(functionName: functionName) { return ctx }
        guard let desc = model.functionDescriptor(for: functionName) else {
            return config.maxContextLength
        }
        return contextLength(descriptor: desc, config: config)
    }

    /// Parse the ctx bucket from a `<prefix>_<ctx>_<q>` function name (e.g. `extend_256_16`
    /// → 256, `prompt_opt_1024_64` → 1024): the ctx is the second-to-last `_`-component.
    /// Authoritative over the descriptor heuristic, whose `shape.max()` can exceed the ctx
    /// when a non-seq dim is larger than a small bucket (e.g. qwen3 `[.,1024,.,256]` @ ctx 256).
    static func parseContextLength(functionName: String) -> Int? {
        let parts = functionName.split(separator: "_")
        guard parts.count >= 2, let ctx = Int(parts[parts.count - 2]) else { return nil }
        return ctx
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
        func pairs(prefix: String) -> [(ctx: Int, q: Int, name: String)] {
            extendFunctionNames.filter { $0.hasPrefix(prefix) }.compactMap { name in
                let parts = Array(name.split(separator: "_").suffix(2))
                guard parts.count == 2, let c = Int(parts[0]), let q = Int(parts[1]) else { return nil }
                return (c, q, name)
            }
        }
        var candidatesAll = pairs(prefix: isPrefill ? "prompt" : "extend")
        if candidatesAll.isEmpty { candidatesAll = pairs(prefix: isPrefill ? "extend" : "prompt") }
        guard !candidatesAll.isEmpty else {
            throw InferenceRuntimeError.invalidState("No extend/prompt functions in static-shape model")
        }

        // Smallest q >= numInputTokens (clamp to the largest available q).
        let qs = Set(candidatesAll.map(\.q)).sorted()
        let selectedQ = qs.first(where: { $0 >= numInputTokens }) ?? qs.last!
        // Among that q, the smallest ctx strictly greater than currentPosition (clamp to largest).
        let candidates = candidatesAll.filter { $0.q == selectedQ }.sorted { $0.ctx < $1.ctx }
        guard let selected = candidates.first(where: { $0.ctx > currentPosition }) ?? candidates.last else {
            throw InferenceRuntimeError.invalidState(
                "No graph with ctx > \(currentPosition) and q = \(selectedQ)")
        }
        return selected.name
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
            if commonPrefix < input.count && commonPrefix < history.count {
                // Divergence — full reset (static engine has fixed-size KV)
                processedTokenCount = 0
                history.clear()
            } else if processedTokenCount >= input.count {
                // Extension — rewind for seeding
                let resetTo = Swift.max(0, commonPrefix - 1)
                processedTokenCount = resetTo
                history.truncate(to: resetTo)
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
        inputTokens: [Int32], samplingConfig: SamplingConfiguration, returnsLogits: Bool
    ) async throws -> (logits: [LogitsScalarType]?, token: Int32) {
        CLILogger.log("Inference: \(inputTokens.count) tokens, processed: \(processedTokenCount)")

        let totalTokenCount = inputTokens.count
        guard processedTokenCount < totalTokenCount else {
            throw InferenceRuntimeError.invalidState("No new tokens to process")
        }

        var logitBuffer = [LogitsScalarType](repeating: 0, count: config.vocabSize)
        var currentPosition = processedTokenCount

        if processedTokenCount == 0 {
            // Fresh sequence (no prefix reuse): zero the state caches so no KV from a prior
            // generation leaks into this one.
            stateHandler.reset()
        }

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

            let fn = try loadFunction(named: graphName)
            let desc = try functionDescriptor(for: graphName)
            let graphInputs = Set(desc.inputNames)

            let prepareSpan = InstrumentsProfiler.beginPrepareStep(
                operation: "providers", engine: "StaticShape")
            // Build the shared per-step context: the running graph's input descriptors let
            // each handler size its own buffer to this bucket.
            var descriptors: [String: NDArrayDescriptor] = [:]
            for name in desc.inputNames {
                if case .ndArray(let d) = desc.inputDescriptor(of: name) { descriptors[name] = d }
            }
            let ctx = InputContext(
                tokens: inputTokens[batchStartToken...batchEndToken],
                processedTokenCount: currentPosition,
                alignedStep: batchStartToken,
                batchSize: batchSize,
                slidingWindow: nil,
                descriptors: descriptors)

            var inputs: [String: NDArray] = [:]
            // Embedding gather is engine-produced (needs the model's gather graph).
            if let txName = desc.inputNames.first(where: { $0.contains(Self.transformerInputToken) }) {
                if graphInputs.contains(Self.embeddingTableName) { inputs[Self.embeddingTableName] = embeddingTable }
                guard let gathered = try await runGather(tokenIDs: Array(ctx.tokens), batchSize: batchSize) else {
                    throw InferenceRuntimeError.invalidState(
                        "Embedding gather returned no output for batch \(batchSize)")
                }
                inputs[txName] = gathered
            }
            // Remaining inputs via the shared SyncInputHandler registry.
            for var provider in providers where provider.inputNames.contains(where: graphInputs.contains) {
                let produced = try await provider.prepare(ctx)
                for (name, value) in produced where graphInputs.contains(name) {
                    inputs[name] = value
                }
            }
            prepareSpan.end()

            let logitsSpan = InstrumentsProfiler.beginLogitsInference(
                step: batchStartToken, tokens: tokensInBatch, engine: "StaticShape")

            // Right-size the state caches to this graph's ctx bucket, then bind full buffers
            // by name. Because the buffers are laid out for the exact running bucket, no
            // slicing is needed — the graph's compiled per-ctx strides match the allocation.
            try ensureStateCtx((try? contextLength(of: graphName)) ?? currentStateCtx)
            // Local ref binding so the borrow spans the run() call (bind ties `states`'
            // lifetime to the handler; a stored-property borrow would end at the call).
            let handler = stateHandler
            var states = InferenceFunction.MutableViews()
            handler.bind(into: &states)
            var outputs = try await fn.run(
                inputs: inputs,
                states: _unsafeEscapeMutableViews(consume states),
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
        inputs[Self.embeddingTableName] = embeddingTable

        var outputs = try await fn.run(
            inputs: inputs,
            outputViews: InferenceFunction.MutableViews()
        )

        let expectedOutput = "out_transformer_input"
        return outputs.remove(expectedOutput)?.ndArray
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
        let resetSpan = InstrumentsProfiler.beginReset(engine: "StaticShape")
        if tokenIndex == 0 {
            processedTokenCount = 0
            history.clear()
            stateHandler.reset()
        } else {
            processedTokenCount = tokenIndex
            history.truncate(to: tokenIndex)
        }
        resetSpan.end()
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
                    returnsLogits: returnsLogits || forcedContinuation != nil
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
