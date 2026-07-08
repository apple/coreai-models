// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import FoundationModels
import Synchronization
import Tokenizers

/// FoundationModels Adoption for Core AI inference engines.
///
/// Wraps any `InferenceEngine` (pipelined, sequential, or static-shape) and exposes it
/// through the FoundationModels `LanguageModel` protocol. It uses the modern `tokenSequence()`
/// API for efficient streaming token generation.
/// ## Engine Selection
/// The engine type is determined by `EngineFactory` based on model structure:
/// - **Pipelined**: GPU-accelerated with pipeline-depth-matched buffering (fastest for GPU models)
/// - **Sequential**: CPU-based synchronous execution (fallback)
/// - **Static-shape**: Neural Engine optimized for chunked static models
///
/// ## Usage
/// ```swift
/// let model = try await CoreAILanguageModel(resourcesAt: url)  // .lazy by default
/// print(model.estimatedSizeOnDiskBytes ?? 0)
/// try await model.load()                                       // optional; respond auto-loads
/// let session = LanguageModelSession(model: model)
/// // ... generate ...
/// model.unload()
/// ```
public struct CoreAILanguageModel: LanguageModel {
    public enum LoadMode: Sendable {
        case lazy
        case eager
    }

    // MARK: - Properties

    private let url: URL
    private let variant: String?
    private let kvCacheStrategy: KVCacheStrategy
    fileprivate let samplingConfig: SamplingConfiguration
    fileprivate let bundle: LanguageBundle
    fileprivate let tokenizer: any Tokenizer
    fileprivate let thinkingMarkers: (open: String, close: String)
    fileprivate let toolCallMarkers: (open: String, close: String)?
    private let supportsToolCalling: Bool
    fileprivate let supportsReasoning: Bool
    fileprivate let resources: ModelResources
    /// All EOS-like token IDs beyond the tokenizer's main `eosTokenId` — e.g.
    /// Gemma's `<end_of_turn>`, read from tokenizer_config.json at init.
    fileprivate let additionalEosTokenIds: [Int32]

    // MARK: - Protocol Requirements

    public typealias Executor = CoreAIExecutor

    public var capabilities: LanguageModelCapabilities {
        var caps: [LanguageModelCapabilities.Capability] = []
        if supportsToolCalling { caps.append(.toolCalling) }
        if supportsReasoning { caps.append(.reasoning) }
        if isGuidedGenerationSupported { caps.append(.guidedGeneration) }
        return LanguageModelCapabilities(capabilities: caps)
    }

    public var executorConfiguration: CoreAIExecutor.Configuration {
        CoreAIExecutor.Configuration(
            url: url,
            variant: variant,
            kvCacheStrategy: kvCacheStrategy,
            modelIdentifier: bundle.name,
            samplingConfig: samplingConfig,
            vocabSize: bundle.vocabSize
        )
    }

    // MARK: - Initialization

    /// Creates a model from a resource bundle on disk.
    ///
    /// ```swift
    /// let model = try await CoreAILanguageModel(resourcesAt: url)             // lazy
    /// let model = try await CoreAILanguageModel(resourcesAt: url, mode: .eager)
    /// ```
    ///
    /// - Parameter url: URL to the model bundle directory.
    /// - Parameter mode: When to load the engine. Defaults to `.lazy`. With
    ///   `.eager`, the tokenizer and engine load concurrently
    /// - Parameter variant: Engine variant override (e.g. "coreai-sequential",
    ///   "ane"). Nil for auto-detect from model structure.
    /// - Parameter kvCacheStrategy: KV cache memory strategy. Defaults to
    ///   `.auto` (256-token initial size for dynamic models). Pass
    ///   `.fixedSize` to pre-allocate at full `maxContextLength`.
    /// - Throws: If the asset bundle is invalid or the tokenizer fails to load.
    ///   With `.eager`, also throws on engine-creation failure.
    public init(
        resourcesAt url: URL,
        mode: LoadMode = .lazy,
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto
    ) async throws {
        let bundle = try LanguageBundle(at: url)
        let configuration = CoreAIExecutor.Configuration(
            url: url,
            variant: variant,
            kvCacheStrategy: kvCacheStrategy,
            modelIdentifier: bundle.name,
            samplingConfig: .greedy,
            vocabSize: bundle.vocabSize
        )
        let resources = ModelResources.shared(for: configuration)

        async let engineLoad: Void = {
            if mode == .eager { try await resources.loadResources() }
        }()

        let tokenizerLoadSpan = InstrumentsProfiler.beginTokenizerLoad(id: bundle.tokenizer)
        let tokenizer = try await bundle.loadTokenizer()
        tokenizerLoadSpan.end()

        try await engineLoad
        self.init(
            configuration: configuration, bundle: bundle, tokenizer: tokenizer,
            resources: resources)
    }

    private init(
        configuration: CoreAIExecutor.Configuration,
        bundle: LanguageBundle,
        tokenizer: any Tokenizer,
        resources: ModelResources
    ) {
        let toolCallMarkers = CoreAIExecutor.detectToolCallMarkers(using: tokenizer)
        self.url = configuration.url
        self.variant = configuration.variant
        self.kvCacheStrategy = configuration.kvCacheStrategy
        self.samplingConfig = configuration.samplingConfig
        self.bundle = bundle
        self.tokenizer = tokenizer
        self.thinkingMarkers = CoreAIExecutor.detectThinkingMarkers(using: tokenizer)
        self.toolCallMarkers = toolCallMarkers
        self.supportsToolCalling = toolCallMarkers != nil
        self.supportsReasoning =
            tokenizer.convertTokenToId("<think>") != nil
            || tokenizer.convertTokenToId("<|reasoning_start|>") != nil
        self.resources = resources
        // Read additional stop token IDs from tokenizer_config.json (e.g. Gemma's
        // <end_of_turn>). Empty when the bundle has no tokenizer directory.
        if let tokenizerDir = bundle.tokenizerPath {
            self.additionalEosTokenIds = LanguageConfig.additionalStopTokenIds(
                from: tokenizerDir, tokenizer: tokenizer)
        } else {
            self.additionalEosTokenIds = []
        }
    }

    // MARK: - Resource control

    /// Estimated on-disk size of the model's main asset, in bytes.

    public var estimatedSizeOnDiskBytes: Int? {
        guard let assetURL = bundle.modelURL(for: ModelBundle.ComponentKey.main) else { return nil }
        return assetURL.recursiveFileSizeInBytes()
    }

    public func load() async throws {
        try await resources.loadResources()
    }

    public func unload() {
        resources.unloadResources()
    }

    /// Whether guided generation is available for this model.
    private var isGuidedGenerationSupported: Bool {
        if let supportsLogits = resources.loadedEngineSupportsLogits {
            return supportsLogits
        }
        return variant != "coreai-pipelined"
    }

    // MARK: - Executor

    public struct CoreAIExecutor: LanguageModelExecutor {
        public typealias Model = CoreAILanguageModel

        public struct Configuration: Hashable, Sendable {
            let url: URL
            let variant: String?
            let kvCacheStrategy: KVCacheStrategy
            let modelIdentifier: String
            let samplingConfig: SamplingConfiguration
            let vocabSize: Int?
        }

        // MARK: - Properties

        private let resources: ModelResources

        // MARK: - Initialization

        public init(configuration: Configuration) throws {
            self.resources = ModelResources.shared(for: configuration)
        }

        /// Probes the tokenizer for known reasoning marker pairs. Each
        /// candidate pair is verified to exist as added/special tokens via
        /// `convertTokenToId(_:)` — only models that actually have these
        /// tokens in their vocab match. First match wins; falls back to
        /// `<think>`/`</think>` so the parser is harmless on models that
        /// don't emit reasoning markup at all.
        ///
        /// Add a new pair here when onboarding a model with different
        /// markers. For models with non-pair-symmetric formats (e.g.
        /// gpt-oss / Harmony), a different parser is needed; this one
        /// covers the `<open>...</close>` shape.
        fileprivate static func detectThinkingMarkers(
            using tokenizer: any Tokenizer
        ) -> (open: String, close: String) {
            let candidates: [(open: String, close: String)] = [
                ("<think>", "</think>"),
                ("<|reasoning_start|>", "<|reasoning_end|>"),
            ]
            for pair in candidates {
                if tokenizer.convertTokenToId(pair.open) != nil,
                    tokenizer.convertTokenToId(pair.close) != nil
                {
                    return pair
                }
            }
            return ("<think>", "</think>")
        }

        /// Probes the tokenizer for known tool call marker pairs. Each
        /// candidate tag-pair is verified to exist as special tokens via
        /// `convertTokenToId(_:)`. Returns nil when the model's tokenizer
        /// has no tool call tokens at all.
        ///
        /// Mistral uses `[TOOL_CALLS]` as a single special token with no
        /// paired close token; `"\n"` is used as a synthetic close because
        /// the JSON array is always emitted on a single line. The open marker
        /// matches the bare token without a trailing space — `parseToolCalls`
        /// already trims leading whitespace so optional spacing is handled.
        fileprivate static func detectToolCallMarkers(
            using tokenizer: any Tokenizer
        ) -> (open: String, close: String)? {
            // Standard tag-pair formats — both markers must be special tokens.
            let tagPairs: [(open: String, close: String)] = [
                ("<tool_call>", "</tool_call>"),
                ("<function_calls>", "</function_calls>"),
            ]
            for pair in tagPairs
            where tokenizer.convertTokenToId(pair.open) != nil
                && tokenizer.convertTokenToId(pair.close) != nil
            {
                return pair
            }
            // Mistral: [TOOL_CALLS] is a special token but has no paired close token.
            // Use "\n" as a synthetic close — the JSON array is always on a single line.
            if tokenizer.convertTokenToId("[TOOL_CALLS]") != nil {
                return (open: "[TOOL_CALLS]", close: "\n")
            }
            return nil
        }

        // MARK: - Prewarm (FoundationModels, synchronous)

        /// Kicks off the engine load in the background.
        public func prewarm(model: CoreAILanguageModel, transcript: Transcript) {
            Task { try? await resources.loadResources() }
        }

        // MARK: - respond(to:model:streamingInto:) — new channel-based API

        public nonisolated(nonsending) func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: CoreAILanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            // Tokenization span
            let tokenizationSpan = InstrumentsProfiler.beginTokenization(inputLength: 0)
            let promptTokens = Self.makeTokens(
                from: Array(request.transcript),
                using: model.tokenizer,
                tools: request.enabledToolDefinitions,
                component: "CoreAIExecutor"
            )
            guard !promptTokens.isEmpty else {
                tokenizationSpan.end()
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: Array(request.transcript),
                        debugDescription: "CoreAI could not tokenize the conversation transcript."
                    )
                )
            }
            tokenizationSpan.end()

            CLILogger.log("Tokenized \(promptTokens.count) tokens", component: "CoreAIExecutor")

            let effectiveSamplingConfig = makeSamplingConfig(
                from: request.generationOptions, base: model.samplingConfig)
            let defaultMaxTokens = model.supportsReasoning ? 2048 : 512
            let maxTokens = request.generationOptions.maximumResponseTokens ?? defaultMaxTokens

            // Borrow the engine for the whole generation.
            try await resources.withEngine { engine in
                // FoundationModels now threads entry identity itself based on event
                // ordering — we no longer mint an entryID and pass it down.

                // Check if guided generation is requested
                if let schema = request.schema {
                    guard engine.supportsLogits else {
                        throw LanguageModelError.unsupportedCapability(
                            .init(
                                capability: .guidedGeneration,
                                debugDescription:
                                    "This model's inference engine does not support guided generation "
                                    + "(constrained decoding requires per-step logits)."
                            )
                        )
                    }
                    try await respondConstrained(
                        engine: engine,
                        model: model,
                        schema: schema,
                        promptTokens: promptTokens,
                        samplingConfig: effectiveSamplingConfig,
                        maxTokens: maxTokens,
                        channel: channel
                    )
                } else {
                    try await respondVanilla(
                        engine: engine,
                        model: model,
                        promptTokens: promptTokens,
                        samplingConfig: effectiveSamplingConfig,
                        maxTokens: maxTokens,
                        channel: channel
                    )
                }
            }
        }

        // MARK: - Vanilla Generation (no schema)

        private func respondVanilla(
            engine: any InferenceEngine,
            model: CoreAILanguageModel,
            promptTokens: [Int],
            samplingConfig: SamplingConfiguration,
            maxTokens: Int,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let tokenizer = model.tokenizer
            let tokenStream = try await engine.generate(
                with: promptTokens.map(Int32.init),
                samplingConfiguration: samplingConfig,
                inferenceOptions: InferenceOptions(maxTokens: maxTokens)
            )

            // All EOS-like tokens: the tokenizer's main EOS plus any additional
            // stop tokens from tokenizer_config.json (e.g. Gemma's <end_of_turn>).
            var eosTokens = Set<Int32>()
            if let id = tokenizer.eosTokenId { eosTokens.insert(Int32(id)) }
            eosTokens.formUnion(model.additionalEosTokenIds)
            // Incremental-decode buffer. After a clean emit, one token is
            // retained as context for the next step (see below). During a
            // multi-byte sequence that hasn't decoded cleanly yet, multiple
            // tokens accumulate until the sequence is complete. In the steady
            // state the buffer holds at most 2 tokens, so tokenizer.decode
            // is O(1) per step.
            var pendingTokens: [Int32] = []
            var previousDecodedText: String = ""
            var tokenStep: Int = 0
            // Segments the decoded stream into `.text` and `.reasoning`
            // events on the fly. Reasoning content (model's chain-of-thought
            // emitted inside the configured open/close markers) is routed
            // to a top-level `.reasoning(...)` channel event so it lands as
            // its own `Transcript.Reasoning` entry, not mixed into the
            // user-facing `Transcript.Response`. Markers were resolved at
            // model init from the tokenizer's known token ids.
            var thinkParser = ThinkTagParser(
                open: model.thinkingMarkers.open,
                close: model.thinkingMarkers.close
            )
            // Routes tool call markup to .toolCalls(...) channel events.
            // nil when the model's tokenizer has no tool call tokens.
            var toolCallParser: ToolCallParser? = model.toolCallMarkers.map {
                ToolCallParser(openMarker: $0.open, closeMarker: $0.close)
            }
            var generatedTokenCount: Int = 0
            var reasoningTokenCount: Int = 0

            for try await output in tokenStream {
                let token = output.tokenId
                if eosTokens.contains(token) {
                    tokenStream.setStopReason(.eos)
                    break
                }

                pendingTokens.append(token)
                tokenStep += 1
                generatedTokenCount += 1

                let decodeSpan = InstrumentsProfiler.beginDecode(step: tokenStep)
                let decodedText = tokenizer.decode(tokens: pendingTokens.map { Int($0) })
                decodeSpan.end()

                let common = decodedText.commonPrefix(with: previousDecodedText)
                let delta = String(decodedText.dropFirst(common.count))
                // Check for replacement char on the full `decodedText`, not on
                // `delta`. Some tokenizers emit one U+FFFD per attempted decode
                // of an incomplete multi-byte sequence (rather than one per
                // bad byte), so two consecutive partial tokens can produce
                // identical "\u{FFFD}" strings — making `delta` empty and
                // hiding the still-incomplete state. Checking `decodedText`
                // catches that case.
                let hasReplacementChar = decodedText.unicodeScalars.contains { $0 == "\u{FFFD}" }

                if hasReplacementChar {
                    // UTF-8 bytes don't form a clean character yet. Hold the
                    // token and wait for the next iteration to extend the
                    // buffer; don't drop or advance.
                    await channel.send(
                        .response(action: .appendText("", tokenCount: 1))
                    )
                    previousDecodedText = decodedText
                    continue
                }

                for event in thinkParser.consume(delta) {
                    if case .reasoning = event { reasoningTokenCount += 1 }
                    await dispatch(event: event, toolCallParser: &toolCallParser, channel: channel)
                }

                // Retain the last token as O(1) context for the next decode.
                // SentencePiece needs at least one prior token to infer the leading
                // ▁ (space) on the following token; clearing to empty decodes each
                // new token in isolation and drops inter-word spaces.
                // Keeping one token bounds re-decode cost to 2 tokens per step.
                // Safe for all supported tokenizers: decode([last]) is a prefix of
                // decode([last, next]) when addPrefixSpace=true (Mistral, Llama, Qwen)
                // and for ByteLevel tokenizers (GPT-2 style) where spaces are direct bytes.
                if let last = pendingTokens.last {
                    pendingTokens = [last]
                    previousDecodedText = tokenizer.decode(tokens: [Int(last)])
                } else {
                    pendingTokens.removeAll(keepingCapacity: true)
                    previousDecodedText = ""
                }
            }

            // Flush parsers — drains any content held back waiting for a marker.
            // Without this, content right at the EOS boundary (or inside an
            // unclosed block) would be lost.
            for event in thinkParser.flush() {
                await dispatch(event: event, toolCallParser: &toolCallParser, channel: channel)
            }
            if var tcp = toolCallParser {
                for event in tcp.flush() {
                    await dispatchToolCall(for: event, channel: channel)
                }
                toolCallParser = tcp
            }

            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(totalTokenCount: promptTokens.count, cachedTokenCount: 0),
                        output: .init(
                            totalTokenCount: generatedTokenCount,
                            reasoningTokenCount: reasoningTokenCount
                        )
                    )))

            // Yield to let the engine's tokenSequence Task finish cleanup
            // (putBackEngine, state reset, etc.) before the next respond().
            await Task.yield()
        }

        // MARK: - Event Dispatch

        /// Routes a parser event to the matching FoundationModels channel event.
        /// Text is forwarded to the tool call parser (if present) or emitted as
        /// `.response(...).appendText`. Reasoning becomes a top-level
        /// `.reasoning(...).appendText`. Reasoning is a sibling of
        /// response/tool-calls in the new API (not nested under response)
        /// because at parse time we don't yet know whether the model will
        /// follow the thought block with a response or a tool call.
        ///
        /// We deliberately do not pass `entryID` — FoundationModels threads
        /// entry identity itself based on event ordering.
        private func dispatch(
            event: ThinkTagParser.Event,
            toolCallParser: inout ToolCallParser?,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            switch event {
            case .reasoning(let text):
                await channel.send(
                    .reasoning(action: .appendText(text, tokenCount: 1))
                )
            case .text(let text):
                if var tcp = toolCallParser {
                    for toolEvent in tcp.consume(text) {
                        await dispatchToolCall(for: toolEvent, channel: channel)
                    }
                    toolCallParser = tcp
                } else if !text.isEmpty {
                    await channel.send(
                        .response(action: .appendText(text, tokenCount: 1))
                    )
                }
            }
        }

        private func dispatchToolCall(
            for event: ToolCallParser.Event,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            switch event {
            case .text(let text):
                if !text.isEmpty {
                    await channel.send(
                        .response(action: .appendText(text, tokenCount: 1))
                    )
                }
            case .toolCall(let id, let name, let argsJSON):
                CLILogger.log(
                    "ToolCallParser: dispatching tool call id=\(id) name=\(name) args=\(argsJSON)",
                    component: "CoreAIExecutor")
                await channel.send(
                    .toolCalls(
                        action: .toolCall(
                            id: id,
                            name: name,
                            action: .appendArguments(argsJSON, tokenCount: 1)
                        )
                    )
                )
            }
        }

        // MARK: - Constrained Generation (with schema)

        private func respondConstrained(
            engine: any InferenceEngine,
            model: CoreAILanguageModel,
            schema: GenerationSchema,
            promptTokens: [Int],
            samplingConfig: SamplingConfiguration,
            maxTokens: Int,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let schemaData = try JSONEncoder().encode(schema)

            guard let jsonSchema = String(data: schemaData, encoding: .utf8) else {
                preconditionFailure("GenerationSchema JSON encoding produced invalid UTF-8")
            }

            let strategy = ConstrainedDecodingStrategy(
                jsonSchema: jsonSchema, vocabSize: model.bundle.vocabSize)
            let stopSequences = StopSequences(
                for: model.tokenizer,
                additionalEosTokenIds: model.additionalEosTokenIds
            )

            let stream = try await strategy.decode(
                from: .tokens(promptTokens),
                tokenizer: model.tokenizer,
                inferenceEngine: engine,
                samplingConfiguration: samplingConfig,
                options: InferenceOptions(maxTokens: maxTokens),
                stopSequences: stopSequences
            )

            // Bridge AsyncThrowingStream -> LanguageModelExecutorGenerationChannel
            var generatedTokenCount = 0
            for try await result in stream {
                generatedTokenCount += 1
                await channel.send(
                    .response(action: .appendText(result.text, tokenCount: 1))
                )
            }

            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(totalTokenCount: promptTokens.count, cachedTokenCount: 0),
                        output: .init(
                            totalTokenCount: generatedTokenCount,
                            reasoningTokenCount: 0
                        )
                    )))

            // Yield to let the engine's tokenSequence Task finish cleanup
            // (putBackEngine, state reset, etc.) before the next respond().
            await Task.yield()
        }

        // MARK: - Transcript → Tokens

        /// Extracts plain text from a segment collection, joining with `separator`.
        private static func textContent(
            of segments: some Collection<Transcript.Segment>,
            separator: String = ""
        ) -> String {
            segments.compactMap {
                if case .text(let t) = $0 { return t.content }
                return nil
            }.joined(separator: separator)
        }

        /// Tool call entry for the assistant message's `tool_calls` array.
        private struct ToolCallEntry: Sendable {
            let id: String
            let name: String
            let arguments: String

            var message: [String: any Sendable] {
                [
                    "id": id,
                    "type": "function",
                    "function": ["name": name, "arguments": arguments] as [String: any Sendable],
                ]
            }
        }

        /// Converts transcript entries to tokens using the provided tokenizer.
        ///
        /// Handles all entry types including prior tool calls and tool outputs.
        /// Tool definitions are forwarded to `applyChatTemplate` so the model
        /// sees the available functions in the system prompt.
        static func makeTokens(
            from entries: [Transcript.Entry],
            using tokenizer: any Tokenizer,
            tools: [Transcript.ToolDefinition] = [],
            component: String = "CoreAIExecutor"
        ) -> [Int] {
            var messages: [Message] = []

            for entry in entries {
                switch entry {
                case .instructions(let instructions):
                    let text = textContent(of: instructions.segments, separator: "\n")
                    if !text.isEmpty { messages.append(["role": "system", "content": text]) }

                case .prompt(let prompt):
                    let text = textContent(of: prompt.segments)
                    if !text.isEmpty { messages.append(["role": "user", "content": text]) }

                case .response(let response):
                    let text = textContent(of: response.segments)
                    if !text.isEmpty { messages.append(["role": "assistant", "content": text]) }

                case .toolCalls(let toolCalls):
                    let calls = toolCalls.map {
                        ToolCallEntry(id: $0.id, name: $0.toolName, arguments: $0.arguments.jsonString)
                    }
                    messages.append([
                        "role": "assistant",
                        "content": "" as any Sendable,
                        "tool_calls": calls.map(\.message) as any Sendable,
                    ])

                case .toolOutput(let output):
                    // Tool result turn.
                    let content = textContent(of: output.segments)
                    messages.append([
                        "role": "tool",
                        "tool_call_id": output.id,
                        "name": output.toolName,
                        "content": content,
                    ])

                case .reasoning:
                    // Don't echo the model's prior reasoning back into the prompt.
                    continue

                @unknown default:
                    continue
                }
            }

            if messages.isEmpty { return [] }

            let toolSpecs: [ToolSpec]? = tools.isEmpty ? nil : tools.compactMap { makeToolSpec(from: $0) }

            do {
                CLILogger.log("Applying chat template via tokenizer", component: component)
                return try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
            } catch {
                CLILogger.log(
                    "Failed to apply chat template: \(error), falling back to simple encoding",
                    component: component)
                let text = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
                return tokenizer.encode(text: text)
            }
        }

        /// Converts a `ToolDefinition` into the `ToolSpec` format expected by
        /// `applyChatTemplate`. Parameters are decoded into a typed `JSONValue`
        /// tree — avoids passing raw `Any` through the codebase.
        private static func makeToolSpec(from definition: Transcript.ToolDefinition) -> ToolSpec? {
            guard
                let schemaData = try? JSONEncoder().encode(definition.parameters),
                let rawObj = try? JSONSerialization.jsonObject(with: schemaData),
                let dict = rawObj as? [String: Any]
            else {
                CLILogger.log(
                    "Failed to encode parameters for tool '\(definition.name)'",
                    component: "CoreAIExecutor")
                return nil
            }
            let function: [String: any Sendable] = [
                "name": definition.name,
                "description": definition.description,
                "parameters": dict.mapValues { JSONValue($0).sendable },
            ]
            return ["type": "function", "function": function]
        }

        /// Typed, `Sendable` representation of an arbitrary JSON value.
        ///
        /// Bridges the `NSObject`-bridged output of `JSONSerialization` into an
        /// explicit Swift enum so nothing untyped escapes into the rest of the code.
        private indirect enum JSONValue: Sendable {
            case string(String)
            case int(Int)
            case double(Double)
            case bool(Bool)
            case array([JSONValue])
            case object([String: JSONValue])
            case null

            init(_ value: Any) {
                switch value {
                case let s as String: self = .string(s)
                case let n as NSNumber where CFGetTypeID(n) == CFBooleanGetTypeID():
                    self = .bool(n.boolValue)
                case let n as NSNumber:
                    let d = n.doubleValue
                    self = (d == d.rounded() && !d.isInfinite) ? .int(n.intValue) : .double(d)
                case let a as [Any]: self = .array(a.map { JSONValue($0) })
                case let o as [String: Any]: self = .object(o.mapValues { JSONValue($0) })
                default: self = .null
                }
            }

            var sendable: any Sendable {
                switch self {
                case .string(let v): return v
                case .int(let v): return v
                case .double(let v): return v
                case .bool(let v): return v
                case .null: return NSNull()
                case .array(let v): return v.map(\.sendable)
                case .object(let v): return v.mapValues(\.sendable)
                }
            }
        }

        // MARK: - Helper Methods

        private func makeSamplingConfig(
            from options: GenerationOptions,
            base: SamplingConfiguration
        ) -> SamplingConfiguration {
            if let temperature = options.temperature {
                return SamplingConfiguration(temperature: temperature)
            }
            return base
        }
    }
}
