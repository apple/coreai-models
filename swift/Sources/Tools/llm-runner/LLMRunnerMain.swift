// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAI
import CoreAILanguageModels
import CoreAIShared
import Darwin
import Foundation
import Tokenizers

/// Warmup mode for inference engines.
///
/// - `default`: engine-specific default behavior
///   - Core AI pipelined: warm decode shape [1] + prefill shape [256] (<1s)
///   - Core AI static-shape: bank warmup all bucket shapes (~2.6s)
/// - `off` / `none`: skip warmup entirely
/// - `exact`: warm a specific shape (requires `--warmup-length N`)
enum WarmupMode: ExpressibleByArgument, Equatable {
    case defaultMode
    case off
    case exact

    init?(argument: String) {
        switch argument.lowercased() {
        case "default": self = .defaultMode
        case "off", "none": self = .off
        case "exact": self = .exact
        default: return nil
        }
    }

    var defaultValueDescription: String {
        switch self {
        case .defaultMode: return "default"
        case .off: return "off"
        case .exact: return "exact"
        }
    }
}

@main
struct Main {
    static func main() async throws {
        await LLMRunner.main()
    }
}

// MARK: - Main Runner Command (Refactored)
struct LLMRunner: AsyncParsableCommand, Sendable {
    static let configuration = CommandConfiguration(
        commandName: "llm-runner",
        abstract: "Run LLM models using CoreAI inference engines"
    )

    @Option(
        name: .customLong("model"),
        help: "Path to a model bundle directory"
    )
    var model: String?

    @Option(help: "Input text prompt for generation (default: 'Hello, how are you?')")
    var prompt: String?

    @Option(name: .customLong("prompt-file"), help: "Read prompt text from file (UTF-8 text file)")
    var promptFile: String?

    @Option(name: .customLong("raw-tokens"), help: "JSON file with pre-tokenized tokens: {\"tokens\": [...]}")
    var rawTokens: String?

    @Option(help: "Maximum number of tokens to generate (default: 50)")
    var maxTokens: Int = 50

    @Option(help: "Temperature for sampling (0.0 = greedy, higher = more random)")
    var temperature: Double = 0.7

    @Option(name: .customLong("top-k"), help: "Top-K sampling: only consider K most likely tokens (e.g., 50)")
    var topK: Int?

    @Option(
        name: .customLong("top-p"),
        help: "Top-P (nucleus) sampling: consider tokens in top P probability mass (e.g., 0.9)")
    var topP: Double?

    @Option(
        name: .customLong("min-p"),
        help: "Min-P sampling: keep tokens with probability >= minP × max probability (e.g., 0.05)")
    var minP: Double?

    @Option(help: "Sampling strategy. Options: 'temperature' (default), 'greedy'")
    var samplingStrategy: String = "temperature"

    @Flag(help: "Use synchronous sampling (disabled by default). Useful for isolating inference bottlenecks")
    var synchronousSampling: Bool = false

    @Option(
        name: .customLong("json-schema"),
        help:
            "Constrain output to match a JSON schema. Accepts a JSON schema string or a file path. Example: '{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},\"required\":[\"name\"]}'"
    )
    var jsonSchema: String?

    @Option(help: "Model inference engine variant. Defaults to 'default'")
    var inferenceEngineVariant: String = "default"

    @Option(
        name: .customLong("kv-cache-strategy"),
        help:
            "KV cache memory strategy: 'auto' (default), 'growing', 'chunked', or 'fixed_size'. Auto selects 'growing' for compatible models, 'fixed_size' for legacy models."
    )
    var kvCacheStrategy: KVCacheStrategy = .auto

    @Option(
        name: .customLong("kv-cache-initial-capacity"),
        help:
            "Initial KV cache capacity in tokens. For 'growing' strategy defaults to 256, for 'fixed_size' defaults to maxContextLength"
    )
    var kvCacheInitialCapacity: Int?

    @Option(
        name: .customLong("stop-tokens"),
        help: "Additional stop tokens that will halt generation. Can be specified multiple times."
    )
    var stopTokens: [String] = []

    @Option(help: "Save logits to JSON file (path to output file)")
    var saveLogits: String?

    @Option(
        name: .customLong("save-logits-length"),
        help: "Number of top K tokens to save (1-20), or 'full' for complete logits (default: 5)"
    )
    var saveLogitsLength: LogitsLength = .count(5)

    @Option(
        name: .customLong("apply-chat-template"),
        help: "Applies default chat template from model configuration"
    )
    var applyChatTemplate: Bool = true

    @Option(
        name: .customLong("continuation"),
        help:
            "Expected continuation text for evaluation (requires --apply-chat-template=false and --print-logits or --save-logits)"
    )
    var continuation: String?

    @Flag(help: "Print top-5 token probabilities to console during generation")
    var printLogits: Bool = false

    @Option(
        name: .long,
        help: "Warmup mode: 'default' (engine-specific), 'off' (skip), or 'exact' (requires --warmup-length)")
    var warmup: WarmupMode = .defaultMode

    @Option(
        name: .customLong("warmup-length"),
        help: ArgumentHelp("Exact warmup query length (requires --warmup exact)", visibility: .hidden)
    )
    var warmupLength: Int?

    @Option(
        name: .customLong("bucket-size"),
        help: ArgumentHelp(
            "Query length bucket granularity for CoreAI (0 to disable, default: 64)", visibility: .hidden)
    )
    var bucketSize: Int?

    @Option(
        name: .customLong("chunk-size"),
        help: ArgumentHelp(
            "Prefill chunk threshold for CoreAI — prompts above this are chunked (default: 1024, use 128 for MoE)",
            visibility: .hidden)
    )
    var chunkSize: Int?

    @Option(name: .customLong("image"), help: "Path to an image file for vision-language models")
    var imagePath: String?

    @Flag(help: "Enable verbose logging")
    var verbose: Bool = false

    @Option(name: .customLong("verbose-level"), help: "Verbosity level (default: 1, implies --verbose)")
    var verboseLevel: Int?

    func validate() throws {
        if warmup == .exact && warmupLength == nil {
            throw ValidationError("--warmup exact requires --warmup-length N")
        }
        if warmup != .exact && warmupLength != nil {
            throw ValidationError("--warmup-length can only be used with --warmup exact")
        }
        if let b = bucketSize, b < 0 {
            throw ValidationError("--bucket-size must be >= 0 (0 disables bucketing)")
        }
        if let c = chunkSize, c <= 0 {
            throw ValidationError("--chunk-size must be > 0")
        }
    }

    func run() async throws {
        let verboseLevel = max(self.verboseLevel ?? 0, verbose ? 1 : 0)
        CLILogger.setLevel(to: verboseLevel)

        let resolver = ModelPaths()
        let resolvedPath = try validateAndResolveModelPath(resolver: resolver)

        // Test signpost right at the start
        InstrumentsProfiler.logMemoryUsage(phase: "AppStart")

        try await runModel(path: resolvedPath, resolver: resolver)
    }

    func validateAndResolveModelPath(resolver: ModelPaths) throws -> String {
        guard let path = model else {
            print("Error: --model is required")
            throw ExitCode.failure
        }

        guard let url = resolver.resolve(path) else {
            print("Error: \(resolver.notFoundError(for: path))")
            throw ExitCode.failure
        }

        return url.path
    }

    /// Resolve the effective prompt from CLI options (mutually exclusive)
    func resolvePromptInput() throws -> PromptInput {
        do {
            let input = try PromptInputResolver.resolve(
                prompt: prompt,
                promptFile: promptFile,
                rawTokens: rawTokens,
                default: "Hello, how are you?"
            )

            // Add CLI-specific logging
            switch input {
            case .text(let text):
                if promptFile != nil {
                    CLILogger.log("Loaded prompt from file (\(text.count) characters)", component: "Main")
                }
            case .rawTokens(let container):
                CLILogger.log("Loaded \(container.tokens.count) pre-tokenized tokens", component: "Main")
            }

            return input
        } catch let error as PromptInputError {
            print("Error: \(error.localizedDescription)")
            if rawTokens != nil {
                print("Expected format: {\"tokens\": [1, 2, 3, ...]}")
            }
            throw ExitCode.failure
        } catch {
            print("Error: \(error)")
            throw ExitCode.failure
        }
    }

    func runModel(path modelFile: String, resolver: ModelPaths) async throws {
        // Validate continuation mode requirements
        if let continuation = continuation {
            guard !applyChatTemplate else {
                CLILogger.log(
                    "Validation failed: chat template must be disabled for continuation mode", component: "Main")
                throw ContinuationEvaluationError.requiresDisabledChatTemplate
            }

            guard saveLogits != nil || printLogits else {
                CLILogger.log("Validation failed: logits output required for continuation mode", component: "Main")
                throw ContinuationEvaluationError.requiresLogitsOutput
            }

            guard !continuation.isEmpty else {
                CLILogger.log("Validation failed: continuation string is empty", component: "Main")
                throw ContinuationEvaluationError.emptyContinuation
            }

            CLILogger.log("Continuation evaluation mode enabled", component: "Main")
            CLILogger.log("Continuation: \"\(continuation)\"", component: "Main")
        }

        // Initialize performance metrics
        await PerformanceMetrics.shared.reset()
        await PerformanceMetrics.shared.startOverallTiming()

        // Set up verbose logging environment variable first
        let verboseLevel = max(self.verboseLevel ?? 0, verbose ? 1 : 0)
        CLILogger.setLevel(to: verboseLevel)

        // Bridge hidden CLI overrides to environment variables read by the Core AI engine
        if let b = bucketSize {
            setenv("COREAI_QUERY_BUCKET_SIZE", "\(b)", 1)
            CLILogger.log("Override: COREAI_QUERY_BUCKET_SIZE=\(b)", component: "Main")
        }
        if let c = chunkSize {
            setenv("COREAI_CHUNK_THRESHOLD", "\(c)", 1)
            CLILogger.log("Override: COREAI_CHUNK_THRESHOLD=\(c)", component: "Main")
        }

        CLILogger.log("Starting LLM Runner", component: "Main")
        CLILogger.log("Model: \(modelFile)", component: "Main")
        CLILogger.log("Sampling: \(samplingStrategy)", component: "Main")
        CLILogger.log("Temperature: \(temperature)", component: "Main")
        if let k = topK {
            CLILogger.log("TopK: \(k)", component: "Main")
        }
        if let p = topP {
            CLILogger.log("TopP: \(p)", component: "Main")
        }
        if let m = minP {
            CLILogger.log("MinP: \(m)", component: "Main")
        }
        if kvCacheStrategy != .auto {
            CLILogger.log("KV Cache Strategy: \(kvCacheStrategy.rawValue)", component: "Main")
            if let capacity = kvCacheInitialCapacity {
                CLILogger.log("KV Cache Initial Capacity: \(capacity)", component: "Main")
            }
        }

        let bundle = try LanguageBundle(from: modelFile)
        try bundle.bundle.verify()
        let modelName = bundle.name
        let modelVocabSize = bundle.vocabSize

        let assetLabel = try modelAssetTypeLabel(for: bundle.modelAssetPath)
        if !CLILogger.isVerbose {
            print("\n⏳ Preparing AI asset from \(assetLabel)...", terminator: "")
            fflush(stdout)
        }

        let samplingConfiguration = try parseSamplingStrategy()
        CLILogger.log("Sampling strategy configured: \(samplingConfiguration)", component: "Main")

        // Create inference engine
        CLILogger.log("Creating inference engine...", component: "Main")

        let engineOptions = EngineOptions(
            variant: inferenceEngineVariant,
            kvCacheStrategy: kvCacheStrategy,
            kvCacheSize: kvCacheInitialCapacity
        )

        // Parallel loading: engine compilation + tokenizer are independent.
        let modelLoadSpan = InstrumentsProfiler.beginModelLoad(name: bundle.name)
        let tokenizerLoadSpan = InstrumentsProfiler.beginTokenizerLoad(id: modelFile)
        async let tokenizerResult = bundle.loadTokenizer()

        let inferenceEngine: any InferenceEngine
        if bundle.bundle.kind == .vlm {
            // VLM: load 3 models and create VLM engine directly
            let visionURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.vision)
            let embedURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.embedding)
            let mainURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

            guard let visionConfig = bundle.visionConfig else {
                throw InferenceRuntimeError.invalidArgument(
                    "VLM bundle missing 'vision' config in metadata.json")
            }

            let baseConfig = ModelConfig(
                name: bundle.name,
                tokenizer: bundle.tokenizer,
                vocabSize: bundle.vocabSize,
                maxContextLength: bundle.maxContextLength,
                serializedModel: [mainURL.path],
                function: bundle.language.functionMap?.name(for: "main") ?? "main"
            )
            let vlmConfig = VLMModelConfig(base: baseConfig, visionConfig: visionConfig)

            let visionModel = try await PreparedModel.prepare(at: visionURL)
            let embedModel = try await PreparedModel.prepare(at: embedURL)
            let llmModel = try await PreparedModel.prepare(at: mainURL)

            inferenceEngine = try await CoreAISequentialVLMEngine(
                config: vlmConfig,
                visionModel: visionModel,
                embedModel: embedModel,
                llmModel: llmModel,
                options: engineOptions
            )
        } else {
            // Standard LLM: use EngineFactory
            let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
            let engineConfig = ModelConfig(
                name: bundle.name,
                tokenizer: bundle.tokenizer,
                vocabSize: bundle.vocabSize,
                maxContextLength: bundle.maxContextLength,
                serializedModel: [bundle.modelAssetPath],
                function: bundle.language.functionMap?.name(for: "main") ?? "main"
            )
            let configData = try JSONEncoder().encode(engineConfig)
            inferenceEngine = try await EngineFactory.createEngine(
                config: configData,
                modelURL: modelURL,
                options: engineOptions
            )
        }

        modelLoadSpan.end()
        let tokenizer = try await tokenizerResult
        tokenizerLoadSpan.end()
        CLILogger.log(
            "Tokenizer loaded from \(bundle.hasEmbeddedTokenizer ? "embedded bundle" : "HuggingFace")",
            component: "Main")

        // Read additional stop token IDs from tokenizer_config.json (e.g. <end_of_turn> for Gemma)
        let additionalEosTokenIds: [Int32]
        if let tokenizerDir = bundle.tokenizerPath {
            additionalEosTokenIds = LanguageConfig.additionalStopTokenIds(
                from: tokenizerDir, tokenizer: tokenizer)
            if !additionalEosTokenIds.isEmpty {
                CLILogger.log(
                    "Found \(additionalEosTokenIds.count) additional stop token(s) from tokenizer config: \(additionalEosTokenIds)",
                    component: "Main")
            }
        } else {
            additionalEosTokenIds = []
        }

        CLILogger.log("Model loaded successfully:", component: "Main")
        CLILogger.log("   Name: \(modelName)", component: "Main")
        CLILogger.log("   Source: model bundle", component: "Main")

        // Validate that the sampling strategy is supported by this engine
        try inferenceEngine.validateSamplingStrategy(samplingConfiguration)

        // Warmup: trigger kernel compilation before first inference
        try await performWarmup(
            mode: warmup,
            warmupLength: warmupLength,
            engine: inferenceEngine,
            samplingConfiguration: samplingConfiguration
        )

        if !CLILogger.isVerbose {
            let prepareElapsed = await PerformanceMetrics.shared.modelLoadTime
            print(" done in \(String(format: "%.3f", prepareElapsed))s\n")
        }

        // Resolve prompt input and tokenize
        let promptInput = try resolvePromptInput()
        let promptTokens: [Int]
        let input: Input
        let displayPrompt: String

        switch promptInput {
        case .text(let text):
            displayPrompt = text
            let tokenizationSpan = InstrumentsProfiler.beginTokenization(inputLength: text.count)
            input = applyChatTemplate ? .prompt(text) : .rawText(text)
            promptTokens = try PromptUtils.maybeApplyTokenizerChatTemplate(input, tokenizer: tokenizer)
            tokenizationSpan.end()

        case .rawTokens(let container):
            // Skip tokenization - use pre-tokenized input directly
            promptTokens = container.tokens.map { Int($0) }
            input = .tokens(promptTokens)

            // Use the preview method for decoded text summary
            let preview = container.preview(using: tokenizer)
            displayPrompt = preview.summary
            CLILogger.log(
                "Using \(promptTokens.count) pre-tokenized tokens (skipping tokenizer)", component: "Main")
        }

        let actualInputTokens = promptTokens.count

        // Log tokenizer completion (Jinja template compilation happens during chat template application)
        InstrumentsProfiler.logTokenizerComplete(tokenCount: actualInputTokens)

        // Token calculation (engines handle context validation internally)
        let requiredContextLength = actualInputTokens + maxTokens

        CLILogger.log("Token calculation:", component: "Main")
        CLILogger.log("   Input tokens: \(actualInputTokens)", component: "Main")
        CLILogger.log("   Max generation tokens: \(maxTokens)", component: "Main")
        CLILogger.log("   Required context length: \(requiredContextLength)", component: "Main")
        // Engines will validate context length during inference

        // VLM path: if --image is provided and engine supports multimodal
        if let imagePath = imagePath {
            try await runVLMInference(
                imagePath: imagePath,
                inferenceEngine: inferenceEngine,
                bundle: bundle,
                tokenizer: tokenizer,
                samplingConfiguration: samplingConfiguration,
                maxTokens: maxTokens,
                additionalEosTokenIds: additionalEosTokenIds,
                displayPrompt: displayPrompt
            )
            return
        }

        // Build text generator with preloaded inference engine
        CLILogger.log("Building text generator...", component: "Main")

        let generator = try await TextGeneratorBuilder()
            .withInferenceEngine(inferenceEngine)
            .withSampling(configuration: samplingConfiguration)
            .withDecoding(type: .vanilla, parameters: DecodingParameters())
            .withTokenizer(tokenizer)
            .build()

        CLILogger.log("Text generator built successfully", component: "Main")

        // Apply chat template and count tokens for metrics
        await PerformanceMetrics.shared.setPromptTokenCount(promptTokens.count)

        CLILogger.log("Generating text...", component: "Main")
        CLILogger.log("Input: \(displayPrompt)", component: "Main")
        CLILogger.log("Max tokens: \(maxTokens)", component: "Main")

        if !CLILogger.isVerbose {
            print("Generating...")
        }

        // Check if this is continuation evaluation mode
        if let continuation = continuation {
            // CONTINUATION EVALUATION MODE
            // Extract context string from prompt input
            let contextString: String
            switch promptInput {
            case .text(let text):
                contextString = text
            case .rawTokens:
                // Raw tokens not supported for continuation - requires text-based tokenization
                throw ContinuationEvaluationError.rawTokensNotSupported
            }

            let signpostID = InstrumentsProfiler.beginInference(promptTokens: actualInputTokens, maxTokens: 0)

            CLILogger.log("Running continuation evaluation...", component: "Main")

            let result = try await generator.evaluateContinuation(
                context: contextString,
                continuation: continuation
            )

            InstrumentsProfiler.endInference(
                generatedTokens: result.continuationTokens.count,
                signpostID: signpostID
            )

            // End overall timing
            await PerformanceMetrics.shared.endOverallTiming()

            // Handle evaluation output using LogitsWriter
            try LogitsWriter.handleEvaluationOutput(
                result: result,
                context: contextString,
                continuation: continuation,
                tokenizer: tokenizer,
                saveLogitsLength: saveLogitsLength,
                saveJsonPath: saveLogits,
                printToConsole: printLogits
            )
        } else if let schemaInput = jsonSchema {
            // CONSTRAINED GENERATION MODE (--json-schema)
            try await runConstrainedGeneration(
                schemaInput: schemaInput,
                input: input,
                tokenizer: tokenizer,
                inferenceEngine: inferenceEngine,
                samplingConfiguration: samplingConfiguration,
                maxTokens: maxTokens,
                actualInputTokens: actualInputTokens,
                modelVocabSize: modelVocabSize,
                additionalEosTokenIds: additionalEosTokenIds
            )
        } else {
            // Generate text (timing handled by decoding strategies)
            let inferenceID = InstrumentsProfiler.beginInference(
                promptTokens: actualInputTokens, maxTokens: maxTokens)
            let decodingID = InstrumentsProfiler.beginDecoding(strategy: "vanilla")

            // Encode stop tokens to sequences
            let stopSequences = try validateAndEncodeStopTokens(
                stopTokens: stopTokens,
                tokenizer: tokenizer,
                additionalEosTokenIds: additionalEosTokenIds
            )

            // Check if logits are requested
            let needsLogits = saveLogits != nil || printLogits
            let generatedText: String
            var allLogits: [[LogitsScalarType]] = []

            if needsLogits {
                // Generate with logits
                let result = try await generator.generateWithLogits(
                    input: input,
                    maxTokens: maxTokens,
                    stopSequences: stopSequences
                )
                generatedText = result.text
                allLogits = result.logits
            } else {
                // Standard generation without logits
                generatedText = try await generator.generate(
                    input: input,
                    maxTokens: maxTokens,
                    stopSequences: stopSequences
                )
            }

            InstrumentsProfiler.endDecoding(signpostID: decodingID)

            // Generated token count is already set by the decoding strategy
            let generatedTokenCount = await PerformanceMetrics.shared.getGeneratedTokenCount
            InstrumentsProfiler.endInference(generatedTokens: generatedTokenCount, signpostID: inferenceID)

            // End overall timing now that core inference is complete
            await PerformanceMetrics.shared.endOverallTiming()

            // Extract only the generated part (remove prompt from output)
            let formattedPrompt = tokenizer.decode(tokens: promptTokens)
            let generatedOnly =
                generatedText.hasPrefix(formattedPrompt)
                ? String(generatedText.dropFirst(formattedPrompt.count))
                : generatedText

            CLILogger.log("Generation complete!", component: "Main")
            CLILogger.log("Generated text:", component: "Main")

            print(generatedOnly)

            // Handle logits output if requested
            if needsLogits && !allLogits.isEmpty {
                try LogitsWriter.handleOutput(
                    logits: allLogits,
                    generatedText: generatedOnly,
                    tokenizer: tokenizer,
                    saveLogitsLength: saveLogitsLength,
                    saveJsonPath: saveLogits,
                    printToConsole: printLogits
                )
            }
        }

        // Print performance summary
        await PerformanceMetrics.shared.printSummary(verbose: CLILogger.isVerbose)

        // Print detailed profiling statistics table when --verbose is enabled
        if verbose {
            await StatsReporter(storage: .shared).printVerboseTable()
        }

        // Log final memory usage
        InstrumentsProfiler.logMemoryUsage(phase: "ModelFinal")

        // Cleanup
        CLILogger.log("Resources cleaned up", component: "Main")
    }

    // MARK: - Constrained Generation Helper

    private func runConstrainedGeneration(
        schemaInput: String,
        input: Input,
        tokenizer: any Tokenizer,
        inferenceEngine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration,
        maxTokens: Int,
        actualInputTokens: Int,
        modelVocabSize: Int?,
        additionalEosTokenIds: [Int32] = []
    ) async throws {
        let schema: String
        if FileManager.default.fileExists(atPath: schemaInput) {
            schema = try String(contentsOfFile: schemaInput, encoding: .utf8)
        } else {
            schema = schemaInput
        }

        CLILogger.log("Constrained generation with JSON schema", component: "Main")

        let stopSequences = try validateAndEncodeStopTokens(
            stopTokens: stopTokens,
            tokenizer: tokenizer,
            additionalEosTokenIds: additionalEosTokenIds
        )

        guard let vocabSize = modelVocabSize else {
            print(
                "Error: --json-schema requires vocab_size in model config. Add \"vocab_size\" to your model JSON or use --vocab-size."
            )
            throw ExitCode.failure
        }

        let constrainedStrategy = ConstrainedDecodingStrategy(
            jsonSchema: schema,
            vocabSize: vocabSize
        )

        let inferenceID = InstrumentsProfiler.beginInference(
            promptTokens: actualInputTokens, maxTokens: maxTokens)

        var generatedText = ""
        let constrainedStream = try await constrainedStrategy.decode(
            from: input,
            tokenizer: tokenizer,
            inferenceEngine: inferenceEngine,
            samplingConfiguration: samplingConfiguration,
            options: InferenceOptions(maxTokens: maxTokens, includeLogits: true),
            stopSequences: stopSequences
        )
        for try await result in constrainedStream {
            generatedText += result.text
            print(result.text, terminator: "")
        }
        print()

        let generatedTokenCount = tokenizer.encode(text: generatedText).count
        InstrumentsProfiler.endInference(generatedTokens: generatedTokenCount, signpostID: inferenceID)
        await PerformanceMetrics.shared.endOverallTiming()
    }

    func parseSamplingStrategy() throws -> SamplingConfiguration {
        let strategy = samplingStrategy.lowercased()

        let config: SamplingConfiguration
        switch strategy {
        case "temperature":
            config = SamplingConfiguration(
                temperature: temperature,
                topK: topK,
                topP: topP,
                minP: minP,
                combined: !synchronousSampling
            )
        case "greedy":
            // Fatal error if topK/topP/minP set with greedy
            if topK != nil || topP != nil || minP != nil {
                print("Error: --top-k, --top-p, and --min-p cannot be used with --sampling-strategy greedy")
                print("Use --sampling-strategy temperature with --top-k/--top-p/--min-p, or remove them for greedy")
                throw ExitCode.failure
            }
            config = SamplingConfiguration(temperature: 0, combined: !synchronousSampling)
        default:
            print("Error: Unknown sampling strategy '\(samplingStrategy)'")
            print("Valid options: 'temperature', 'greedy'")
            throw ExitCode.failure
        }

        // Validate and warn about suboptimal configurations
        config.validateAndWarn()
        return config
    }

    /// Validate and encode stop tokens to token sequences
    /// - Parameters:
    ///   - stopTokens: Array of stop token strings from CLI
    ///   - tokenizer: Tokenizer to use for encoding
    ///   - additionalEosTokenIds: Additional EOS token IDs from tokenizer config
    /// - Returns: StopSequences containing all valid sequences plus tokenizer EOS tokens
    func validateAndEncodeStopTokens(
        stopTokens: [String],
        tokenizer: any Tokenizer,
        additionalEosTokenIds: [Int32] = []
    ) throws -> StopSequences {
        var sequences: [[Int32]] = []

        for stopString in stopTokens {
            // Encode without adding BOS/EOS so special token strings like
            // "<end_of_turn>" resolve to their single token ID, not [BOS, id].
            let tokens = tokenizer.encode(text: stopString, addSpecialTokens: false).map { Int32($0) }

            // Fatal error for empty encodings - user explicitly requested this stop token
            guard !tokens.isEmpty else {
                print("Error: Stop token '\(stopString)' encodes to 0 tokens")
                print("Please check your stop token and ensure it contains valid characters")
                throw ExitCode.failure
            }

            sequences.append(tokens)

            // Log based on sequence length
            if tokens.count == 1 {
                CLILogger.log(
                    "Added stop token: '\(stopString)' → token ID \(tokens[0])",
                    component: "Main"
                )
            } else {
                CLILogger.log(
                    "Added stop sequence: '\(stopString)' → \(tokens.count) tokens \(tokens)",
                    component: "Main"
                )
            }
        }

        // Use new initializer that automatically includes EOS tokens from tokenizer
        return StopSequences(
            for: tokenizer,
            additionalSequences: sequences,
            additionalEosTokenIds: additionalEosTokenIds
        )
    }

    // MARK: - VLM Inference

    private func runVLMInference(
        imagePath: String,
        inferenceEngine: any InferenceEngine,
        bundle: LanguageBundle,
        tokenizer: any Tokenizer,
        samplingConfiguration: SamplingConfiguration,
        maxTokens: Int,
        additionalEosTokenIds: [Int32],
        displayPrompt: String
    ) async throws {
        guard let vlmEngine = inferenceEngine as? any MultimodalInferenceEngine else {
            print("Error: --image requires a vision-language model (engine does not support multimodal)")
            throw ExitCode.failure
        }

        let imageURL = URL(fileURLWithPath: imagePath)
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            print("Error: image not found at \(imagePath)")
            throw ExitCode.failure
        }

        guard let visionConfig = bundle.visionConfig else {
            print("Error: VLM bundle missing 'vision' config in metadata.json")
            throw ExitCode.failure
        }

        if !CLILogger.isVerbose {
            print("Generating...")
        }

        CLILogger.log("Encoding image: \(imagePath)", component: "VLM")
        let embeddedInput = try await vlmEngine.encodeImage(at: imageURL)
        CLILogger.log("Image encoded: \(embeddedInput.tokenCount) visual tokens", component: "VLM")

        // Build VLM prompt with image placeholder tokens.
        // Try using the tokenizer's chat template if available; fall back to
        // generic "USER: <image>×N \n prompt \nASSISTANT:" format.
        let imageTokenCount = embeddedInput.tokenCount
        let imageTokenId = visionConfig.imageTokenId
        let vlmTokens: [Int32]

        if let chatTemplateTokens = try? buildVLMPromptFromChatTemplate(
            prompt: displayPrompt,
            imageTokenCount: imageTokenCount,
            imageTokenId: imageTokenId,
            tokenizer: tokenizer
        ) {
            vlmTokens = chatTemplateTokens
            CLILogger.log("VLM prompt: used tokenizer chat template", component: "VLM")
        } else {
            CLILogger.log(
                "VLM prompt: no chat template found, using fallback USER/ASSISTANT format",
                component: "VLM")
            var tokens = tokenizer.encode(text: "USER: ", addSpecialTokens: true).map { Int32($0) }
            tokens.append(contentsOf: [Int32](repeating: imageTokenId, count: imageTokenCount))
            let suffix = "\n" + displayPrompt + "\nASSISTANT:"
            tokens.append(
                contentsOf: tokenizer.encode(text: suffix, addSpecialTokens: false).map { Int32($0) })
            vlmTokens = tokens
        }

        CLILogger.log(
            "VLM prompt: \(vlmTokens.count) tokens (\(imageTokenCount) image placeholders)",
            component: "VLM")

        // Build stop token set
        var eosTokenIds = Set<Int32>()
        if let eos = tokenizer.eosTokenId { eosTokenIds.insert(Int32(eos)) }
        eosTokenIds.formUnion(additionalEosTokenIds)

        let stopSequences = try validateAndEncodeStopTokens(
            stopTokens: stopTokens,
            tokenizer: tokenizer,
            additionalEosTokenIds: additionalEosTokenIds
        )
        for seq in stopSequences.sequences where seq.count == 1 {
            eosTokenIds.insert(seq[0])
        }

        let inferenceID = InstrumentsProfiler.beginInference(
            promptTokens: vlmTokens.count, maxTokens: maxTokens)

        let tokenStream = try vlmEngine.generate(
            with: embeddedInput,
            tokens: vlmTokens,
            samplingConfiguration: samplingConfiguration,
            inferenceOptions: InferenceOptions(maxTokens: maxTokens)
        )

        var generatedTokens: [Int] = []
        var previousText = ""
        for try await output in tokenStream {
            let token = output.tokenId
            if eosTokenIds.contains(token) { break }
            generatedTokens.append(Int(token))
            let fullText = tokenizer.decode(tokens: generatedTokens)
            let delta = String(fullText.dropFirst(previousText.count))
            previousText = fullText
            print(delta, terminator: "")
            fflush(stdout)
        }
        print()

        InstrumentsProfiler.endInference(
            generatedTokens: generatedTokens.count, signpostID: inferenceID)
        await PerformanceMetrics.shared.setGeneratedTokenCount(generatedTokens.count)
        await PerformanceMetrics.shared.endOverallTiming()
        await PerformanceMetrics.shared.printSummary(verbose: CLILogger.isVerbose)

        if verbose {
            await StatsReporter(storage: .shared).printVerboseTable()
        }
        InstrumentsProfiler.logMemoryUsage(phase: "ModelFinal")
    }

    /// Build a VLM prompt using the tokenizer's chat template.
    /// Returns nil if the tokenizer doesn't support multimodal chat templates
    /// or if the template doesn't produce the expected image placeholder tokens.
    private func buildVLMPromptFromChatTemplate(
        prompt: String,
        imageTokenCount: Int,
        imageTokenId: Int32,
        tokenizer: any Tokenizer
    ) throws -> [Int32]? {
        // Use the tokenizer's applyChatTemplate with a multimodal message.
        // The template should emit a single <image> token that we expand.
        let imageToken = tokenizer.convertIdToToken(Int(imageTokenId)) ?? "<image>"
        let templatedPrompt = "\(imageToken)\n\(prompt)"
        guard let tokens = try? PromptUtils.maybeApplyTokenizerChatTemplate(
            .prompt(templatedPrompt), tokenizer: tokenizer
        ) else {
            return nil
        }

        // Expand the single image placeholder to imageTokenCount copies
        var result: [Int32] = []
        result.reserveCapacity(tokens.count + imageTokenCount - 1)
        var foundPlaceholder = false
        for tokenInt in tokens {
            let token = Int32(tokenInt)
            if token == imageTokenId && !foundPlaceholder {
                result.append(contentsOf: [Int32](repeating: imageTokenId, count: imageTokenCount))
                foundPlaceholder = true
            } else if token == imageTokenId {
                // Skip additional single image tokens (already expanded the first one)
                continue
            } else {
                result.append(token)
            }
        }

        // If we never found the placeholder, the template didn't produce it — fall back
        guard foundPlaceholder else { return nil }
        return result
    }

    // MARK: - Asset Type Label

    private func modelAssetTypeLabel(for path: String) throws -> String {
        switch URL(fileURLWithPath: path).pathExtension.lowercased() {
        case "aimodelc": return "compiled"
        case "aimodel": return "source"
        default:
            print("Unsupported model file: only .aimodel or .aimodelc")
            throw ExitCode.failure
        }
    }

    // MARK: - Warmup

    /// Dispatches warmup based on the `--warmup` mode.
    /// - `default`: engine-specific default (Core AI pipelined: decode+prefill shapes; static-shape: essential graphs)
    /// - `exact` + `--warmup-length N`: warm a specific shape (GPU only; static-shape always warms essential graphs)
    /// - `off`: skip warmup entirely
    private func performWarmup(
        mode: WarmupMode,
        warmupLength: Int?,
        engine: any InferenceEngine,
        samplingConfiguration: SamplingConfiguration
    ) async throws {
        let queryLength: Int
        switch mode {
        case .defaultMode:
            queryLength = 0
            CLILogger.log("Running engine warmup (mode=default)...", component: "Main")
        case .exact:
            queryLength = warmupLength!  // validate() ensures this is set
            CLILogger.log("Running engine warmup (exact shape \(queryLength))...", component: "Main")
        case .off:
            CLILogger.log("Skipping warmup (--warmup off)", component: "Main")
            return
        }

        let span = InstrumentsProfiler.beginWarmup()
        try await engine.warmup(queryLength: queryLength, sampling: samplingConfiguration)
        span.end()

        if let warmupStats = await StatsStorage.shared.stats(for: .warmup) {
            CLILogger.log(
                "Warmup completed in \(String(format: "%.1f", warmupStats.totalSeconds * 1000))ms",
                component: "Main")
        }
    }
}
