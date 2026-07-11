// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Foundation Models protocol implementation for VLM bundles.

import CoreImage
import Foundation
import FoundationModels
import ImageIO
import Tokenizers
import UniformTypeIdentifiers

#if canImport(CoreAI)
import CoreAI

// MARK: - CoreAIVisionLanguageModel

/// Foundation Models adapter for VLM bundles.
///
/// ```swift
/// let model = try await CoreAIVisionLanguageModel(resourcesAt: vlmBundleURL)
/// let session = LanguageModelSession(model: model)
/// let response = try await session.respond {
///     Prompt {
///         Attachment(image)
///         "What is in this image?"
///     }
/// }
/// ```
@available(macOS 27, iOS 27, *)
public struct CoreAIVisionLanguageModel: LanguageModel {
    public typealias Executor = CoreAIVLMExecutor

    public var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities(capabilities: [.vision])
    }

    public var executorConfiguration: CoreAIVLMExecutor.Configuration

    /// Loads a VLM bundle and builds the backing engine.
    ///
    /// - Parameter url: URL to the bundle directory (`kind=vlm`).
    public init(resourcesAt url: URL) async throws {
        let bundle = try LanguageBundle(at: url)
        guard bundle.bundle.kind == .vlm else {
            throw InferenceRuntimeError.invalidArgument(
                "CoreAIVisionLanguageModel requires a VLM bundle (kind=vlm)")
        }
        guard let visionConfig = bundle.visionConfig else {
            throw InferenceRuntimeError.invalidArgument(
                "VLM bundle missing 'vision' config in metadata.json")
        }

        let visionURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.vision)
        let embedURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.embedding)
        let mainURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

        let baseConfig = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [mainURL.path],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let vlmConfig = VLMModelConfig(base: baseConfig, visionConfig: visionConfig)

        async let tokenizerResult = bundle.loadTokenizer()
        let visionModel = try await PreparedModel.prepare(at: visionURL)
        let embedModel = try await PreparedModel.prepare(at: embedURL)
        let llmModel = try await PreparedModel.prepare(at: mainURL)

        let engine = try await CoreAISequentialVLMEngine(
            config: vlmConfig,
            visionModel: visionModel,
            embedModel: embedModel,
            llmModel: llmModel,
            options: EngineOptions()
        )
        let tokenizer = try await tokenizerResult

        self.executorConfiguration = CoreAIVLMExecutor.Configuration(
            bundleURL: url,
            engine: engine,
            tokenizer: tokenizer,
            visionConfig: visionConfig
        )
    }

    public func validate(_ transcript: some Collection<Transcript.Entry>) async throws {}
}

// MARK: - CoreAIVLMExecutor

@available(macOS 27, iOS 27, *)
public struct CoreAIVLMExecutor: LanguageModelExecutor {
    public typealias Model = CoreAIVisionLanguageModel

    public struct Configuration: Hashable, Sendable {
        let bundleURL: URL
        let engine: CoreAISequentialVLMEngine
        let tokenizer: any Tokenizer
        let visionConfig: VisionConfig

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.bundleURL == rhs.bundleURL
        }
        public func hash(into hasher: inout Hasher) {
            hasher.combine(bundleURL)
        }
    }

    private let engine: CoreAISequentialVLMEngine
    private let tokenizer: any Tokenizer
    private let visionConfig: VisionConfig

    public init(configuration: Configuration) throws {
        self.engine = configuration.engine
        self.tokenizer = configuration.tokenizer
        self.visionConfig = configuration.visionConfig
    }

    public func prewarm(model: CoreAIVisionLanguageModel, transcript: Transcript) {}

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: CoreAIVisionLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Extract the image attachment + user text from the transcript.
        var cgImage: CGImage?
        var userText = ""
        for entry in request.transcript {
            guard case .prompt(let prompt) = entry else { continue }
            for segment in prompt.segments {
                switch segment {
                case .text(let text):
                    userText += text.content
                case .attachment(let attachment):
                    if case .image(let image) = attachment.content {
                        cgImage = image.cgImage
                    }
                default:
                    break
                }
            }
        }

        guard let cgImage else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: Array(request.transcript),
                    debugDescription:
                        "CoreAIVisionLanguageModel requires an image attachment in the prompt."
                ))
        }

        // Encode the image through the vision encoder. `encodeImage(at:)`
        let imageURL = try Self.writeTemporaryPNG(cgImage)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        try await engine.reset()
        let embeddedInput = try await engine.encodeImage(at: imageURL)

        // Build the token sequence with expanded image placeholders.
        let promptTokens = Self.buildPromptTokens(
            userText: userText,
            imageTokenCount: embeddedInput.tokenCount,
            imageTokenId: visionConfig.imageTokenId,
            tokenizer: tokenizer
        )

        // Generate, streaming decoded text deltas into the FM channel.
        let maxTokens = request.generationOptions.maximumResponseTokens ?? 512
        var stopTokens = Set<Int32>()
        if let eos = tokenizer.eosTokenId { stopTokens.insert(Int32(eos)) }
        if let imEnd = tokenizer.convertTokenToId("<|im_end|>") { stopTokens.insert(Int32(imEnd)) }

        let stream = try await engine.generate(
            with: embeddedInput,
            tokens: promptTokens,
            samplingConfiguration: SamplingConfiguration(temperature: 1.0, topK: 1),
            inferenceOptions: InferenceOptions(maxTokens: maxTokens, includeLogits: false)
        )

        var generatedTokens: [Int] = []
        var previousText = ""
        for try await output in stream {
            let token = output.tokenId
            if stopTokens.contains(token) { break }
            generatedTokens.append(Int(token))

            // Incremental decode: emit only the newly-completed suffix, and
            // hold back partial UTF-8 (replacement char) until it resolves.
            let fullText = tokenizer.decode(tokens: generatedTokens)
            let delta = String(fullText.dropFirst(previousText.count))
            if delta.unicodeScalars.contains("\u{FFFD}") { continue }
            previousText = fullText
            if !delta.isEmpty {
                await channel.send(.response(action: .appendText(delta, tokenCount: 1)))
            }
        }

        await channel.send(
            .response(
                action: .updateUsage(
                    input: .init(totalTokenCount: promptTokens.count, cachedTokenCount: 0),
                    output: .init(totalTokenCount: generatedTokens.count, reasoningTokenCount: 0)
                )))
        await Task.yield()
    }

    // MARK: - Prompt Construction

    /// Builds the token sequence for a single-image prompt.
    private static func buildPromptTokens(
        userText: String,
        imageTokenCount: Int,
        imageTokenId: Int32,
        tokenizer: any Tokenizer
    ) -> [Int32] {
        let imageToken = tokenizer.convertIdToToken(Int(imageTokenId)) ?? "<|image_pad|>"
        if let templated = try? PromptUtils.maybeApplyTokenizerChatTemplate(
            .prompt("\(imageToken)\n\(userText)"), tokenizer: tokenizer)
        {
            var result: [Int32] = []
            result.reserveCapacity(templated.count + imageTokenCount)
            var expanded = false
            for tokenInt in templated {
                let token = Int32(tokenInt)
                if token == imageTokenId {
                    if !expanded {
                        result.append(
                            contentsOf: [Int32](repeating: imageTokenId, count: imageTokenCount))
                        expanded = true
                    }
                    continue
                }
                result.append(token)
            }
            if expanded { return result }
        }

        // Fallback
        let placeholder = String(repeating: "<|image_pad|>", count: imageTokenCount)
        let chatText =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n<|vision_start|>\(placeholder)<|vision_end|>\n"
            + "\(userText)<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer.encode(text: chatText).map { Int32($0) }
    }

    // MARK: - Image Staging

    private static func writeTemporaryPNG(_ image: CGImage) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("coreai-vlm-\(UUID().uuidString).png")
        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else {
            throw InferenceRuntimeError.genericError("Failed to create image destination")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw InferenceRuntimeError.genericError("Failed to write staged image to \(url.path)")
        }
        return url
    }
}

#endif
