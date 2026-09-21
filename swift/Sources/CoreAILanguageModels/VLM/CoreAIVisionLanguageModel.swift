// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Foundation Models protocol implementation for VLM bundles.

import CoreAI
import CoreGraphics
import Foundation
import FoundationModels
import Tokenizers

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
public struct CoreAIVisionLanguageModel: LanguageModel {
    public typealias Executor = CoreAIVLMExecutor

    public var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities([.vision])
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

        // Load the tokenizer while the factory prepares the engine's components.
        async let tokenizerResult = bundle.loadTokenizer()

        let engine = try await EngineFactory.createEngine(
            bundle: bundle,
            options: EngineOptions(
                prefillChunkSize: bundle.language.prefillChunkSize,
                prefillChunkThreshold: bundle.language.prefillChunkThreshold
            )
        )
        guard let vlmEngine = engine as? CoreAISequentialVLMEngine else {
            throw InferenceRuntimeError.invalidArgument(
                "Expected a vision-language engine for a VLM bundle, got \(type(of: engine))")
        }

        let tokenizer = try await tokenizerResult

        // Turn-end tokens the model stops on beyond the main EOS (e.g. <|im_end|>,
        // Gemma's <end_of_turn>, Phi's <|end|>), resolved once at load like the text path.
        let additionalStopTokenIds = StopTokens.additionalIds(bundle: bundle, tokenizer: tokenizer)

        self.executorConfiguration = CoreAIVLMExecutor.Configuration(
            bundleURL: url,
            engine: vlmEngine,
            tokenizer: tokenizer,
            visionConfig: visionConfig,
            additionalStopTokenIds: additionalStopTokenIds
        )
    }
}

// MARK: - CoreAIVLMExecutor

public struct CoreAIVLMExecutor: LanguageModelExecutor {
    public typealias Model = CoreAIVisionLanguageModel

    public struct Configuration: Hashable, Sendable {
        let bundleURL: URL
        let engine: CoreAISequentialVLMEngine
        let tokenizer: any Tokenizer
        let visionConfig: VisionConfig
        let additionalStopTokenIds: [Int32]

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
    private let additionalStopTokenIds: [Int32]

    public init(configuration: Configuration) throws {
        self.engine = configuration.engine
        self.tokenizer = configuration.tokenizer
        self.visionConfig = configuration.visionConfig
        self.additionalStopTokenIds = configuration.additionalStopTokenIds
    }

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: CoreAIVisionLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        var cgImage: CGImage?
        var userText = ""
        for entry in request.transcript {
            guard case .prompt(let prompt) = entry else { continue }
            for segment in prompt.segments {
                switch segment {
                case .text(let text):
                    userText += text.content
                case .attachment(let attachment):
                    if cgImage == nil, case .image(let image) = attachment.content {
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

        try await engine.reset()
        let embeddedInput = try await engine.encodeImage(cgImage: cgImage)

        let promptTokens = Self.buildPromptTokens(
            userText: userText,
            imageTokenCount: embeddedInput.tokenCount,
            imageTokenId: visionConfig.imageTokenId,
            tokenizer: tokenizer
        )

        let maxTokens = request.generationOptions.maximumResponseTokens ?? 512
        let stopTokens = StopTokens.set(
            tokenizer: tokenizer, additional: additionalStopTokenIds)

        let stream = try await engine.generate(
            with: embeddedInput,
            tokens: promptTokens,
            samplingConfiguration: SamplingConfiguration(temperature: 1.0, topK: 1),
            inferenceOptions: InferenceOptions(maxTokens: maxTokens, includeLogits: false)
        )

        var generatedCount = 0
        var pendingTokens: [Int] = []
        var previousText = ""
        for try await output in stream {
            if stopTokens.contains(output.tokenId) { break }
            generatedCount += 1
            pendingTokens.append(Int(output.tokenId))

            let decoded = tokenizer.decode(tokens: pendingTokens)
            if decoded.unicodeScalars.contains("\u{FFFD}") {
                previousText = decoded
                continue
            }
            let common = decoded.commonPrefix(with: previousText)
            let delta = String(decoded.dropFirst(common.count))
            if !delta.isEmpty {
                await channel.send(.response(action: .appendText(delta, tokenCount: 1)))
            }
            if let last = pendingTokens.last {
                pendingTokens = [last]
                previousText = tokenizer.decode(tokens: pendingTokens)
            }
        }

        await channel.send(
            .response(
                action: .updateUsage(
                    input: .init(totalTokenCount: promptTokens.count, cachedTokenCount: 0),
                    output: .init(totalTokenCount: generatedCount, reasoningTokenCount: 0)
                )))
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

        // Fallback for tokenizers without a multimodal chat template. Uses the
        // Qwen3-VL ChatML format.
        let placeholder = String(repeating: "<|image_pad|>", count: imageTokenCount)
        let chatText =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n<|vision_start|>\(placeholder)<|vision_end|>\n"
            + "\(userText)<|im_end|>\n<|im_start|>assistant\n"
        return tokenizer.encode(text: chatText).map { Int32($0) }
    }
}
