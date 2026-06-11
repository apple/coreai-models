// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Foundation Models protocol adapter for VLM bundles.
// Wires VisionEncoderRunner + EmbeddingsInferenceEngine into a LanguageModel
// that processes image attachments from Transcript prompts.

import CoreAI
import CoreAIShared
import CoreImage
import CoreVideo
import Foundation
import FoundationModels
import Tokenizers



// MARK: - CoreAIVisionLanguageModel

#if canImport(CoreAI)
/// Foundation Models adapter for VLM .llmasset bundles.
///
/// ```swift
/// let model = try await CoreAIVisionLanguageModel(resourcesAt: vlmBundleURL)
/// let session = LanguageModelSession(model: model)
/// let response = try await session.respond(options: options) {
///     Attachment(imageURL: imageURL)
///     "What is in this image?"
/// }
/// ```
@available(macOS 27, iOS 27, *)
public struct CoreAIVisionLanguageModel: LanguageModel {

    @_spi(_) public typealias Executor = CoreAIVLMExecutor

    @_spi(_)
    public var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities(capabilities: [])
    }

    @_spi(DoNotImport) public var executorConfiguration: CoreAIVLMExecutor.Configuration

    public init(resourcesAt url: URL) async throws {
        let bundle = try LanguageBundle(at: url)

        guard let visionAssetURL = bundle.modelURL(for: ModelBundle.ComponentKey.vision) else {
            throw InferenceRuntimeError.genericError("VLM bundle missing 'vision' component in assets")
        }

        guard let vlmConfig = bundle.bundle.vlm else {
            throw InferenceRuntimeError.genericError("VLM bundle missing 'vision' config block in metadata.json")
        }

        let bundlePath = bundle.bundlePath
        let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        let embedTableURL = bundlePath.appending(path: vlmConfig.embedTokensPath)

        let config = try JSONDecoder().decode(ModelConfig.self, from: bundle.rawMetadata)
        let tokenizer = try await bundle.loadTokenizer()
        let visionRunner = try await VisionEncoderRunner(
            assetURL: visionAssetURL.resolvingSymlinksInPath())
        let engine = try await EmbeddingsInferenceEngine(
            config: config, modelURL: modelURL.resolvingSymlinksInPath(), embedTableURL: embedTableURL)

        self.executorConfiguration = CoreAIVLMExecutor.Configuration(
            visionRunner: visionRunner, engine: engine, tokenizer: tokenizer,
            embedTableURL: embedTableURL, hiddenSize: vlmConfig.hiddenSize,
            imageTokenId: Int32(vlmConfig.imageTokenId), numVisualTokens: vlmConfig.numVisualTokens)
    }

    public func validate(_ transcript: some Collection<Transcript.Entry>) async throws {}
}

// MARK: - CoreAIVLMExecutor

@available(macOS 27, iOS 27, *)
@_spi(DoNotImport) @_spi(Implicit)
public struct CoreAIVLMExecutor: LanguageModelExecutor {
    @_spi(DoNotImport) public typealias Model = CoreAIVisionLanguageModel

    @_spi(DoNotImport)
    public struct Configuration: Hashable, Sendable {
        let visionRunner: VisionEncoderRunner
        let engine: EmbeddingsInferenceEngine
        let tokenizer: any Tokenizer
        let embedTableURL: URL
        let hiddenSize: Int
        let imageTokenId: Int32
        let numVisualTokens: Int

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.embedTableURL == rhs.embedTableURL
        }
        public func hash(into hasher: inout Hasher) {
            hasher.combine(embedTableURL)
        }
    }

    private let visionRunner: VisionEncoderRunner
    private let engine: EmbeddingsInferenceEngine
    private let tokenizer: any Tokenizer
    private let embedTableURL: URL
    private let hiddenSize: Int
    private let imageTokenId: Int32
    private let numVisualTokens: Int

    @_spi(_)
    public init(configuration: Configuration) throws {
        self.visionRunner = configuration.visionRunner
        self.engine = configuration.engine
        self.tokenizer = configuration.tokenizer
        self.embedTableURL = configuration.embedTableURL
        self.hiddenSize = configuration.hiddenSize
        self.imageTokenId = configuration.imageTokenId
        self.numVisualTokens = configuration.numVisualTokens
    }

    public func prewarm(transcript: Transcript) throws {}

    @_spi(_)
    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: CoreAIVisionLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        var cgImage: CGImage? = nil
        var userText = ""

        for entry in request.transcript {
            switch entry {
            case .prompt(let prompt):
                for segment in prompt.segments {
                    switch segment {
                    case .text(let text): userText += text.content
                    case .attachment(let attachment):
                        if case .image(let img) = attachment.content { cgImage = img.cgImage }
                    default: break
                    }
                }
            default: break
            }
        }

        let maxTokens = request.generationOptions.maximumResponseTokens ?? 256
        try await engine.reset()

        guard let image = cgImage else {
            await channel.send(.response(action: .appendText(
                "No image provided. Attach an image to your prompt.",
                segmentID: nil, tokenCount: 1)))
            return
        }

        // Preprocess image → vision encoder
        let pixelValues = try preprocessCGImage(image, targetSize: 448)
        let imageFeatures = try await visionRunner.run(
            pixelValues: pixelValues, numVisualTokens: numVisualTokens, hiddenSize: hiddenSize)

        // Build chat text with image placeholders
        let placeholder = String(repeating: "<|image_pad|>", count: numVisualTokens)
        let chatText = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
            "<|im_start|>user\n<|vision_start|>\(placeholder)<|vision_end|>\n" +
            "\(userText)<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = tokenizer.encode(text: chatText)

        // Build combined embeddings
        let embeddings = try buildCombinedEmbeddings(inputIds: inputIds, imageFeatures: imageFeatures)

        // Prefill — send progress metadata so callers know we're working
        let seqLen = inputIds.count
        await channel.send(.response(action: .updateMetadata([
            "prefill_status": "started" as (any Sendable & Codable & Equatable),
            "prefill_tokens": seqLen as (any Sendable & Codable & Equatable),
        ])))
        var lastLogits = try await engine.prefillWithEmbeddings(embeddings, seqLen: seqLen) { done, total in
            CLILogger.log("Prefill \(done)/\(total)")
        }
        await channel.send(.response(action: .updateMetadata([
            "prefill_status": "done" as (any Sendable & Codable & Equatable),
        ])))

        // Generate + stream
        let eosId: Int32 = 151645
        let eotId: Int32 = 151643
        var samplingConfig = SamplingConfiguration(temperature: 0.7)
        var token = samplingConfig.fallbackSampler(from: &lastLogits)
        var generated: [Int32] = []
        var prevDecoded = ""

        for _ in 0..<maxTokens {
            if token == eosId || token == eotId { break }
            generated.append(token)
            let fullText = tokenizer.decode(tokens: generated.map { Int($0) })
            let delta = String(fullText.dropFirst(fullText.commonPrefix(with: prevDecoded).count))
            if !delta.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
                prevDecoded = fullText
                await channel.send(.response(action: .appendText(delta, segmentID: nil, tokenCount: 1)))
            }
            let (_, next) = try await engine.inference(
                inputTokens: [token], samplingConfig: samplingConfig, returnsLogits: false)
            token = next
        }
    }

    // MARK: - Image Preprocessing (448×448 → [784, 1536] patches)

    private func preprocessCGImage(_ cgImage: CGImage, targetSize: Int) throws -> Data {
        let patchSize = 16, temporalPatchSize = 2, mergeSize = 2, channels = 3
        let ci = CIImage(cgImage: cgImage)
        let sx = CGFloat(targetSize) / ci.extent.width
        let sy = CGFloat(targetSize) / ci.extent.height
        let resized = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, targetSize, targetSize,
                            kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferWidthKey: targetSize,
                             kCVPixelBufferHeightKey: targetSize,
                             kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA] as CFDictionary,
                            &pb)
        guard let pb else { throw InferenceRuntimeError.genericError("CVPixelBuffer creation failed") }
        CIContext(options: [.useSoftwareRenderer: false]).render(resized, to: pb)

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let base = CVPixelBufferGetBaseAddress(pb)!
        let bpr = CVPixelBufferGetBytesPerRow(pb)

        var img = [Float](repeating: 0, count: channels * targetSize * targetSize)
        for y in 0..<targetSize {
            let row = base.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self)
            for x in 0..<targetSize {
                let o = x * 4
                img[0 * targetSize * targetSize + y * targetSize + x] = (Float(row[o+2])/255.0 - 0.5) / 0.5
                img[1 * targetSize * targetSize + y * targetSize + x] = (Float(row[o+1])/255.0 - 0.5) / 0.5
                img[2 * targetSize * targetSize + y * targetSize + x] = (Float(row[o+0])/255.0 - 0.5) / 0.5
            }
        }

        let gridH = targetSize / patchSize, gridW = targetSize / patchSize
        let patchDim = channels * temporalPatchSize * patchSize * patchSize
        let numPatches = gridH * gridW
        var patches = [Float](repeating: 0, count: numPatches * patchDim)
        let hB = gridH / mergeSize, wB = gridW / mergeSize

        for hb in 0..<hB {
            for wb in 0..<wB {
                for mh in 0..<mergeSize {
                    for mw in 0..<mergeSize {
                        let pi = hb*wB*mergeSize*mergeSize + wb*mergeSize*mergeSize + mh*mergeSize + mw
                        var off = 0
                        for c in 0..<channels {
                            for _ in 0..<temporalPatchSize {
                                for ph in 0..<patchSize {
                                    for pw in 0..<patchSize {
                                        let sy = (hb*mergeSize+mh)*patchSize+ph
                                        let sx = (wb*mergeSize+mw)*patchSize+pw
                                        patches[pi*patchDim+off] = img[c*targetSize*targetSize+sy*targetSize+sx]
                                        off += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return Data(bytes: patches, count: patches.count * MemoryLayout<Float>.size)
    }

    // MARK: - Embedding Builder

    private func buildCombinedEmbeddings(inputIds: [Int], imageFeatures: [LogitsScalarType]) throws -> Data {
        let embedData = try Data(contentsOf: embedTableURL)
        let seqLen = inputIds.count
        var combined = [LogitsScalarType](repeating: 0, count: seqLen * hiddenSize)

        embedData.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: LogitsScalarType.self)
            var vi = 0
            for i in 0..<seqLen {
                let tid = Int32(inputIds[i])
                if tid == imageTokenId && vi < numVisualTokens {
                    let src = vi * hiddenSize
                    for h in 0..<hiddenSize { combined[i*hiddenSize+h] = imageFeatures[src+h] }
                    vi += 1
                } else {
                    let src = Int(tid) * hiddenSize
                    for h in 0..<hiddenSize { combined[i*hiddenSize+h] = table[src+h] }
                }
            }
        }
        return Data(bytes: combined, count: combined.count * MemoryLayout<LogitsScalarType>.size)
    }
}



#endif
