// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAILMCommon
import CoreAILanguageModels
import CoreGraphics
import Foundation
import ImageIO
import TestUtilities
import Testing
import Tokenizers
import UniformTypeIdentifiers

@testable import llm_server

/// A multimodal engine that encodes an image of width `w` to `w + 1` zero tokens and generates
/// nothing. Counts encodes and records the embeddings of the last multimodal generate.
final class StubImageEngine: MultimodalInferenceEngine, @unchecked Sendable {
    struct Config: Codable, InferenceConfiguration {
        var maxContextLength: Int
    }

    struct EmptySequence: InferenceOutputSequence {
        typealias Element = InferenceOutput
        typealias Failure = any Error

        var stopReason: StopReason? { nil }
        func setStopReason(_ reason: StopReason) {}

        struct Iterator: AsyncIteratorProtocol {
            mutating func next() async throws -> InferenceOutput? { nil }
        }
        func makeAsyncIterator() -> Iterator { Iterator() }
    }

    let config = Config(maxContextLength: 4096)
    let acceptsMultimodalInput: Bool
    private(set) var encodeCount = 0
    private(set) var lastEmbeddingsTokenCount: Int?

    init(acceptsMultimodalInput: Bool = true) {
        self.acceptsMultimodalInput = acceptsMultimodalInput
    }

    func generate(
        with input: [TokenId], samplingConfiguration: SamplingConfiguration, inferenceOptions: InferenceOptions
    ) async throws -> EmptySequence { EmptySequence() }

    func generate(
        with input: InputEmbeddings, tokens: [TokenId], samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> EmptySequence {
        lastEmbeddingsTokenCount = input.tokenCount
        return EmptySequence()
    }

    func reset(to tokenIndex: Int) async throws {}

    func encodeImage(at url: URL) async throws -> InputEmbeddings {
        throw InferenceRuntimeError.invalidState("not used")
    }

    func encodeImage(cgImage: CGImage) async throws -> InputEmbeddings {
        encodeCount += 1
        let tokenCount = cgImage.width + 1
        return try InputEmbeddings(
            embeddings: NDArray(shape: [1, tokenCount, 2], scalarType: .float16), embeddingPositions: 0..<tokenCount)
    }

    func encodeVideo(_ video: VideoInput) async throws -> InputEmbeddings {
        throw InferenceRuntimeError.invalidState("not used")
    }
}

@Suite("Chat requests with images")
struct ImageInputTests {
    static let imageTokenId = Int32(ByteLevelTokenizerFixture.firstAdditionalTokenId)
    /// Renders content as text.
    static let textTemplate = "{%- for m in messages %}[{{ m.role }}]{{ m.content }}\n{% endfor -%}"
    /// Takes content as a list of parts and writes its own image token, as SmolVLM's template does.
    static let partsTemplate = """
        {%- for m in messages %}[{{ m.role }}]{% for p in m.content %}{% if p.type == 'text' %}{{ p.text }}\
        {% elif p.type == 'image' %}<image>{% endif %}{% endfor %}{% endfor -%}
        """

    /// A `width` x `height` PNG as a data URL.
    static func pngURL(width: Int = 1, height: Int = 1) throws -> String {
        let context = try #require(
            CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let data = NSMutableData()
        let destination = try #require(
            CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try #require(context.makeImage()), nil)
        #expect(CGImageDestinationFinalize(destination))
        return "data:image/png;base64,\((data as Data).base64EncodedString())"
    }

    static func imagePart(_ url: String) -> String {
        #"{"type":"image_url","image_url":{"url":"\#(url)"}}"#
    }

    static func request(_ messages: String, extra: String = "") throws -> ChatCompletionRequest {
        try JSONDecoder().decode(
            ChatCompletionRequest.self, from: Data(#"{"messages":\#(messages),"max_tokens":4\#(extra)}"#.utf8))
    }

    static func makeState(engine: StubImageEngine, template: String = textTemplate) async throws -> ServerState {
        let directory = try ByteLevelTokenizerFixture.directory(
            chatTemplate: template, additionalSpecialTokens: ["<image>"])
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = ServerConfig(
            modelName: "stub", defaultMaxTokens: 4, defaultTemperature: 0, defaultTopP: nil, defaultTopK: nil,
            defaultMinP: nil, defaultReasoningEffort: nil, supportsLogprobs: false, maxContextLength: 4096,
            vocabSize: nil, additionalEosTokenIds: [], maxQueueDepth: 1)
        return ServerState(
            engine: engine, tokenizer: try await AutoTokenizer.from(modelFolder: directory), config: config,
            visionConfig: VisionConfig(imageSize: 16, patchSize: 16, imageTokenCount: 2, imageTokenId: imageTokenId))
    }

    enum BadRequest: String, CaseIterable {
        case remoteURL, textDataURL, rawDataURL, notAnImage, oversizedImage, modelWithoutVision, jsonSchema
        case imageTokenInText, imageTokenInTextOnly
    }

    @Test("Images the model or server cannot take are a 400", arguments: BadRequest.allCases)
    func badRequests(_ bad: BadRequest) async throws {
        let url: String
        switch bad {
        case .remoteURL: url = "https://example.com/cat.png"
        case .textDataURL: url = "data:text/plain;base64,aGVsbG8="
        case .rawDataURL: url = "data:image/png,rawbytes"
        case .notAnImage: url = "data:image/png;base64,\(Data("not an image".utf8).base64EncodedString())"
        case .oversizedImage: url = try Self.pngURL(width: ImageInput.maxImageSide + 1)
        default: url = try Self.pngURL()
        }
        let text = bad == .imageTokenInText || bad == .imageTokenInTextOnly ? "<image>" : "Describe."
        let image = bad == .imageTokenInTextOnly ? "" : Self.imagePart(url) + ","
        let request = try Self.request(
            #"[{"role":"user","content":[\#(image){"type":"text","text":"\#(text)"}]}]"#,
            extra: bad == .jsonSchema
                ? #","response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}"#
                : "")
        let engine = StubImageEngine(acceptsMultimodalInput: bad != .modelWithoutVision)
        let state = try await Self.makeState(engine: engine)
        await #expect(throws: ServerError.self) {
            _ = try await makeChatPrompt(request, reasoningEffort: nil, state: state, sessionID: "s")
        }
        #expect(engine.encodeCount == 0)
    }

    @Test("A session's later turns and rewinds decode with its cached embeddings; other sessions encode")
    func sessionImageCache() async throws {
        let engine = StubImageEngine()
        let state = try await Self.makeState(engine: engine)
        let user = #"{"role":"user","content":[\#(Self.imagePart(try Self.pngURL())),{"type":"text","text":"What?"}]}"#
        let turns: [(String?, String, Int)] = [
            ("s", "[\(user)]", 1),
            ("s", #"[\#(user),{"role":"assistant","content":"A dot."},{"role":"user","content":"Color?"}]"#, 1),
            ("s", #"[{"role":"system","content":"Be brief."},\#(user)]"#, 1),
            ("t", "[\(user)]", 2),
            (nil, "[\(user)]", 3),
        ]
        for (sessionID, messages, encodes) in turns {
            _ = try await runChatCompletion(chatRequest: try Self.request(messages), state: state, sessionID: sessionID)
            #expect(engine.encodeCount == encodes)
            #expect(engine.lastEmbeddingsTokenCount == 2)
        }
    }

    @Test("Each image's placeholder expands to its own token count, in order")
    func multiImagePrompt() async throws {
        let state = try await Self.makeState(engine: StubImageEngine(), template: Self.partsTemplate)
        let request = try Self.request(
            #"[{"role":"user","content":[\#(Self.imagePart(try Self.pngURL(width: 1))),{"type":"text","text":"and"},"#
                + #"\#(Self.imagePart(try Self.pngURL(width: 2)))]}]"#)
        let prompt = try await makeChatPrompt(request, reasoningEffort: nil, state: state, sessionID: "s")
        let image = Int(Self.imageTokenId)
        #expect(prompt.tokens.split { $0 != image }.map(\.count) == [2, 3])
        #expect(state.tokenizer.decode(tokens: prompt.tokens.filter { $0 != image }) == "[user]and")
        #expect(try #require(prompt.images).joined.tokenCount == 5)
    }
}
