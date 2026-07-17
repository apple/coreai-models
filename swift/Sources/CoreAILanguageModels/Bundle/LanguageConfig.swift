// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// `language` block of `metadata.json` schema 0.2 — LLM-specific config.
public struct LanguageConfig: Codable, Sendable, Equatable {
    public let tokenizer: String
    public let vocabSize: Int
    public let maxContextLength: Int

    /// `true` if the bundle ships its own tokenizer directory; `false` to
    /// load via HuggingFace at runtime. Defaults to `true` when omitted.
    public let embeddedTokenizer: Bool

    /// Optional override for graph-function role → physical names. When
    /// absent, the runtime probes via `AIModelAsset.summary()` and applies
    /// known role conventions (`main`, `extend_<N>`, `load_embeddings`, ...).
    public let functionMap: FunctionMap?

    /// Vision-specific configuration. Nil for text-only language models.
    public let vision: VisionConfig?

    public init(
        tokenizer: String,
        vocabSize: Int,
        maxContextLength: Int,
        embeddedTokenizer: Bool = true,
        functionMap: FunctionMap? = nil,
        vision: VisionConfig? = nil
    ) {
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.embeddedTokenizer = embeddedTokenizer
        self.functionMap = functionMap
        self.vision = vision
    }

    enum CodingKeys: String, CodingKey {
        case tokenizer
        case vocabSize = "vocab_size"
        case maxContextLength = "max_context_length"
        case embeddedTokenizer = "embedded_tokenizer"
        case functionMap = "function_map"
        case vision
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.tokenizer = try c.decode(String.self, forKey: .tokenizer)
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.maxContextLength = try c.decode(Int.self, forKey: .maxContextLength)
        self.embeddedTokenizer = try c.decodeIfPresent(Bool.self, forKey: .embeddedTokenizer) ?? true
        self.functionMap = try c.decodeIfPresent(FunctionMap.self, forKey: .functionMap)
        self.vision = try c.decodeIfPresent(VisionConfig.self, forKey: .vision)
    }

    // MARK: - Additional Stop Tokens

    /// Extract additional stop token IDs from the tokenizer config.
    /// Reads `additional_special_tokens` from tokenizer_config.json and
    /// cross-references with the tokenizer to get their IDs.
    ///
    /// Also checks for array-valued `eos_token` (some models list multiple).
    ///
    /// Best-effort: returns empty if the file doesn't exist or can't be parsed.
    ///
    /// TODO: Upstream this to swift-transformers as `Tokenizer.additionalEosTokenIds`
    /// so we don't need to parse tokenizer_config.json ourselves.
    public static func additionalStopTokenIds(
        from tokenizerDir: URL,
        tokenizer: any Tokenizer
    ) -> [Int32] {
        let configURL = tokenizerDir.appending(path: "tokenizer_config.json")
        guard let data = try? Data(contentsOf: configURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return []
        }

        let mainEos = tokenizer.eosTokenId.map { Int32($0) }
        var result = Set<Int32>()

        // 1. Check additional_special_tokens array
        if let specials = json["additional_special_tokens"] as? [Any] {
            for item in specials {
                // Each item can be a string or a dict with a "content" key
                let tokenString: String?
                if let s = item as? String {
                    tokenString = s
                } else if let dict = item as? [String: Any],
                    let content = dict["content"] as? String
                {
                    tokenString = content
                } else {
                    tokenString = nil
                }
                guard let token = tokenString else { continue }

                if let id = tokenizer.convertTokenToId(token) {
                    let id32 = Int32(id)
                    if id32 != mainEos {
                        result.insert(id32)
                    }
                }
            }
        }

        // 2. Check if eos_token is an array (some models list multiple)
        if let eosArray = json["eos_token"] as? [String] {
            for token in eosArray {
                if let id = tokenizer.convertTokenToId(token) {
                    let id32 = Int32(id)
                    if id32 != mainEos {
                        result.insert(id32)
                    }
                }
            }
        }

        // 3. Check added_tokens_decoder for turn-ending special tokens
        //    (e.g. Gemma's <end_of_turn> ID 106, Qwen's <|im_end|>)
        //    Only include tokens whose content matches known turn-ending patterns.
        let turnEndPatterns = ["end_of_turn", "im_end", "eot_id"]
        if let addedTokens = json["added_tokens_decoder"] as? [String: Any] {
            for (idString, value) in addedTokens {
                guard let dict = value as? [String: Any],
                    let isSpecial = dict["special"] as? Bool, isSpecial,
                    let content = dict["content"] as? String,
                    let id = Int32(idString)
                else { continue }
                let lower = content.lowercased()
                if id != mainEos && turnEndPatterns.contains(where: { lower.contains($0) }) {
                    result.insert(id)
                }
            }
        }

        return Array(result)
    }
}

/// Vision-specific configuration for VLM bundles.
/// Nil for text-only language models.
public struct VisionConfig: Codable, Sendable, Equatable {
    /// Input image size (square). Vision encoder expects this resolution.
    public let imageSize: Int

    /// Patch size for the vision transformer.
    public let patchSize: Int

    /// Number of embedding tokens produced per image after projection.
    public let imageTokenCount: Int

    /// Token ID used as a placeholder in the text sequence for image positions.
    public let imageTokenId: Int32

    /// Per-channel normalization mean (RGB). Defaults to CLIP values when omitted.
    public let imageMean: [Double]

    /// Per-channel normalization std (RGB). Defaults to CLIP values when omitted.
    public let imageStd: [Double]

    /// Pixel rescale factor applied before normalization. Defaults to 1.0 when omitted.
    public let rescaleFactor: Double

    /// Image preprocessing strategy. Defaults to stretch when omitted.
    public let imageStrategy: ImageStrategy

    /// Whether to include original image dimensions in the text prompt. Defaults to false.
    public let includeImageInfo: Bool

    /// CLIP normalization (Qwen VL, Pixtral, InternVL, Phi-3.5-vision).
    public static let clipMean = [0.48145466, 0.4578275, 0.40821073]
    public static let clipStd = [0.26862954, 0.26130258, 0.27577711]

    public init(
        imageSize: Int,
        patchSize: Int,
        imageTokenCount: Int,
        imageTokenId: Int32,
        imageMean: [Double]? = nil,
        imageStd: [Double]? = nil,
        rescaleFactor: Double? = nil,
        imageStrategy: ImageStrategy? = nil,
        includeImageInfo: Bool? = nil
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.imageTokenCount = imageTokenCount
        self.imageTokenId = imageTokenId
        self.imageMean = imageMean ?? Self.clipMean
        self.imageStd = imageStd ?? Self.clipStd
        self.rescaleFactor = rescaleFactor ?? 1.0
        self.imageStrategy = imageStrategy ?? .stretch
        self.includeImageInfo = includeImageInfo ?? false
    }

    enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case imageTokenCount = "image_token_count"
        case imageTokenId = "image_token_id"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case rescaleFactor = "rescale_factor"
        case imageStrategy = "image_strategy"
        case includeImageInfo = "include_image_info"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.imageSize = try c.decode(Int.self, forKey: .imageSize)
        self.patchSize = try c.decode(Int.self, forKey: .patchSize)
        self.imageTokenCount = try c.decode(Int.self, forKey: .imageTokenCount)
        self.imageTokenId = try c.decode(Int32.self, forKey: .imageTokenId)
        self.imageMean = try c.decodeIfPresent([Double].self, forKey: .imageMean) ?? Self.clipMean
        self.imageStd = try c.decodeIfPresent([Double].self, forKey: .imageStd) ?? Self.clipStd
        self.rescaleFactor = try c.decodeIfPresent(Double.self, forKey: .rescaleFactor) ?? 1.0
        self.imageStrategy = try c.decodeIfPresent(ImageStrategy.self, forKey: .imageStrategy) ?? .stretch
        self.includeImageInfo = try c.decodeIfPresent(Bool.self, forKey: .includeImageInfo) ?? false
    }
}
