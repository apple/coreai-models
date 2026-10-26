// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

/// Dual-RoPE parameters for models that precompute cos/sin in the runner (Gemma4
/// large-context). The graph takes precomputed `rope_cos`/`rope_sin` rows instead
/// of `position_ids`, so the runner builds the combined sliding+global table rows
/// per step from these. nil for models that gather RoPE in-graph from `position_ids`.
public struct RoPEConfig: Codable, Sendable, Equatable {
    public let slidingHeadDim: Int
    public let globalHeadDim: Int
    public let slidingRopeTheta: Double
    public let globalRopeTheta: Double
    public let partialRotaryFactor: Double

    public init(
        slidingHeadDim: Int,
        globalHeadDim: Int,
        slidingRopeTheta: Double,
        globalRopeTheta: Double,
        partialRotaryFactor: Double
    ) {
        self.slidingHeadDim = slidingHeadDim
        self.globalHeadDim = globalHeadDim
        self.slidingRopeTheta = slidingRopeTheta
        self.globalRopeTheta = globalRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
    }

    enum CodingKeys: String, CodingKey {
        case slidingHeadDim = "sliding_head_dim"
        case globalHeadDim = "global_head_dim"
        case slidingRopeTheta = "sliding_rope_theta"
        case globalRopeTheta = "global_rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

/// Model-specific runtime settings, from the `language.overrides` block of
/// `metadata.json`, for exports that leave part of the model's work to the runner.
/// Absent for most models.
public struct LanguageOverrides: Codable, Sendable, Equatable {
    /// Sliding-window size for models with a sliding KV cache (Gemma4). The
    /// runner uses it to build the windowed `sliding_causal_mask`. nil when the
    /// model has no sliding-window attention.
    public let slidingWindow: Int?

    /// Dual-RoPE parameters (Gemma4 large-context). When present, the graph takes
    /// precomputed `rope_cos`/`rope_sin` and the runner builds the rows from these;
    /// nil for models that gather RoPE in-graph from `position_ids`.
    public let rope: RoPEConfig?

    /// Final-logit soft cap `c` for `c · tanh(logits / c)` (Gemma family). Present when
    /// the export left the cap out of the graph — `tanh` is best run on the CPU rather
    /// than in the graph — so the runner must apply it on the CPU before sampling. nil
    /// when the model has no cap, or when the graph already applies it.
    public let finalLogitSoftcapping: Double?

    public init(slidingWindow: Int? = nil, rope: RoPEConfig? = nil, finalLogitSoftcapping: Double? = nil) {
        self.slidingWindow = slidingWindow
        self.rope = rope
        self.finalLogitSoftcapping = finalLogitSoftcapping
    }

    enum CodingKeys: String, CodingKey {
        case slidingWindow = "sliding_window"
        case rope
        case finalLogitSoftcapping = "final_logit_softcapping"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.rope = try c.decodeIfPresent(RoPEConfig.self, forKey: .rope)
        self.finalLogitSoftcapping = try c.decodeIfPresent(Double.self, forKey: .finalLogitSoftcapping)
        // The runner divides by the cap.
        if let finalLogitSoftcapping, finalLogitSoftcapping <= 0 {
            throw DecodingError.dataCorruptedError(
                forKey: .finalLogitSoftcapping, in: c,
                debugDescription: "final_logit_softcapping must be positive, got \(finalLogitSoftcapping)")
        }
    }
}

/// `language` block of `metadata.json` schema 0.2 — LLM-specific config.
public struct LanguageConfig: Codable, Sendable, Equatable {
    public let tokenizer: String
    public let vocabSize: Int
    public let maxContextLength: Int

    /// Largest batch the graph accepts. 1 for single-sequence graphs and for bundles without `max_batch_size`.
    package let maxBatchSize: Int

    /// `true` if the bundle ships its own tokenizer directory; `false` to
    /// load via HuggingFace at runtime. Defaults to `true` when omitted.
    public let embeddedTokenizer: Bool

    /// Optional override for graph-function role → physical names. When
    /// absent, the runtime probes via `AIModelAsset.summary()` and applies
    /// known role conventions (`main`, `extend_<N>`, `load_embeddings`, ...).
    public let functionMap: FunctionMap?

    /// Vision-specific configuration. Nil for text-only language models.
    public let vision: VisionConfig?

    /// Explicit state classification. Nil = use shape-based heuristic.
    /// Keys are state names from the model descriptor, values are StateKind.
    public let states: [String: StateKind]?

    /// Optional prefill chunk size override from metadata.json.
    public let prefillChunkSize: Int?

    /// Optional chunk threshold override from metadata.json.
    public let prefillChunkThreshold: Int?

    /// Model-specific runtime settings (`overrides`); nil for most models.
    public let overrides: LanguageOverrides?

    public init(
        tokenizer: String,
        vocabSize: Int,
        maxContextLength: Int,
        embeddedTokenizer: Bool = true,
        functionMap: FunctionMap? = nil,
        vision: VisionConfig? = nil,
        states: [String: StateKind]? = nil,
        prefillChunkSize: Int? = nil,
        prefillChunkThreshold: Int? = nil,
        overrides: LanguageOverrides? = nil
    ) {
        self.tokenizer = tokenizer
        self.vocabSize = vocabSize
        self.maxContextLength = maxContextLength
        self.embeddedTokenizer = embeddedTokenizer
        self.functionMap = functionMap
        self.vision = vision
        self.states = states
        self.prefillChunkSize = prefillChunkSize
        self.prefillChunkThreshold = prefillChunkThreshold
        self.overrides = overrides
        self.maxBatchSize = 1
    }

    enum CodingKeys: String, CodingKey {
        case tokenizer
        case vocabSize = "vocab_size"
        case maxContextLength = "max_context_length"
        case embeddedTokenizer = "embedded_tokenizer"
        case functionMap = "function_map"
        case vision
        case states
        case prefillChunkSize = "prefill_chunk_size"
        case prefillChunkThreshold = "prefill_chunk_threshold"
        case overrides
        case maxBatchSize = "max_batch_size"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.tokenizer = try c.decode(String.self, forKey: .tokenizer)
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.maxContextLength = try c.decode(Int.self, forKey: .maxContextLength)
        guard vocabSize > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .vocabSize, in: c, debugDescription: "vocab_size must be positive, got \(vocabSize)")
        }
        guard maxContextLength > 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .maxContextLength, in: c,
                debugDescription: "max_context_length must be positive, got \(maxContextLength)")
        }
        self.embeddedTokenizer = try c.decodeIfPresent(Bool.self, forKey: .embeddedTokenizer) ?? true
        self.functionMap = try c.decodeIfPresent(FunctionMap.self, forKey: .functionMap)
        self.vision = try c.decodeIfPresent(VisionConfig.self, forKey: .vision)
        self.states = try c.decodeIfPresent([String: StateKind].self, forKey: .states)
        self.prefillChunkSize = try c.decodeIfPresent(Int.self, forKey: .prefillChunkSize)
        self.prefillChunkThreshold = try c.decodeIfPresent(Int.self, forKey: .prefillChunkThreshold)
        self.overrides = try c.decodeIfPresent(LanguageOverrides.self, forKey: .overrides)
        self.maxBatchSize = try c.decodeIfPresent(Int.self, forKey: .maxBatchSize) ?? 1
    }

    // MARK: - Additional Stop Tokens

    /// Resolve the EOS-like token IDs (beyond the tokenizer's main EOS) that
    /// should terminate generation, as a `Set` for order-independent membership.
    ///
    /// Reads `additional_special_tokens` from tokenizer_config.json and
    /// cross-references with the tokenizer to get their IDs.
    ///
    /// Also checks for array-valued `eos_token` (some models list multiple).
    ///
    /// Scans `tokenizer.json`'s `added_tokens` for turn-ending specials. Fast
    /// tokenizers saved via `save_pretrained` (e.g. Gemma) keep their added
    /// specials there and omit `added_tokens_decoder` from
    /// tokenizer_config.json, so this is where the exported bundle's
    /// `<end_of_turn>` (ID 106) is found.
    ///
    /// Best-effort on the config: if tokenizer_config.json is missing or can't be
    /// parsed the config-derived IDs are skipped.
    ///
    /// TODO: Upstream the config parsing to swift-transformers as
    /// `Tokenizer.additionalEosTokenIds` so we don't parse tokenizer_config.json
    /// ourselves.
    public static func additionalStopTokenIds(
        from tokenizerDir: URL,
        tokenizer: any Tokenizer
    ) -> Set<Int32> {
        let mainEos = tokenizer.eosTokenId.map { Int32($0) }
        var result = Set<Int32>()

        let turnEndPatterns = ["end_of_turn", "im_end", "eot_id", "endoftext", "eot_token", "|eot|", "|end|"]

        let configURL = tokenizerDir.appending(path: "tokenizer_config.json")
        if let data = try? Data(contentsOf: configURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
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

                    if let id = tokenizer.vocabContains(token) ? tokenizer.convertTokenToId(token) : nil {
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
                    if let id = tokenizer.vocabContains(token) ? tokenizer.convertTokenToId(token) : nil {
                        let id32 = Int32(id)
                        if id32 != mainEos {
                            result.insert(id32)
                        }
                    }
                }
            }

            // 3. Check added_tokens_decoder for turn-ending special tokens, matched
            //    against the shared pattern list above (covers Gemma's <end_of_turn>,
            //    Qwen's <|im_end|>, Phi's <|end|>, etc., when present here — not every
            //    model exposes added_tokens_decoder; see step 5 for the fallback).
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

            // 4. Check for turn-ending tokens in in top level of config
            for turnEndPattern in turnEndPatterns {
                if let eotToken = json[turnEndPattern] as? String {
                    if let id = tokenizer.vocabContains(eotToken) ? tokenizer.convertTokenToId(eotToken) : nil {
                        let id32 = Int32(id)
                        if id32 != mainEos {
                            result.insert(id32)
                        }
                    }
                }
            }

            // 5. Check tokenizer.json's added_tokens for turn-ending specials.
            //    Fast tokenizers saved via save_pretrained store their added
            //    specials here and drop added_tokens_decoder from
            //    tokenizer_config.json, so this recovers Gemma's <end_of_turn>
            //    (ID 106) and Phi's <|end|> (ID 200020) in exported bundles.
            let tokenizerURL = tokenizerDir.appending(path: "tokenizer.json")
            if let tokenizerData = try? Data(contentsOf: tokenizerURL),
                let tokenizerJSON = try? JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any],
                let addedTokens = tokenizerJSON["added_tokens"] as? [[String: Any]]
            {
                for entry in addedTokens {
                    guard let isSpecial = entry["special"] as? Bool, isSpecial,
                        let content = entry["content"] as? String,
                        let id = entry["id"] as? Int
                    else { continue }
                    let id32 = Int32(id)
                    let lower = content.lowercased()
                    if id32 != mainEos && turnEndPatterns.contains(where: { lower.contains($0) }) {
                        result.insert(id32)
                    }
                }
            }
        }

        return result
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

    /// Whether this model supports video (multi-frame) input.
    public var supportsVideo: Bool { maxVideoFrames != nil }

    /// Maximum number of video frames for multi-frame models. Nil for image-only models.
    public let maxVideoFrames: Int?

    /// Visual tokens produced per frame. Nil defaults to `imageTokenCount`.
    public let tokensPerFrame: Int?

    /// Begin-of-image / end-of-image marker token IDs of the model's image region (Gemma4:
    /// `<|image>` / `<image|>`), as the checkpoint defines them. Nil for models that don't
    /// delimit the image region (e.g. Qwen). They are recorded for prompt builders; the
    /// builders that use the tokenizer's chat template do not add them.
    public let boiTokenId: Int32?
    public let eoiTokenId: Int32?

    /// Pad token ID. Used to mask image-placeholder positions before the per-layer
    /// embedding gather (matching HF's `where(image_mask, pad_token_id, input_ids)`),
    /// so a placeholder's own per-layer embedding isn't spuriously injected. Nil for
    /// bundles exported before this field existed — callers fall back to `0`.
    public let padTokenId: Int32?

    /// How the encoder's soft tokens reach the decoder. Absent in bundles that predate the field,
    /// which read as ``VisionTokenMerge/separateEmbedding``.
    public let tokenMerge: VisionTokenMerge

    /// Patch count per pooled soft token, along one axis (Gemma4's 3x3 average pool).
    /// Nil for models with no pooling stage (e.g. Qwen).
    public let poolingKernelSize: Int?

    /// Square canvas sizes (px) of a variable-input-shape vision encoder (Gemma4), smallest
    /// first, with one exported function per size (see ``encoderFunctionName(imageSize:)``). Nil
    /// for a fixed single-size encoder (e.g. Qwen), which uses `imageSize`/`imageTokenCount`.
    public let imageSizes: [Int]?

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
        includeImageInfo: Bool? = nil,
        maxVideoFrames: Int? = nil,
        tokensPerFrame: Int? = nil,
        boiTokenId: Int32? = nil,
        eoiTokenId: Int32? = nil,
        padTokenId: Int32? = nil,
        poolingKernelSize: Int? = nil,
        imageSizes: [Int]? = nil,
        tokenMerge: VisionTokenMerge = .separateEmbedding
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
        self.maxVideoFrames = maxVideoFrames
        self.tokensPerFrame = tokensPerFrame
        self.boiTokenId = boiTokenId
        self.eoiTokenId = eoiTokenId
        self.padTokenId = padTokenId
        self.poolingKernelSize = poolingKernelSize
        self.imageSizes = imageSizes
        self.tokenMerge = tokenMerge
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
        case maxVideoFrames = "max_video_frames"
        case tokensPerFrame = "tokens_per_frame"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case padTokenId = "pad_token_id"
        case poolingKernelSize = "pooling_kernel_size"
        case imageSizes = "image_sizes"
        case tokenMerge = "token_merge"
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
        self.maxVideoFrames = try c.decodeIfPresent(Int.self, forKey: .maxVideoFrames)
        self.tokensPerFrame = try c.decodeIfPresent(Int.self, forKey: .tokensPerFrame)
        self.boiTokenId = try c.decodeIfPresent(Int32.self, forKey: .boiTokenId)
        self.eoiTokenId = try c.decodeIfPresent(Int32.self, forKey: .eoiTokenId)
        self.padTokenId = try c.decodeIfPresent(Int32.self, forKey: .padTokenId)
        self.poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize)
        self.imageSizes = try c.decodeIfPresent([Int].self, forKey: .imageSizes)
        self.tokenMerge = try c.decodeIfPresent(VisionTokenMerge.self, forKey: .tokenMerge) ?? .separateEmbedding
        try validate(in: c)
    }

    /// Side in pixels of one pooled soft-token cell: `patchSize * poolingKernelSize`.
    public var blockPixels: Int { patchSize * (poolingKernelSize ?? 1) }

    /// Side length of the pooled output grid for a canvas of `imageSize` pixels.
    public func pooledGrid(imageSize: Int) -> Int { imageSize / blockPixels }

    /// Soft tokens a canvas of `imageSize` pixels always produces. The real (non-padding) subset
    /// is generally fewer; see the pooled-grid math in `CoreAIStaticShapeEngine+Multimodal.swift`.
    public func softTokenCount(imageSize: Int) -> Int {
        let grid = pooledGrid(imageSize: imageSize)
        return grid * grid
    }

    /// Name of the `vision.aimodel` function for a canvas size, as the exporter names it
    /// (`export/gemma4_vision.py`: `encode_image_<size>`). The bundle also lists the names under
    /// `language.function_map.vision`.
    public static func encoderFunctionName(imageSize: Int) -> String { "encode_image_\(imageSize)" }

    /// The geometry the vision path divides and indexes by. A size ladder must be sorted
    /// ascending and unique (canvas selection takes the first that fits) and every size a multiple
    /// of `patchSize * poolingKernelSize` (the letterbox pads in whole pool blocks).
    private func validate(in c: KeyedDecodingContainer<CodingKeys>) throws {
        func fail(_ key: CodingKeys, _ message: String) -> DecodingError {
            DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: message)
        }
        guard imageSize > 0 else { throw fail(.imageSize, "image_size must be positive, got \(imageSize)") }
        guard patchSize > 0 else { throw fail(.patchSize, "patch_size must be positive, got \(patchSize)") }
        if let poolingKernelSize, poolingKernelSize <= 0 {
            throw fail(.poolingKernelSize, "pooling_kernel_size must be positive, got \(poolingKernelSize)")
        }
        guard let imageSizes else { return }
        guard !imageSizes.isEmpty else { throw fail(.imageSizes, "image_sizes must not be empty") }
        var previous = 0
        for size in imageSizes {
            guard size > previous else {
                throw fail(
                    .imageSizes,
                    "image_sizes must be positive, ascending and unique (\(size) follows \(previous))")
            }
            previous = size
            guard size % blockPixels == 0 else {
                throw fail(
                    .imageSizes,
                    "image_sizes entry \(size) is not a multiple of patch_size * pooling_kernel_size "
                        + "(\(blockPixels))")
            }
        }
    }
}

/// How a vision encoder's soft tokens are merged into the decoder. The exporter records the choice in
/// `vision.token_merge`, and ``EngineFactory`` picks the engine from it.
public enum VisionTokenMerge: String, Codable, Sendable {
    /// Text embeddings come from a separate `embedding` asset. The soft tokens are scattered into
    /// them before the decoder, which consumes embeddings (Qwen-style, `CoreAISequentialVLMEngine`).
    case separateEmbedding = "separate_embedding"
    /// The decoder graph embeds its own tokens. The soft tokens overwrite the placeholder rows
    /// after that gather (Gemma 4, `StaticShapeEngine`).
    case decoderGather = "decoder_gather"
}
