// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// The `text` block (R6): token ids, truncation and the default prompt.
struct TextConfig: Sendable {
    /// The tokenizer adds BOS; the host does not.
    var addBosToken: Bool
    /// The tokenizer adds EOS; the host does not.
    var addEosToken: Bool
    /// `nil` when the tokenizer adds no BOS (Qwen3-Embedding).
    var bosTokenId: Int32?
    var eosTokenId: Int32
    var padTokenId: Int32
    /// Longest token sequence, special tokens included.
    var maxTokens: Int
    /// Truncation keeps EOS as the last token.
    var keepEos: Bool
    /// Prefix of every text: `prompts[default_prompt_name]`, or empty.
    var prompt: String
}

/// The `image` block (R7).
struct ImageConfig: Sendable {
    enum Strategy: String, Sendable, Decodable {
        /// Resize the shortest edge to `size`, crop the center square (CLIP).
        case centerCrop = "center_crop"
        /// The largest size with sides divisible by `patchSize * poolingKernelSize` and at most
        /// `maxSoftTokens * poolingKernelSize²` patches, cut into patches (EmbeddingGemma 2).
        case aspectRatioPreserving = "aspect_ratio_preserving"
    }

    /// Patch geometry and image token ids of `aspectRatioPreserving`.
    struct Patches: Sendable, Decodable {
        var patchSize: Int
        var poolingKernelSize: Int
        var maxSoftTokens: Int
        var imageTokenId: Int32
        var boiTokenId: Int32
        var eoiTokenId: Int32

        /// Patches per image after padding: `maxSoftTokens * poolingKernelSize²`.
        var maxPatches: Int { maxSoftTokens * poolingKernelSize * poolingKernelSize }
        /// Values per patch: `3 * patchSize²`.
        var patchDimension: Int { 3 * patchSize * patchSize }

        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
            case maxSoftTokens = "max_soft_tokens"
            case imageTokenId = "image_token_id"
            case boiTokenId = "boi_token_id"
            case eoiTokenId = "eoi_token_id"
        }
    }

    var strategy: Strategy
    var mean: [Float]
    var std: [Float]
    /// The crop side of `centerCrop`, required there.
    var size: Int?
    /// The patch geometry of `aspectRatioPreserving`, required there.
    var patches: Patches?
}

// MARK: - Decoding

extension TextConfig: Decodable {
    enum CodingKeys: String, CodingKey {
        case truncation, prompts
        case addBosToken = "add_bos_token"
        case addEosToken = "add_eos_token"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case defaultPromptName = "default_prompt_name"
        case maxTokens = "max_tokens"
        case keepEos = "keep_eos"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        addBosToken = try c.decode(Bool.self, forKey: .addBosToken)
        addEosToken = try c.decode(Bool.self, forKey: .addEosToken)
        bosTokenId = try c.decodeIfPresent(Int32.self, forKey: .bosTokenId)
        eosTokenId = try c.decode(Int32.self, forKey: .eosTokenId)
        padTokenId = try c.decode(Int32.self, forKey: .padTokenId)
        let truncation = try c.nestedContainer(keyedBy: CodingKeys.self, forKey: .truncation)
        maxTokens = try truncation.decode(Int.self, forKey: .maxTokens)
        guard maxTokens >= 1 else {
            throw EmbeddingError.invalidMetadata("text.truncation.max_tokens must be at least 1")
        }
        keepEos = try truncation.decode(Bool.self, forKey: .keepEos)
        let prompts = try c.decodeIfPresent([String: String].self, forKey: .prompts) ?? [:]
        if let name = try c.decodeIfPresent(String.self, forKey: .defaultPromptName) {
            guard let prompt = prompts[name] else {
                throw EmbeddingError.invalidMetadata("default_prompt_name '\(name)' is not in prompts")
            }
            self.prompt = prompt
        } else {
            prompt = ""
        }
    }
}

extension ImageConfig: Decodable {
    enum CodingKeys: String, CodingKey {
        case imageStrategy = "image_strategy"
        case imageSize = "image_size"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        strategy = try c.decode(Strategy.self, forKey: .imageStrategy)
        mean = try c.decode([Float].self, forKey: .imageMean)
        std = try c.decode([Float].self, forKey: .imageStd)
        guard mean.count == 3, std.count == 3 else {
            throw EmbeddingError.invalidMetadata("image_mean and image_std must hold 3 values")
        }
        size = strategy == .centerCrop ? try c.decode(Int.self, forKey: .imageSize) : nil
        patches = strategy == .aspectRatioPreserving ? try Patches(from: decoder) : nil
    }
}
