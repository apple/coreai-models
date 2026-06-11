// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// VLM-specific configuration block from the `"vision"` key in metadata.json.
///
/// Decoded from bundles with `kind == .vlm`. Provides vision encoder parameters
/// and the path to the embed_tokens table used for embedding fusion.
public struct VLMConfig: Codable, Sendable, Equatable {
    /// Token ID used for image patch placeholders (e.g. `<|image_pad|>` = 151655 in Qwen3-VL).
    public let imageTokenId: Int
    /// Number of visual tokens the vision encoder produces (e.g. 196 for 448×448 / merge_size=2).
    public let numVisualTokens: Int
    /// Hidden dimension of the LLM decoder (must match `inputs_embeds` second dim).
    public let hiddenSize: Int
    /// Bundle-relative path to the float16 embedding table (vocab_size × hidden_size).
    public let embedTokensPath: String

    enum CodingKeys: String, CodingKey {
        case imageTokenId = "image_token_id"
        case numVisualTokens = "num_visual_tokens"
        case hiddenSize = "hidden_size"
        case embedTokensPath = "embed_tokens_path"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 151655
        numVisualTokens = try c.decodeIfPresent(Int.self, forKey: .numVisualTokens) ?? 196
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        embedTokensPath = try c.decode(String.self, forKey: .embedTokensPath)
    }
}

// MARK: - ModelBundle extension

extension ModelBundle {
    /// Decoded `vision` block for VLM bundles. Returns nil for LLM/diffusion bundles
    /// or if the vision block is absent or malformed.
    public var vlm: VLMConfig? {
        guard kind == .vlm else { return nil }
        struct Wrapper: Decodable { let vision: VLMConfig? }
        return (try? JSONDecoder().decode(Wrapper.self, from: raw))?.vision
    }
}
