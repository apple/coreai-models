// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - VLM Model Config

/// Configuration for a Vision-Language Model engine.
///
/// Extends the base `ModelConfig` with vision-specific parameters: image size,
/// patch geometry, placeholder token ID, and per-image embedding token count.
public struct VLMModelConfig: InferenceConfiguration, Codable, Sendable {
    public let base: ModelConfig
    public let visionConfig: VisionConfig

    public var maxContextLength: Int { base.maxContextLength }
    public var vocabSize: Int { base.vocabSize }
    public var function: String { base.function }
    public var name: String { base.name }

    /// Prefill chunk size from base config.
    public var prefillChunkSize: Int { base.prefillChunkSize }
    public var chunkThreshold: Int { base.chunkThreshold }

    public init(base: ModelConfig, visionConfig: VisionConfig) {
        self.base = base
        self.visionConfig = visionConfig
    }
}
