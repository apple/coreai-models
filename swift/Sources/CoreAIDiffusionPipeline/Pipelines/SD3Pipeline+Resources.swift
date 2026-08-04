// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

extension SD3Pipeline {
    /// Load an SD 3.x pipeline from a directory containing .aimodel files,
    /// tokenizer/, tokenizer_2/, and pipeline.json.
    public init(
        from url: URL,
        config: PipelineDescriptor.ConfigSource = .auto
    ) async throws {
        let descriptor = try PipelineDescriptor.resolve(at: url, config: config)

        guard let transformerPath = descriptor.components.unet else {
            throw PipelineLoadError.missingComponent("transformer")
        }
        guard let textEncoderPath = descriptor.components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }
        guard let textEncoder2Path = descriptor.components.textEncoder2 else {
            throw PipelineLoadError.missingComponent("text_encoder_2")
        }
        guard let decoderPath = descriptor.components.vaeDecoder else {
            throw PipelineLoadError.missingComponent("vae_decoder")
        }

        let transformer = CoreAIDiffusionModelFunction(
            modelURL: ModelBundle.resolveAssetURL(transformerPath, in: url))
        let textEncoder = CoreAIDiffusionModelFunction(
            modelURL: ModelBundle.resolveAssetURL(textEncoderPath, in: url))
        let textEncoder2 = CoreAIDiffusionModelFunction(
            modelURL: ModelBundle.resolveAssetURL(textEncoder2Path, in: url))
        let decoder = CoreAIDiffusionModelFunction(
            modelURL: ModelBundle.resolveAssetURL(decoderPath, in: url))

        let tokenizer = try Self.loadBPETokenizer(
            at: url.appendingPathComponent("tokenizer"))
        let tokenizer2 = try Self.loadBPETokenizer(
            at: url.appendingPathComponent("tokenizer_2"))

        self.init(
            descriptor: descriptor,
            transformer: transformer,
            textEncoder: textEncoder,
            textEncoder2: textEncoder2,
            decoder: decoder,
            tokenizer: tokenizer,
            tokenizer2: tokenizer2)
    }

    private static func loadBPETokenizer(at dir: URL) throws -> BPETokenizer {
        let mergesURL = dir.appendingPathComponent("merges.txt")
        let vocabURL = dir.appendingPathComponent("vocab.json")
        return try BPETokenizer(mergesAt: mergesURL, vocabularyAt: vocabURL)
    }
}
