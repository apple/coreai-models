// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

@Suite("PipelineDescriptor")
struct PipelineDescriptorTests {
    @Test("Loads from pipeline.json with all fields")
    func loadFromJSON() throws {
        let json = """
            {
                "version": "1.5",
                "prediction_type": "epsilon",
                "image_size": 512,
                "components": {
                    "text_encoder": "TextEncoder.aimodel",
                    "transformer": "Transformer.aimodel",
                    "vae_decoder": "VAEDecoder.aimodel",
                    "vae_encoder": "VAEEncoder.aimodel"
                },
                "scheduler": {
                    "training_steps": 1000,
                    "beta_start": 0.00085,
                    "beta_end": 0.012,
                    "beta_schedule": "scaled_linear"
                },
                "decoder_scale_factor": 0.18215
            }
            """

        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("pipeline_\(UUID()).json")
        try json.write(to: tmp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let descriptor = try PipelineDescriptor.load(from: tmp)
        #expect(descriptor.type == nil)
        #expect(descriptor.version == "1.5")
        #expect(descriptor.predictionType == .epsilon)
        #expect(descriptor.imageSize == 512)
        #expect(descriptor.components.textEncoder == "TextEncoder.aimodel")
        #expect(descriptor.components.transformer == "Transformer.aimodel")
        #expect(descriptor.components.vaeDecoder == "VAEDecoder.aimodel")
        #expect(descriptor.components.vaeEncoder == "VAEEncoder.aimodel")
        #expect(descriptor.decoderScaleFactor == 0.18215)
        #expect(descriptor.scheduler?.trainingSteps == 1000)
    }

    @Test("Minimal pipeline.json — only components required")
    func loadMinimal() throws {
        let json = """
            {
                "components": {
                    "transformer": "Transformer.aimodel",
                    "vae_decoder": "VAEDecoder.aimodel"
                }
            }
            """

        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("pipeline_min_\(UUID()).json")
        try json.write(to: tmp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let descriptor = try PipelineDescriptor.load(from: tmp)
        #expect(descriptor.type == nil)
        #expect(descriptor.version == nil)
        #expect(descriptor.predictionType == nil)
        #expect(descriptor.imageSize == nil)
        #expect(descriptor.components.transformer == "Transformer.aimodel")
        #expect(descriptor.components.vaeDecoder == "VAEDecoder.aimodel")
    }

    @Test("Auto-detects components from directory")
    func detectFromDirectory() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("detect_model_\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "".write(to: dir.appendingPathComponent("TextEncoder.aimodel"), atomically: true, encoding: .utf8)
        try "".write(to: dir.appendingPathComponent("Transformer.aimodel"), atomically: true, encoding: .utf8)
        try "".write(to: dir.appendingPathComponent("VAEDecoder.aimodel"), atomically: true, encoding: .utf8)

        let descriptor = PipelineDescriptor.detect(at: dir)
        #expect(descriptor.components.textEncoder == "TextEncoder.aimodel")
        #expect(descriptor.components.transformer == "Transformer.aimodel")
        #expect(descriptor.components.vaeDecoder == "VAEDecoder.aimodel")
        #expect(descriptor.components.vaeEncoder == nil)
        #expect(descriptor.type == nil)
        #expect(descriptor.imageSize == nil)
    }

    @Test("Auto-detects snake_case component names")
    func detectSnakeCase() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("detect_snake_\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "".write(to: dir.appendingPathComponent("text_encoder.aimodel"), atomically: true, encoding: .utf8)
        try "".write(to: dir.appendingPathComponent("transformer.aimodel"), atomically: true, encoding: .utf8)
        try "".write(to: dir.appendingPathComponent("vae_decoder.aimodel"), atomically: true, encoding: .utf8)
        try "".write(to: dir.appendingPathComponent("vae_encoder.aimodel"), atomically: true, encoding: .utf8)

        let descriptor = PipelineDescriptor.detect(at: dir)
        #expect(descriptor.components.textEncoder == "text_encoder.aimodel")
        #expect(descriptor.components.transformer == "transformer.aimodel")
        #expect(descriptor.components.vaeDecoder == "vae_decoder.aimodel")
        #expect(descriptor.components.vaeEncoder == "vae_encoder.aimodel")
    }

    @Test("Detects transformer component")
    func detectTransformer() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("transformer_\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "".write(to: dir.appendingPathComponent("Transformer.aimodel"), atomically: true, encoding: .utf8)

        let descriptor = PipelineDescriptor.detect(at: dir)
        #expect(descriptor.components.transformer == "Transformer.aimodel")
    }

    @Test("Resolve errors on legacy pipeline.json")
    func resolveRejectsLegacyPipelineJSON() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("resolve_legacy_\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let json = """
            {
                "version": "2.0",
                "image_size": 768,
                "components": {
                    "text_encoder": "custom_encoder.aimodel",
                    "unet": "custom_unet.aimodel",
                    "vae_decoder": "custom_decoder.aimodel"
                }
            }
            """
        try json.write(to: dir.appendingPathComponent("pipeline.json"), atomically: true, encoding: .utf8)

        #expect(throws: PipelineLoadError.self) {
            _ = try PipelineDescriptor.resolve(at: dir)
        }
    }

    @Test("Resolve with explicit config ignores directory")
    func resolveExplicit() throws {
        let explicit = PipelineDescriptor(
            version: "custom",
            imageSize: 256,
            components: .init(textEncoder: "my.aimodel")
        )

        let dir = FileManager.default.temporaryDirectory
        let descriptor = try PipelineDescriptor.resolve(at: dir, config: .explicit(explicit))
        #expect(descriptor.version == "custom")
        #expect(descriptor.imageSize == 256)
    }

    @Test("Default init has nil metadata, empty components")
    func defaults() {
        let descriptor = PipelineDescriptor()
        #expect(descriptor.type == nil)
        #expect(descriptor.version == nil)
        #expect(descriptor.predictionType == nil)
        #expect(descriptor.imageSize == nil)
        #expect(descriptor.decoderScaleFactor == nil)
        #expect(descriptor.components.transformer == nil)
    }

    @Test("Encodes to JSON with snake_case keys")
    func encodeToJSON() throws {
        let descriptor = PipelineDescriptor(
            type: .flux2,
            predictionType: .epsilon,
            components: .init(textEncoder: "TE.aimodel", transformer: "U.aimodel", vaeDecoder: "D.aimodel"),
            decoderScaleFactor: 0.18215
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(descriptor)
        let json = String(data: data, encoding: .utf8)!
        #expect(json.contains("text_encoder"))
        #expect(json.contains("vae_decoder"))
        #expect(json.contains("prediction_type"))
        #expect(json.contains("decoder_scale_factor"))
    }
}
