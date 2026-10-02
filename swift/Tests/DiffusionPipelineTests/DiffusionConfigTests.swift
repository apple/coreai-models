// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

@Suite("DiffusionConfig")
struct DiffusionConfigTests {
    private func decode(_ json: String) throws -> DiffusionConfig {
        let decoder = JSONDecoder()
        return try decoder.decode(DiffusionConfig.self, from: Data(json.utf8))
    }

    @Test("Default init leaves every field nil")
    func defaults() {
        let config = DiffusionConfig()
        #expect(config.type == nil)
        #expect(config.predictionType == nil)
        #expect(config.imageSize == nil)
        #expect(config.decoderScaleFactor == nil)
        #expect(config.textDim == nil)
    }

    @Test("Decodes a FLUX.2 diffusion block")
    func decodesFlux2() throws {
        let config = try decode(
            """
            {
                "type": "flux2",
                "prediction_type": "flow_matching",
                "image_size": 1024,
                "encoder_scale_factor": 0.3611,
                "decoder_scale_factor": 0.3611,
                "decoder_shift_factor": 0.1159,
                "batch_norm_eps": 1e-5,
                "rope_axes_dims": [32, 32, 32, 32],
                "default_guidance_scale": 1.0,
                "default_steps": 4
            }
            """)
        #expect(config.type == .flux2)
        #expect(config.predictionType == .flowMatching)
        #expect(config.imageSize == 1024)
        #expect(config.encoderScaleFactor == 0.3611)
        #expect(config.decoderShiftFactor == 0.1159)
        #expect(config.ropeAxesDims == [32, 32, 32, 32])
        #expect(config.defaultSteps == 4)
    }

    @Test("Decodes a Wan video block; unknown type decodes to nil")
    func decodesWan() throws {
        let config = try decode(
            """
            {
                "type": "wan2.1",
                "prediction_type": "flow_matching",
                "text_dim": 4096,
                "z_dim": 16,
                "default_steps": 50,
                "default_guidance_scale": 5.0,
                "default_shift": 3.0,
                "default_num_frames": 81
            }
            """)
        // "wan2.1" is not an image PipelineType — lenient decode yields nil rather than throwing.
        #expect(config.type == nil)
        #expect(config.textDim == 4096)
        #expect(config.latentChannels == 16)
        #expect(config.defaultSteps == 50)
        #expect(config.schedulerShift == 3.0)
        #expect(config.defaultNumFrames == 81)
    }

    @Test("Unrecognized enum raw values decode to nil rather than throwing")
    func lenientEnums() throws {
        let config = try decode(
            """
            { "type": "sdxl", "prediction_type": "made_up" }
            """)
        #expect(config.type == nil)
        #expect(config.predictionType == nil)
    }

    @Test("Round-trips through JSON with snake_case keys")
    func encodeRoundTrip() throws {
        let config = DiffusionConfig(
            type: .sanaSprint,
            predictionType: .flowMatching,
            imageSize: 1024,
            decoderScaleFactor: 0.41407,
            defaultSteps: 2,
            promptPrefix: "Prefix: ")
        let data = try JSONEncoder().encode(config)
        let json = String(data: data, encoding: .utf8)!
        #expect(json.contains("prediction_type"))
        #expect(json.contains("decoder_scale_factor"))
        #expect(json.contains("image_size"))
        #expect(json.contains("prompt_prefix"))

        let decoded = try JSONDecoder().decode(DiffusionConfig.self, from: data)
        #expect(decoded == config)
    }
}
