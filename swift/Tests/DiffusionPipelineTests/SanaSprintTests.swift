// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIDiffusionPipeline

@Suite("Sana Sprint")
struct SanaSprintTests {
    @Test("Sequential torch draws match torch.randn with a shared generator (seed=42)")
    func torchStreamContinuesAcrossDraws() {
        // g = torch.Generator("cpu").manual_seed(42); torch.randn(1,32,32,32, generator=g) twice
        var rng = TorchRandomSource(seed: 42)
        let first = rng.normalArray([32 * 32 * 32])
        let second = rng.normalArray([32 * 32 * 32])
        let expectedFirst: [Float] = [1.9269150, 1.4872842, 0.9007172, -2.1055214]
        let expectedSecond: [Float] = [-0.7837821, -0.3568874, -1.3867873, 1.4456867]
        for (a, b) in zip(first.prefix(4), expectedFirst) { #expect(abs(a - b) < 1e-5) }
        for (a, b) in zip(second.prefix(4), expectedSecond) { #expect(abs(a - b) < 1e-5) }
        #expect(abs(first[first.count - 1] - (-0.2654118)) < 1e-5)
        #expect(abs(second[second.count - 1] - 1.6214950) < 1e-5)
    }

    @Test("TrigFlow angles map to diffusers' scm_timestep values")
    func sigmasMatchDiffusers() {
        // diffusers: sin(t) / (sin(t) + cos(t)) for t in [1.5708, 1.3]
        let two = FlowTransformerPipeline.sanaSprintSigmas(
            steps: 2, maxTimestep: 1.5708, intermediateTimestep: 1.3)
        #expect(two.count == 2)
        #expect(abs(two[0] - 1.0000037) < 1e-6)
        #expect(abs(two[1] - 0.7827080) < 1e-6)

        // Other step counts use linspace(max, 0, steps + 1) and ignore the intermediate angle.
        let four = FlowTransformerPipeline.sanaSprintSigmas(
            steps: 4, maxTimestep: 1.5708, intermediateTimestep: 1.3)
        let angles: [Float] = [1.5708, 1.1781, 0.7854, 0.3927]
        #expect(four.count == 4)
        for (s, t) in zip(four, angles) {
            #expect(abs(s - sin(t) / (sin(t) + cos(t))) < 1e-4)
        }
        #expect(abs(four[2] - 0.5) < 1e-4)  // t = π/4 is the midpoint
    }

    @Test("Explicit sigmas append a terminal zero and drive currentSigma")
    func explicitSigmaSchedule() {
        let scheduler = DiscreteFlowScheduler(sigmas: [1.0, 0.75])
        #expect(scheduler.inferenceStepCount == 2)
        #expect(scheduler.currentSigma == 1.0)
        _ = scheduler.stepStochastic(output: [0], sample: [0], noise: [0])
        #expect(scheduler.currentSigma == 0.75)
        _ = scheduler.stepStochastic(output: [0], sample: [0], noise: [0])
        #expect(scheduler.currentSigma == 0)
    }

    @Test("Stochastic step predicts x0 then renoises to the next sigma")
    func stochasticStepMath() {
        let scheduler = DiscreteFlowScheduler(sigmas: [0.8, 0.5])
        // x0 = x − 0.8·v; next = 0.5·x0 + 0.5·noise
        let next = scheduler.stepStochastic(output: [1, -2], sample: [2, 0], noise: [4, 1])
        #expect(abs(next[0] - (0.5 * 1.2 + 0.5 * 4)) < 1e-6)
        #expect(abs(next[1] - (0.5 * 1.6 + 0.5 * 1)) < 1e-6)
        // The final step lands on x0 exactly: the noise weight is the terminal sigma, 0.
        let last = scheduler.stepStochastic(output: [1, 1], sample: [1, 1], noise: [100, 100])
        #expect(abs(last[0] - 0.5) < 1e-6)
        #expect(abs(last[1] - 0.5) < 1e-6)
    }

    @Test("Mask selection keeps BOS plus the trailing positions")
    func maskSelection() {
        // 6 real tokens then padding, traced length 10, output length 4
        let mask: [Float] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        #expect(FlowTransformerPipeline.sanaSelectMask(mask, outputLength: 4) == [1, 0, 0, 0])
        let full: [Float] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        #expect(FlowTransformerPipeline.sanaSelectMask(full, outputLength: 4) == [1, 1, 1, 0])
    }

    @Test("Decodes the sana-sprint metadata.json")
    func decodesMetadata() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sana_\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let json = """
            {
                "metadata_version": "0.2",
                "kind": "diffusion",
                "assets": {
                    "transformer": "Transformer.aimodel",
                    "text_encoder": "TextEncoder.aimodel",
                    "vae_decoder": "VAEDecoder.aimodel"
                },
                "diffusion": {
                    "type": "sana-sprint",
                    "prediction_type": "flow_matching",
                    "decoder_scale_factor": 0.41407,
                    "image_size": 1024,
                    "default_guidance_scale": 4.5,
                    "default_steps": 2,
                    "max_timesteps": 1.5708,
                    "intermediate_timesteps": 1.3,
                    "prompt_prefix": "Prefix:\\nUser Prompt: ",
                    "text_input_length": 506,
                    "text_sequence_length": 300
                }
            }
            """
        try json.write(to: dir.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)

        let descriptor = try PipelineDescriptor.resolve(at: dir)
        #expect(descriptor.type == .sanaSprint)
        #expect(descriptor.predictionType == .flowMatching)
        #expect(descriptor.decoderScaleFactor == 0.41407)
        #expect(descriptor.defaultSteps == 2)
        #expect(descriptor.maxTimesteps == 1.5708)
        #expect(descriptor.intermediateTimesteps == 1.3)
        #expect(descriptor.promptPrefix == "Prefix:\nUser Prompt: ")
        #expect(descriptor.textInputLength == 506)
        #expect(descriptor.textSequenceLength == 300)
        #expect(descriptor.components.unet == "Transformer.aimodel")
        #expect(descriptor.components.textEncoder == "TextEncoder.aimodel")
        #expect(descriptor.components.vaeDecoder == "VAEDecoder.aimodel")
    }
}
