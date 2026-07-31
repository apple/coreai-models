// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation

/// Core AI denoiser — wraps a UNet or DiT/MMDiT model function.
public final class CoreAIDenoiser: Sendable {
    public let function: CoreAIDiffusionModelFunction

    public init(function: CoreAIDiffusionModelFunction) {
        self.function = function
    }

    public func loadResources() async throws {
        try await function.loadResources()
    }

    public func unloadResources() async {
        await function.unloadResources()
    }

    public func predictNoise(
        latents: NDArray,
        timestep: Float,
        textEmbeddings: NDArray,
        additionalInputs: [String: NDArray]
    ) async throws -> NDArray {
        let batchSize = latents.shape[0]

        var timestepArray = NDArray(shape: [batchSize], scalarType: .float32)
        let timestepView = timestepArray.mutableView(as: Float.self)
        timestepView.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<batchSize { ptr[i] = timestep }
        }

        var inputs: [String: NDArray] = [
            "sample": latents,
            "timestep": timestepArray,
            "encoder_hidden_states": textEmbeddings,
        ]
        for (key, value) in additionalInputs { inputs[key] = value }

        let outputs = try await function.predict(inputs: inputs)

        let outputDescs = try await function.outputDescriptors
        let outputName = outputDescs.keys.first
        let floats: [Float]
        if let name = outputName, let named = outputs[name] {
            floats = named
        } else if outputs.count == 1, let only = outputs.values.first {
            floats = only
        } else {
            throw CoreAIComponentError.missingOutput("noise_pred", "Denoiser")
        }

        let shape = latents.shape
        var result = NDArray(shape: shape, scalarType: .float32)
        let resultView = result.mutableView(as: Float.self)
        resultView.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<floats.count { ptr[i] = floats[i] }
        }
        return result
    }
}
