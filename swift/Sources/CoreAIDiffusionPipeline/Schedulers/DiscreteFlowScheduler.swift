// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Discrete flow matching scheduler for SD3 and Flux models.
/// Uses Euler method on a flow-matching ODE (sigma interpolation between noise and data).
public final class DiscreteFlowScheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let timeSteps: [Int]
    /// The first scheduled sigma after all shifts are applied — use this for img2img noise addition.
    public var startSigma: Float { sigmas.first ?? 1.0 }

    let trainSteps: Float
    let shift: Float
    let mu: Float?
    var counter: Int
    let sigmas: [Float]

    public private(set) var modelOutputs: [[Float]] = []

    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        timeStepShift: Float = 3.0,
        mu: Float? = nil,
        sigmaMax: Float = 1.0
    ) {
        precondition(trainStepCount > 0 && stepCount > 0)
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.trainSteps = Float(trainStepCount)
        self.shift = timeStepShift
        self.mu = mu
        self.counter = 0

        // Lower bound of the pre-shift sigma linspace. Diffusers builds
        //   sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        // (pipeline_flux2_klein.py) and passes it to
        // FlowMatchEulerDiscreteScheduler.set_timesteps, which uses the provided
        // sigmas as-is and only applies the mu/shift transform — it does NOT
        // recompute the endpoints from num_train_timesteps. So the floor must be
        // 1/stepCount for BOTH the dynamic-shift (mu) and static-shift paths.
        // Using 1/trainStepCount here collapsed the final sigma to ~0 at low step
        // counts (e.g. 4-step Klein), wasting the last step and misplacing all
        // intermediate noise levels.
        let sigmaMin: Float = 1.0 / Float(stepCount)
        var inferSigmas: [Float] = linspace(sigmaMax, sigmaMin, stepCount)

        if let mu {
            let expMu = expf(mu)
            inferSigmas = inferSigmas.map { sigma in
                expMu / (expMu + (1.0 / sigma - 1.0))
            }
        } else if timeStepShift != 1.0 {
            inferSigmas = inferSigmas.map { sigma in
                timeStepShift * sigma / (1.0 + (timeStepShift - 1.0) * sigma)
            }
        }

        let ts = trainSteps
        self.sigmas = inferSigmas + [0.0]
        self.timeSteps = inferSigmas.map { Int($0 * ts) }
    }

    static func sigmaFromTimestep(_ timestep: Float, trainSteps: Float, shift: Float) -> Float {
        if shift == 1.0 {
            return timestep / trainSteps
        } else {
            let t = timestep / trainSteps
            return shift * t / (1 + (shift - 1) * t)
        }
    }

    /// Exponential dynamic shift: sigma' = exp(mu) / (exp(mu) + (1/sigma - 1))
    static func applyDynamicShift(_ sigma: Float, mu: Float) -> Float {
        let expMu = exp(mu)
        return expMu / (expMu + (1.0 / sigma - 1.0))
    }

    public func step(output: [Float], timeStep t: Int, sample: [Float]) -> [Float] {
        let stepIndex = timeSteps.firstIndex(of: t) ?? counter
        precondition(stepIndex < sigmas.count, "step() called with invalid timeStep or beyond inferenceStepCount")
        let sigma = sigmas[stepIndex]

        let count = output.count
        var denoised = [Float](repeating: 0, count: count)
        for i in 0..<count {
            denoised[i] = sample[i] - output[i] * sigma
        }
        modelOutputs.append(denoised)

        var dt = sigma
        var prevSigma: Float = 0
        if stepIndex < sigmas.count - 1 {
            prevSigma = sigmas[stepIndex + 1]
            dt = prevSigma - sigma
        }

        var prevSample = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let d = (sample[i] - denoised[i]) / sigma
            prevSample[i] = sample[i] + d * dt
        }

        counter += 1
        return prevSample
    }

    public func calculateTimesteps(strength: Float?) -> [Int] {
        guard let strength else { return timeSteps }
        let startStep = max(inferenceStepCount - Int(Float(inferenceStepCount) * strength), 0)
        return Array(timeSteps[startStep...])
    }

    /// Flow-matching forward noising: x_t = (1 − t)·x_0 + t·ε where t = strength (starting sigma).
    public func addNoise(to sample: [Float], noise: [Float], at strength: Float) -> [Float] {
        zip(sample, noise).map { (1 - strength) * $0 + strength * $1 }
    }
}
