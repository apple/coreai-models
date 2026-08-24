// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
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

        var inferSigmas: [Float]

        if let mu {
            // Dynamic shift (Flux/Klein): linspace in raw sigma space [1.0, 1/stepCount],
            // then apply exponential shift. This path uses 1/stepCount as floor because
            // the Flux pipeline passes sigmas directly to set_timesteps.
            let sigmaMin: Float = 1.0 / Float(stepCount)
            inferSigmas = linspace(sigmaMax, sigmaMin, stepCount)
            let expMu = expf(mu)
            inferSigmas = inferSigmas.map { sigma in
                expMu / (expMu + (1.0 / sigma - 1.0))
            }
        } else if timeStepShift != 1.0 {
            // Static shift (Wan, SD3): match diffusers FlowMatchEulerDiscreteScheduler.
            // Algorithm: linspace in timestep space from sigma_max*T to sigma_min*T
            // (where sigma_min = shift(1/T)), then divide by T, then apply shift.
            let ts = Float(trainStepCount)
            let rawSigmaMin: Float = 1.0 / ts
            let shiftedSigmaMin = timeStepShift * rawSigmaMin / (1.0 + (timeStepShift - 1.0) * rawSigmaMin)
            let tMax = sigmaMax * ts
            let tMin = shiftedSigmaMin * ts
            let timestepLinspace = linspace(tMax, tMin, stepCount)
            let rawSigmas = timestepLinspace.map { $0 / ts }
            inferSigmas = rawSigmas.map { sigma in
                timeStepShift * sigma / (1.0 + (timeStepShift - 1.0) * sigma)
            }
        } else {
            // No shift: uniform linspace
            inferSigmas = linspace(sigmaMax, 1.0 / Float(trainStepCount), stepCount)
        }

        let ts = Float(trainStepCount)
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
        let stepIndex = counter
        precondition(stepIndex < sigmas.count, "step() called beyond inferenceStepCount")
        let sigma = sigmas[stepIndex]

        var dt = sigma
        if stepIndex < sigmas.count - 1 {
            dt = sigmas[stepIndex + 1] - sigma
        }

        // prevSample = sample + output * dt (Euler step, simplified from the full derivation)
        let count = vDSP_Length(output.count)
        var prevSample = [Float](repeating: 0, count: Int(count))
        var dtVal = dt
        vDSP_vsma(output, 1, &dtVal, sample, 1, &prevSample, 1, count)

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
