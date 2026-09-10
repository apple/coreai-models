// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// How to space timesteps for inference
public enum TimeStepSpacing {
    case linspace
    case leading
    case karras
}

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers DPMSolverMultistepScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)
///
/// It uses the DPM-Solver++ algorithm: [code](https://github.com/LuChengTHU/dpm-solver) [paper](https://arxiv.org/abs/2211.01095).
/// Limitations:
///  - Only implemented for DPM-Solver++ algorithm (not DPM-Solver).
///  - Second order only.
///  - No dynamic thresholding.
///  - `midpoint` solver algorithm.

public final class DPMSolverMultistepScheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Int]

    public let alphaT: [Float]
    public let sigmaT: [Float]
    public let lambdaT: [Float]

    public let predictionType: PredictionType
    public let solverOrder = 2
    private(set) var lowerOrderStepped = 0

    private var usingKarrasSigmas = false

    /// Whether to use lower-order solvers in the final steps. Only valid for less than 15 inference steps.
    /// We empirically find this trick can stabilize the sampling of DPM-Solver, especially with 10 or fewer steps.
    public let useLowerOrderFinal = true

    // Stores solverOrder (2) items
    public private(set) var modelOutputs: [[Float]] = []

    /// Create a scheduler that uses a second order DPM-Solver++ algorithm.
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - trainStepCount: Number of training diffusion steps
    ///   - betaSchedule: Method to schedule betas from betaStart to betaEnd
    ///   - betaStart: The starting value of beta for inference
    ///   - betaEnd: The end value for beta for inference
    ///   - timeStepSpacing: How to space time steps
    /// - Returns: A scheduler ready for its first step
    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        timeStepSpacing: TimeStepSpacing = .linspace,
        predictionType: PredictionType = .epsilon
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.predictionType = predictionType

        switch betaSchedule {
        case .linear:
            self.betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            self.betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map({ $0 * $0 })
        }

        self.alphas = betas.map({ 1.0 - $0 })
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i - 1]
        }
        self.alphasCumProd = alphasCumProd

        switch timeStepSpacing {
        case .linspace:
            self.timeSteps = linspace(0, Float(self.trainStepCount - 1), stepCount + 1).dropFirst().reversed().map {
                Int(round($0))
            }
            self.alphaT = vForce.sqrt(self.alphasCumProd)
            self.sigmaT = vForce.sqrt(
                vDSP.subtract([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd))
        case .leading:
            let lastTimeStep = trainStepCount - 1
            let stepRatio = lastTimeStep / (stepCount + 1)
            // Creates integer timesteps by multiplying by ratio
            self.timeSteps = (0...stepCount).map { 1 + $0 * stepRatio }.dropFirst().reversed()
            self.alphaT = vForce.sqrt(self.alphasCumProd)
            self.sigmaT = vForce.sqrt(
                vDSP.subtract([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd))
        case .karras:
            // sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            let scaled = vDSP.multiply(
                subtraction: ([Float](repeating: 1, count: self.alphasCumProd.count), self.alphasCumProd),
                subtraction: (
                    vDSP.divide(1, self.alphasCumProd), [Float](repeating: 0, count: self.alphasCumProd.count)
                )
            )
            let sigmas = vForce.sqrt(scaled)
            let logSigmas = sigmas.map { log($0) }

            let sigmaMin = sigmas.first!
            let sigmaMax = sigmas.last!
            let rho: Float = 7
            let ramp = linspace(0, 1, stepCount)
            let minInvRho = pow(sigmaMin, (1 / rho))
            let maxInvRho = pow(sigmaMax, (1 / rho))

            var karrasSigmas = ramp.map { pow(maxInvRho + $0 * (minInvRho - maxInvRho), rho) }
            let karrasTimeSteps = karrasSigmas.map { sigmaToTimestep(sigma: $0, logSigmas: logSigmas) }
            self.timeSteps = karrasTimeSteps

            karrasSigmas.append(karrasSigmas.last!)

            self.alphaT = vDSP.divide(1, vForce.sqrt(vDSP.add(1, vDSP.square(karrasSigmas))))
            self.sigmaT = vDSP.multiply(karrasSigmas, self.alphaT)
            usingKarrasSigmas = true
        }

        self.lambdaT = zip(self.alphaT, self.sigmaT).map { α, σ in log(α) - log(σ) }
    }

    func timestepToIndex(_ timestep: Int) -> Int {
        guard usingKarrasSigmas else { return timestep }
        return self.timeSteps.firstIndex(of: timestep) ?? 0
    }

    /// Convert the model output to the corresponding type the algorithm needs.
    /// This implementation is for second-order DPM-Solver++.
    public func convertModelOutput(modelOutput: [Float], timestep: Int, sample: [Float]) -> [Float] {
        let count = modelOutput.count
        let sigmaIndex = timestepToIndex(timestep)
        let (at, st) = (self.alphaT[sigmaIndex], self.sigmaT[sigmaIndex])

        var result = [Float](repeating: 0, count: count)
        switch predictionType {
        case .epsilon:
            for i in 0..<count {
                result[i] = (sample[i] - modelOutput[i] * st) / at
            }
        case .vPrediction:
            for i in 0..<count {
                result[i] = at * sample[i] - st * modelOutput[i]
            }
        case .flow:
            preconditionFailure(
                "DPMSolverMultistepScheduler does not support flow prediction; use DiscreteFlowScheduler")
        case .flowMatching:
            fatalError("DPMSolverMultistepScheduler does not support flowMatching prediction type")
        }
        return result
    }

    /// One step for the first-order DPM-Solver (equivalent to DDIM).
    /// See https://arxiv.org/abs/2206.00927 for the detailed derivation.
    /// var names and code structure mostly follow https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
    func firstOrderUpdate(
        modelOutput: [Float],
        timestep: Int,
        prevTimestep: Int,
        sample: [Float]
    ) -> [Float] {
        let prevIndex = timestepToIndex(prevTimestep)
        let currIndex = timestepToIndex(timestep)
        let (pLambdaT, lambdaS) = (Double(lambdaT[prevIndex]), Double(lambdaT[currIndex]))
        let pAlphaT = Double(alphaT[prevIndex])
        let (pSigmaT, sigmaS) = (Double(sigmaT[prevIndex]), Double(sigmaT[currIndex]))
        let h = pLambdaT - lambdaS
        // xt = (sigmaT / sigmaS) * sample - (alphaT * (torch.exp(-h) - 1.0)) * model_output
        let xt = weightedSum(
            [pSigmaT / sigmaS, -pAlphaT * (exp(-h) - 1)],
            [sample, modelOutput]
        )
        return xt
    }

    /// One step for the second-order multistep DPM-Solver++ algorithm, using the midpoint method.
    /// var names and code structure mostly follow https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
    func secondOrderUpdate(
        modelOutputs: [[Float]],
        timesteps: [Int],
        prevTimestep t: Int,
        sample: [Float]
    ) -> [Float] {
        let (s0, s1) = (timesteps[back: 1], timesteps[back: 2])
        let (m0, m1) = (modelOutputs[back: 1], modelOutputs[back: 2])
        let (pLambdaT, lambdaS0, lambdaS1) = (
            Double(lambdaT[timestepToIndex(t)]),
            Double(lambdaT[timestepToIndex(s0)]),
            Double(lambdaT[timestepToIndex(s1)])
        )
        let pAlphaT = Double(alphaT[timestepToIndex(t)])
        let (pSigmaT, sigmaS0) = (Double(sigmaT[timestepToIndex(t)]), Double(sigmaT[timestepToIndex(s0)]))
        let (h, h0) = (pLambdaT - lambdaS0, lambdaS0 - lambdaS1)
        let r0 = h0 / h
        let d0 = m0

        // d1 = (1.0 / r0) * (m0 - m1)
        let d1 = weightedSum(
            [1 / r0, -1 / r0],
            [m0, m1]
        )

        // See https://arxiv.org/abs/2211.01095 for detailed derivations
        // xt = (
        //     (sigmaT / sigmaS0) * sample
        //     - (alphaT * (torch.exp(-h) - 1.0)) * d0
        //     - 0.5 * (alphaT * (torch.exp(-h) - 1.0)) * d1
        // )
        let xt = weightedSum(
            [pSigmaT / sigmaS0, -pAlphaT * (exp(-h) - 1), -0.5 * pAlphaT * (exp(-h) - 1)],
            [sample, d0, d1]
        )
        return xt
    }

    public func step(output: [Float], timeStep t: Int, sample: [Float]) -> [Float] {
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        let prevTimestep = stepIndex == timeSteps.count - 1 ? 0 : timeSteps[stepIndex + 1]

        let lowerOrderFinal = useLowerOrderFinal && stepIndex == timeSteps.count - 1
        let lowerOrderSecond = useLowerOrderFinal && stepIndex == timeSteps.count - 2
        let lowerOrder = lowerOrderStepped < 1 || lowerOrderFinal || lowerOrderSecond

        let modelOutput = convertModelOutput(modelOutput: output, timestep: t, sample: sample)
        if modelOutputs.count == solverOrder { modelOutputs.removeFirst() }
        modelOutputs.append(modelOutput)

        let prevSample: [Float]
        if lowerOrder {
            prevSample = firstOrderUpdate(
                modelOutput: modelOutput, timestep: t, prevTimestep: prevTimestep, sample: sample)
        } else {
            prevSample = secondOrderUpdate(
                modelOutputs: modelOutputs,
                timesteps: [timeSteps[stepIndex - 1], t],
                prevTimestep: prevTimestep,
                sample: sample
            )
        }
        if lowerOrderStepped < solverOrder {
            lowerOrderStepped += 1
        }

        return prevSample
    }
}

func sigmaToTimestep(sigma: Float, logSigmas: [Float]) -> Int {
    let logSigma = log(sigma)
    let dists = logSigmas.map { logSigma - $0 }

    // last index that is not negative, clipped to last index - 1
    var lowIndex = dists.reduce(-1) { partialResult, dist in
        return dist >= 0 && partialResult < dists.endIndex - 2 ? partialResult + 1 : partialResult
    }
    lowIndex = max(lowIndex, 0)
    let highIndex = lowIndex + 1

    let low = logSigmas[lowIndex]
    let high = logSigmas[highIndex]

    // Interpolate sigmas
    let w = ((low - logSigma) / (low - high)).clipped(to: 0...1)

    // transform interpolated value to time range
    let t = (1 - w) * Float(lowIndex) + w * Float(highIndex)
    return Int(round(t))
}

extension FloatingPoint {
    func clipped(to range: ClosedRange<Self>) -> Self {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}
