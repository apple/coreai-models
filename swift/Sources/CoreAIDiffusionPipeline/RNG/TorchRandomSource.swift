// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Default random source. Matches PyTorch's RNG (`torch.manual_seed`).
/// Used by SDXL, SD3, Flux, and most modern diffusion models.
public struct TorchRandomSource: RandomNumberGenerator, RandomSource, Sendable {
    struct State {
        var key = [UInt32](repeating: 0, count: 624)
        var pos: Int = 0
        var nextGauss: Double? = nil
    }

    var state: State

    public init(seed: UInt32) {
        state = .init()
        var s = seed & 0xffff_ffff
        for i in 0..<state.key.count {
            state.key[i] = s
            s = UInt32((UInt64(1_812_433_253) * UInt64(s ^ (s >> 30)) + UInt64(i) + 1) & 0xffff_ffff)
        }
        state.pos = state.key.count
        state.nextGauss = nil
    }

    mutating func nextUInt32() -> UInt32 {
        let n = 624
        let m = 397
        let matrixA: UInt64 = 0x9908_b0df
        let upperMask: UInt32 = 0x8000_0000
        let lowerMask: UInt32 = 0x7fff_ffff

        var y: UInt32
        if state.pos == state.key.count {
            for i in 0..<(n - m) {
                y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
                state.key[i] = state.key[i + m] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
            }
            for i in (n - m)..<(n - 1) {
                y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
                state.key[i] = state.key[i + (m - n)] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
            }
            y = (state.key[n - 1] & upperMask) | (state.key[0] & lowerMask)
            state.key[n - 1] = state.key[m - 1] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
            state.pos = 0
        }
        y = state.key[state.pos]
        state.pos += 1

        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c_5680
        y ^= (y << 15) & 0xefc6_0000
        y ^= (y >> 18)

        return y
    }

    public mutating func next() -> UInt64 {
        let high = nextUInt32()
        let low = nextUInt32()
        return (UInt64(high) << 32) | UInt64(low)
    }

    mutating func nextDouble() -> Double {
        let a = next()
        return Double(a & 9_007_199_254_740_991) * (1.0 / 9007199254740992.0)
    }

    mutating func nextFloat() -> Float {
        let a = nextUInt32()
        return Float(a & 16_777_215) * (1.0 / 16777216.0)
    }

    mutating func nextGauss() -> Double {
        if let nextGauss = state.nextGauss {
            state.nextGauss = nil
            return nextGauss
        }
        let u1: Double = nextDouble()
        let u2: Double = 1 - nextDouble()
        let radius = sqrt(-2.0 * log(u2))
        let theta = 2.0 * .pi * u1
        state.nextGauss = radius * sin(theta)
        return radius * cos(theta)
    }

    public mutating func nextNormal(mean: Double = 0.0, stdev: Double = 1.0) -> Double {
        nextGauss() * stdev + mean
    }

    /// Matches `torch.randn(shape, dtype=.float)`, including PyTorch's `normal_fill_16`
    /// batch-16 Box-Muller path for counts >= 16 (see the step comments below).
    public mutating func normalArray(_ shape: [Int], mean: Double = 0.0, stdev: Double = 1.0) -> [Float] {
        let count = shape.reduce(1, *)
        guard count >= 16 else {
            return (0..<count).map { _ in Float(nextNormal(mean: mean, stdev: stdev)) }
        }

        // Step 1: Fill all elements with float32 uniforms (matches PyTorch's initial fill).
        var data = (0..<count).map { _ in Double(nextFloat()) }

        // Step 2: Box-Muller on complete 16-element blocks.
        for blockStart in stride(from: 0, to: count - 15, by: 16) {
            for j in 0..<8 {
                let u1 = 1 - data[blockStart + j]
                let u2 = data[blockStart + j + 8]
                let radius = sqrt(-2.0 * log(u1))
                let theta = 2.0 * .pi * u2
                data[blockStart + j] = radius * cos(theta) * stdev + mean
                data[blockStart + j + 8] = radius * sin(theta) * stdev + mean
            }
        }

        // Step 3: Remainder — re-fill last 16 with fresh uniforms, re-apply Box-Muller.
        if count % 16 != 0 {
            let blockStart = count - 16
            for i in 0..<16 {
                data[blockStart + i] = Double(nextFloat())
            }
            for j in 0..<8 {
                let u1 = 1 - data[blockStart + j]
                let u2 = data[blockStart + j + 8]
                let radius = sqrt(-2.0 * log(u1))
                let theta = 2.0 * .pi * u2
                data[blockStart + j] = radius * cos(theta) * stdev + mean
                data[blockStart + j + 8] = radius * sin(theta) * stdev + mean
            }
        }

        return data.map { Float($0) }
    }
}
