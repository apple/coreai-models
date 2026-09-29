// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Matches NumPy's legacy RNG (`numpy.random.RandomState`).
public struct NumPyRandomSource: RandomNumberGenerator, RandomSource, Sendable {
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
        let low = nextUInt32()
        let high = nextUInt32()
        return (UInt64(high) << 32) | UInt64(low)
    }

    mutating func nextDouble() -> Double {
        let a = Double(nextUInt32() >> 5)
        let b = Double(nextUInt32() >> 6)
        return (a * 67108864.0 + b) / 9007199254740992.0
    }

    mutating func nextGauss() -> Double {
        if let nextGauss = state.nextGauss {
            state.nextGauss = nil
            return nextGauss
        }
        var x1: Double
        var x2: Double
        var r2: Double
        repeat {
            x1 = 2.0 * nextDouble() - 1.0
            x2 = 2.0 * nextDouble() - 1.0
            r2 = x1 * x1 + x2 * x2
        } while r2 >= 1.0 || r2 == 0.0

        let f = sqrt(-2.0 * log(r2) / r2)
        state.nextGauss = f * x1
        return f * x2
    }

    public mutating func nextNormal(mean: Double = 0.0, stdev: Double = 1.0) -> Double {
        nextGauss() * stdev + mean
    }

    public mutating func normalArray(_ shape: [Int], mean: Double = 0.0, stdev: Double = 1.0) -> [Float] {
        let count = shape.reduce(1, *)
        return (0..<count).map { _ in Float(nextNormal(mean: mean, stdev: stdev)) }
    }
}
