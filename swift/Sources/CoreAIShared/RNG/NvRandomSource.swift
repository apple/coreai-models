// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Matches NVIDIA cuRAND Philox 4x32-10 RNG.
/// Used by some ComfyUI and Automatic1111 workflows for seed-compatible generation.
public struct NvRandomSource: RandomSource, Sendable {
    public let seed: UInt64
    private var offset: UInt32

    public init(seed: UInt32) {
        self.seed = UInt64(seed)
        offset = 0
    }

    private static let philoxM4x32: (UInt32, UInt32) = (0xD251_1F53, 0xCD9E_8D57)
    private static let philoxW32: (UInt32, UInt32) = (0x9E37_79B9, 0xBB67_AE85)

    private static func philox4Round(counter: inout [[UInt32]], key: [[UInt32]]) {
        for i in 0..<counter[0].count {
            let v1: UInt64 = UInt64(counter[0][i]) * UInt64(philoxM4x32.0)
            let v2: UInt64 = UInt64(counter[2][i]) * UInt64(philoxM4x32.1)
            counter[0][i] = UInt32(v2 >> 32) ^ counter[1][i] ^ key[0][i]
            counter[1][i] = UInt32(v2 & 0xffff_ffff)
            counter[2][i] = UInt32(v1 >> 32) ^ counter[3][i] ^ key[1][i]
            counter[3][i] = UInt32(v1 & 0xffff_ffff)
        }
    }

    private static func philox4Bumpkey(key: inout [[UInt32]]) {
        for (i, element) in key[0].enumerated() {
            key[0][i] = element &+ philoxW32.0
        }
        for (i, element) in key[1].enumerated() {
            key[1][i] = element &+ philoxW32.1
        }
    }

    private static func philox4x32(counter: inout [[UInt32]], key: inout [[UInt32]], rounds: Int = 10) {
        for _ in 0..<(rounds - 1) {
            philox4Round(counter: &counter, key: key)
            philox4Bumpkey(key: &key)
        }
        philox4Round(counter: &counter, key: key)
    }

    private func boxMuller(_ counter1: [UInt32], _ counter2: [UInt32], mean: Double, stdev: Double) -> [Double] {
        zip(counter1, counter2).map {
            let u: Double = Double($0) / 4294967296.0 + (1.0 / 8589934592.0)
            let v: Double = Double($1) * (.pi / 2147483648.0) + (.pi / 4294967296.0)
            let radius = stdev * sqrt(-2.0 * log(u))
            return radius * sin(v) + mean
        }
    }

    private mutating func normalDoubleArray(count: Int, mean: Double, stdev: Double) -> [Double] {
        var counter: [[UInt32]] = [
            Array(repeating: offset, count: count),
            Array(repeating: 0, count: count),
            Array(0..<UInt32(count)),
            Array(repeating: 0, count: count),
        ]
        offset += 1
        var key: [[UInt32]] = [
            Array(repeating: UInt32(seed & 0xffff_ffff), count: count),
            Array(repeating: UInt32(seed >> 32), count: count),
        ]
        Self.philox4x32(counter: &counter, key: &key)
        return boxMuller(counter[0], counter[1], mean: mean, stdev: stdev)
    }

    public mutating func nextNormal(mean: Double = 0.0, stdev: Double = 1.0) -> Double {
        normalDoubleArray(count: 1, mean: mean, stdev: stdev)[0]
    }

    public mutating func normalArray(_ shape: [Int], mean: Double = 0.0, stdev: Double = 1.0) -> [Float] {
        let count = shape.reduce(1, *)
        return normalDoubleArray(count: count, mean: mean, stdev: stdev).map { Float($0) }
    }
}
