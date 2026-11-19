// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// A fast, general-purpose pseudo-random number generator implementing `xoshiro256**`
/// (xor-shift-rotate 256 star-star) with a period of 2^256 - 1.
///
/// Designed by David Blackman and Sebastiano Vigna (2018). It passes all BigCrush and
/// PractRand statistical randomness tests and executes with high throughput using only
/// basic 64-bit integer bitwise operations.
///
/// Reference: https://prng.di.unimi.it/xoshiro256starstar.c
///
/// Used exclusively across test suites so multinomial sampling is deterministic across
/// runs and across refactors. Production code uses `SystemRandomNumberGenerator` or
/// `SeededRandomNumberGenerator`.
public struct Xoshiro256StarStar: RandomNumberGenerator, Sendable {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    /// Creates a generator initialized with an explicit 256-bit state.
    ///
    /// At least one word in `state` must be non-zero.
    public init(state: (UInt64, UInt64, UInt64, UInt64)) {
        precondition(
            state.0 != 0 || state.1 != 0 || state.2 != 0 || state.3 != 0,
            "Xoshiro256StarStar state cannot be all zeros."
        )
        self.state = state
    }

    /// Creates a generator seeded from a single 64-bit seed using SplitMix64
    /// to initialize the four 64-bit state registers.
    public init(seed: UInt64) {
        // Seed via SplitMix64 to avoid weak Xoshiro initial states.
        var z = seed &+ 0x9E37_79B9_7F4A_7C15
        let splitmix = { () -> UInt64 in
            z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z &>> 31)
        }
        self.state = (splitmix(), splitmix(), splitmix(), splitmix())
    }

    /// The current 256-bit state tuple `(s0, s1, s2, s3)`.
    public var currentState: (UInt64, UInt64, UInt64, UInt64) {
        state
    }

    public mutating func next() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let t = state.1 &<< 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= t
        state.3 = rotl(state.3, 45)
        return result
    }

    private func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 {
        (x &<< k) | (x &>> (64 &- k))
    }
}
