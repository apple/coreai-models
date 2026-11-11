// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAIShared

@Suite("CoreAIShared RNG")
struct RNGTests {
    // MARK: - NumPyRandomSource

    // Reference values generated with: numpy.random.RandomState(42).standard_normal() × 4
    @Test("NumPy RNG matches Python reference (seed=42)")
    func numpyReference() {
        var rng = NumPyRandomSource(seed: 42)
        let samples = (0..<4).map { _ in rng.nextNormal(mean: 0, stdev: 1) }

        #expect(abs(samples[0] - 0.4967141530) < 1e-6)
        #expect(abs(samples[1] - (-0.1382643012)) < 1e-6)
        #expect(abs(samples[2] - 0.6476885381) < 1e-6)
        #expect(abs(samples[3] - 1.5230298564) < 1e-6)
    }

    @Test("NumPy RNG deterministic across runs")
    func numpyDeterministic() {
        var rng1 = NumPyRandomSource(seed: 123)
        var rng2 = NumPyRandomSource(seed: 123)
        let a = (0..<100).map { _ in rng1.nextNormal(mean: 0, stdev: 1) }
        let b = (0..<100).map { _ in rng2.nextNormal(mean: 0, stdev: 1) }
        #expect(a == b)
    }

    @Test("NumPy RNG different seeds diverge")
    func numpyDifferentSeeds() {
        var rng1 = NumPyRandomSource(seed: 1)
        var rng2 = NumPyRandomSource(seed: 2)
        let a = (0..<10).map { _ in rng1.nextNormal(mean: 0, stdev: 1) }
        let b = (0..<10).map { _ in rng2.nextNormal(mean: 0, stdev: 1) }
        #expect(a != b)
    }

    // MARK: - TorchRandomSource

    @Test("Torch RNG deterministic across runs")
    func torchDeterministic() {
        var rng1 = TorchRandomSource(seed: 42)
        var rng2 = TorchRandomSource(seed: 42)
        let a = rng1.normalArray([64], mean: 0, stdev: 1)
        let b = rng2.normalArray([64], mean: 0, stdev: 1)
        #expect(a == b)
    }

    @Test("Torch RNG batch-16 path produces expected distribution")
    func torchBatch16() {
        var rng = TorchRandomSource(seed: 7)
        let samples = rng.normalArray([1024], mean: 0, stdev: 1)
        let mean = samples.reduce(0, +) / Float(samples.count)
        let variance = samples.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(samples.count)
        #expect(abs(mean) < 0.1)
        #expect(abs(variance - 1.0) < 0.15)
    }

    @Test("Torch RNG handles small and remainder counts")
    func torchSmallAndRemainderCounts() {
        var rng = TorchRandomSource(seed: 99)
        // Count < 16 (scalar fallback path)
        let small = rng.normalArray([7], mean: 2.0, stdev: 0.5)
        #expect(small.count == 7)

        // Count with non-16 remainder
        let odd = rng.normalArray([25], mean: 0, stdev: 1)
        #expect(odd.count == 25)
    }

    @Test("Torch RNG different seeds produce different sequences")
    func torchDifferentSeeds() {
        var rng1 = TorchRandomSource(seed: 1)
        var rng2 = TorchRandomSource(seed: 2)
        let a = rng1.normalArray([16], mean: 0, stdev: 1)
        let b = rng2.normalArray([16], mean: 0, stdev: 1)
        #expect(a != b)
    }

    @Test("Torch RNG normalArray respects mean and stdev")
    func meanStdev() {
        var rng = TorchRandomSource(seed: 42)
        let samples = rng.normalArray([4096], mean: 5.0, stdev: 2.0)
        let mean = samples.reduce(0, +) / Float(samples.count)
        #expect(abs(mean - 5.0) < 0.2)
    }

    // MARK: - Torch RNG parity with Python torch.randn

    // Reference: torch.manual_seed(42); [torch.randn(1, dtype=torch.float64).item() for _ in range(8)]
    @Test("Torch scalar path matches Python reference (seed=42)")
    func torchScalarParity() {
        var rng = TorchRandomSource(seed: 42)
        let expected: [Double] = [
            0.3366903544, 0.1288094051, 0.2344623634, 0.2303330279,
            -1.1228563767, -0.1863282993, 2.2082013356, -0.6379970568,
        ]
        for (i, exp) in expected.enumerated() {
            let got = rng.nextNormal(mean: 0, stdev: 1)
            #expect(abs(got - exp) < 1e-6, "scalar[\(i)]: got \(got), expected \(exp)")
        }
    }

    // Reference: torch.manual_seed(42); torch.randn(32, dtype=torch.float32)
    @Test("Torch batch path matches Python torch.randn (seed=42, count=32)")
    func torchBatchParity32() {
        var rng = TorchRandomSource(seed: 42)
        let expected: [Float] = [
            1.9269150496, 1.4872841835, 0.9007171988, -2.1055214405,
            0.6784184575, -1.2345449924, -0.0430674814, -1.6046669483,
            -0.7521361709, 1.6487228870, -0.3924786448, -1.4036067724,
            -0.7278812528, -0.5594298840, -0.7688389421, 0.7624453902,
            1.6423169374, -0.1595973223, -0.4973974824, 0.4395892322,
            -0.7581311464, 1.0783176422, 0.8008005023, 1.6806205511,
            1.2791243792, 1.2964228392, 0.6104664803, 1.3347377777,
            -0.2316243201, 0.0417594910, -0.2515752614, 0.8598585129,
        ]
        let got = rng.normalArray([32], mean: 0, stdev: 1)
        for (i, (g, e)) in zip(got, expected).enumerated() {
            #expect(abs(g - e) < 1e-4, "batch[\(i)]: got \(g), expected \(e)")
        }
    }

    // Reference: torch.manual_seed(42); torch.randn(16, dtype=torch.float32)
    @Test("Torch batch path boundary (exactly 16 elements)")
    func torchBatchBoundary16() {
        var rng = TorchRandomSource(seed: 42)
        let expected: [Float] = [
            1.9269150496, 1.4872841835, 0.9007171988, -2.1055214405,
            0.6784184575, -1.2345449924, -0.0430674814, -1.6046669483,
            -0.7521361709, 1.6487228870, -0.3924786448, -1.4036067724,
            -0.7278812528, -0.5594298840, -0.7688389421, 0.7624453902,
        ]
        let got = rng.normalArray([16], mean: 0, stdev: 1)
        for (i, (g, e)) in zip(got, expected).enumerated() {
            #expect(abs(g - e) < 1e-4, "boundary[\(i)]: got \(g), expected \(e)")
        }
    }

    // Reference: torch.manual_seed(42); torch.randn(17, dtype=torch.float32)
    @Test("Torch batch path remainder (17 elements)")
    func torchBatchRemainder17() {
        var rng = TorchRandomSource(seed: 42)
        let expected: [Float] = [
            1.9269150496, -0.1595973223, -0.4973974824, 0.4395892322,
            -0.7581311464, 1.0783176422, 0.8008005023, 1.6806205511,
            0.3558597863, 1.2964228392, 0.6104664803, 1.3347377777,
            -0.2316243201, 0.0417594910, -0.2515752614, 0.8598585129,
            -0.3097269237,
        ]
        let got = rng.normalArray([17], mean: 0, stdev: 1)
        for (i, (g, e)) in zip(got, expected).enumerated() {
            #expect(abs(g - e) < 1e-4, "remainder[\(i)]: got \(g), expected \(e)")
        }
    }

    // Reference: torch.manual_seed(0); torch.randn(32, dtype=torch.float32)
    @Test("Torch batch path matches Python torch.randn (seed=0, count=32)")
    func torchBatchParity0() {
        var rng = TorchRandomSource(seed: 0)
        let expected: [Float] = [
            -1.1258398294, -1.1523602009, -0.2505785823, -0.4338788390,
            0.8487103581, 0.6920092106, -0.3160127699, -2.1152195930,
            0.3222749233, -1.2633347511, 0.3499831855, 0.3081339002,
            0.1198415086, 1.2376579046, 1.1167771816, -0.2472776473,
            -1.3526537418, -1.6959313154, 0.5666505098, 0.7935084105,
            0.5988394618, -1.5550950766, -0.3413603008, 1.8530061245,
            0.7501894236, -0.5854971409, -0.1733970195, 0.1834779233,
            1.3893661499, 1.5863343477, 0.9462983608, -0.8436768055,
        ]
        let got = rng.normalArray([32], mean: 0, stdev: 1)
        for (i, (g, e)) in zip(got, expected).enumerated() {
            #expect(abs(g - e) < 1e-4, "batch[\(i)]: got \(g), expected \(e)")
        }
    }

    // Reference: torch.manual_seed(42); t = torch.randn([1,16,16,16], dtype=torch.float32)
    @Test("Torch batch path realistic shape (4096 elements, seed=42)")
    func torchBatchRealisticShape() {
        var rng = TorchRandomSource(seed: 42)
        let got = rng.normalArray([1, 16, 16, 16], mean: 0, stdev: 1)
        #expect(got.count == 4096)

        let expectedFirst: [Float] = [
            1.9269150496, 1.4872841835, 0.9007171988, -2.1055214405,
            0.6784184575, -1.2345449924, -0.0430674814, -1.6046669483,
        ]
        let expectedLast: [Float] = [
            1.5869791508, 0.1421326697, 0.3760589659, -0.7916260362,
            2.6677629948, -0.1403129250, 0.9416193962, -0.0118428767,
        ]
        for (i, (g, e)) in zip(got.prefix(8), expectedFirst).enumerated() {
            #expect(abs(g - e) < 1e-4, "first[\(i)]: got \(g), expected \(e)")
        }
        for (i, (g, e)) in zip(got.suffix(8), expectedLast).enumerated() {
            #expect(abs(g - e) < 1e-4, "last[\(i)]: got \(g), expected \(e)")
        }
    }

    // MARK: - NvRandomSource

    @Test("Nv RNG deterministic across runs")
    func nvDeterministic() {
        var rng1 = NvRandomSource(seed: 42)
        var rng2 = NvRandomSource(seed: 42)
        let a = rng1.normalArray([256], mean: 0, stdev: 1)
        let b = rng2.normalArray([256], mean: 0, stdev: 1)
        #expect(a == b)
    }

    @Test("Nv RNG different seeds diverge")
    func nvDifferentSeeds() {
        var rng1 = NvRandomSource(seed: 100)
        var rng2 = NvRandomSource(seed: 200)
        let a = rng1.normalArray([32], mean: 0, stdev: 1)
        let b = rng2.normalArray([32], mean: 0, stdev: 1)
        #expect(a != b)
    }

    // MARK: - SeededRandomNumberGenerator (SplitMix64)

    @Test("SeededRandomNumberGenerator: same seed reproduces same stream")
    func seededRNGSameSeed() {
        var a = SeededRandomNumberGenerator(seed: 12345)
        var b = SeededRandomNumberGenerator(seed: 12345)
        for _ in 0..<64 {
            #expect(a.next() == b.next())
        }
    }

    @Test("SeededRandomNumberGenerator: different seeds diverge immediately")
    func seededRNGDifferentSeeds() {
        var a = SeededRandomNumberGenerator(seed: 1)
        var b = SeededRandomNumberGenerator(seed: 2)
        #expect(a.next() != b.next())
    }

    @Test("SeededRandomNumberGenerator: boundary seeds produce output")
    func seededRNGBoundaries() {
        var zero = SeededRandomNumberGenerator(seed: 0)
        var max = SeededRandomNumberGenerator(seed: UInt64.max)
        #expect(zero.next() != 0)
        #expect(max.next() != 0)
    }

    // MARK: - Xoshiro256StarStar (TestUtilities)

    @Test("Xoshiro256StarStar: same seed reproduces same stream")
    func xoshiroSameSeed() {
        var a = Xoshiro256StarStar(seed: 0xDEAD_BEEF)
        var b = Xoshiro256StarStar(seed: 0xDEAD_BEEF)
        for _ in 0..<64 {
            #expect(a.next() == b.next())
        }
    }

    @Test("Xoshiro256StarStar: different seeds diverge")
    func xoshiroDifferentSeeds() {
        var a = Xoshiro256StarStar(seed: 10)
        var b = Xoshiro256StarStar(seed: 20)
        #expect(a.next() != b.next())
    }

    @Test("Xoshiro256StarStar: explicit state initialization and progression")
    func xoshiroExplicitState() {
        let initialState: (UInt64, UInt64, UInt64, UInt64) = (1, 2, 3, 4)
        var rng = Xoshiro256StarStar(state: initialState)
        #expect(rng.currentState == initialState)

        let first = rng.next()
        #expect(first != 0)
        #expect(rng.currentState != initialState)
    }
}
