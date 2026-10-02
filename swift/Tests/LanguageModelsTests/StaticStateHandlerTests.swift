// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Bucketed prefix re-layout

@Suite("BucketedStaticState prefix re-layout")
struct BucketedStaticStatePrefixTests {
    /// Builds a `[layers, 1, channels, 1, ctx]` cache filled so each element
    /// encodes its own (group, position) — `group * 1000 + position` — which makes
    /// a misplaced copy obvious rather than merely unequal.
    private func makeCache(layers: Int, channels: Int, context: Int) -> NDArray {
        var array = NDArray(shape: [layers, 1, channels, 1, context], scalarType: .float16)
        let groups = layers * channels
        fillNDArray(&array, as: Float16.self, count: groups * context) { index in
            Float16(index / context * 1000 + index % context)
        }
        return array
    }

    private func read(_ array: NDArray, count: Int) -> [Float16] {
        readNDArray(array, as: Float16.self, count: count)
    }

    @Test("Grow: every group's written prefix lands at the new sequence stride")
    func growPreservesPrefixPerGroup() {
        let layers = 3
        let channels = 4
        let oldContext = 8
        let newContext = 32
        let copyLength = 5

        let source = makeCache(layers: layers, channels: channels, context: oldContext)
        var destination = NDArray(
            shape: [layers, 1, channels, 1, newContext], scalarType: .float16)
        zeroFillNDArray(&destination)

        BucketedStaticState.copyPrefix(from: source, to: &destination, copyLength: copyLength)

        let groups = layers * channels
        let values = read(destination, count: groups * newContext)
        for group in 0..<groups {
            for position in 0..<newContext {
                let actual = values[group * newContext + position]
                let expected: Float16 =
                    position < copyLength ? Float16(group * 1000 + position) : 0
                #expect(
                    actual == expected,
                    "group \(group) position \(position): expected \(expected), got \(actual)")
            }
        }
    }

    @Test("Shrink: copies only what fits, leaving the rest zeroed")
    func shrinkCopiesClampedPrefix() {
        let layers = 2
        let channels = 2
        let source = makeCache(layers: layers, channels: channels, context: 16)
        var destination = NDArray(shape: [layers, 1, channels, 1, 4], scalarType: .float16)
        zeroFillNDArray(&destination)

        BucketedStaticState.copyPrefix(from: source, to: &destination, copyLength: 4)

        let groups = layers * channels
        let values = read(destination, count: groups * 4)
        for group in 0..<groups {
            for position in 0..<4 {
                #expect(values[group * 4 + position] == Float16(group * 1000 + position))
            }
        }
    }

    @Test("Zero copy length leaves the destination untouched")
    func zeroCopyLengthIsANoOp() {
        let source = makeCache(layers: 2, channels: 2, context: 8)
        var destination = NDArray(shape: [2, 1, 2, 1, 16], scalarType: .float16)
        zeroFillNDArray(&destination)

        BucketedStaticState.copyPrefix(from: source, to: &destination, copyLength: 0)

        #expect(read(destination, count: 4 * 16).allSatisfy { $0 == 0 })
    }

    @Test("bf16: the prefix is copied as raw 16-bit elements")
    func bfloat16CopiesRawBits() {
        // A typed Float16 view traps on a BFloat16 array, so fill and read the raw
        // bit patterns; the copy must move them verbatim.
        let groups = 4
        let oldContext = 8
        let newContext = 16
        let copyLength = 5
        var source = NDArray(shape: [2, 1, 2, 1, oldContext], scalarType: .bfloat16)
        source.mutableRawView().withUnsafeMutableBytes { raw, _, _ in
            let bits = raw.assumingMemoryBound(to: UInt16.self)
            for index in 0..<(groups * oldContext) {
                bits[index] = UInt16(index / oldContext * 1000 + index % oldContext)
            }
        }
        var destination = NDArray(shape: [2, 1, 2, 1, newContext], scalarType: .bfloat16)
        zeroFillNDArray(&destination)

        BucketedStaticState.copyPrefix(from: source, to: &destination, copyLength: copyLength)

        var values: [UInt16] = []
        destination.rawView().withUnsafeBytes { raw, _, _ in
            let bits = raw.assumingMemoryBound(to: UInt16.self)
            values = (0..<(groups * newContext)).map { bits[$0] }
        }
        for group in 0..<groups {
            for position in 0..<newContext {
                let expected = position < copyLength ? UInt16(group * 1000 + position) : 0
                #expect(values[group * newContext + position] == expected)
            }
        }
    }

    @Test("fp32: 4-byte elements land at the new sequence stride")
    func float32CopiesFourByteElements() {
        let groups = 4
        let oldContext = 8
        let newContext = 16
        let copyLength = 6
        var source = NDArray(shape: [2, 1, 2, 1, oldContext], scalarType: .float32)
        fillNDArray(&source, as: Float.self, count: groups * oldContext) { index in
            Float(index / oldContext * 1000 + index % oldContext)
        }
        var destination = NDArray(shape: [2, 1, 2, 1, newContext], scalarType: .float32)
        zeroFillNDArray(&destination)

        BucketedStaticState.copyPrefix(from: source, to: &destination, copyLength: copyLength)

        let values = readNDArray(destination, as: Float.self, count: groups * newContext)
        for group in 0..<groups {
            for position in 0..<newContext {
                let expected: Float = position < copyLength ? Float(group * 1000 + position) : 0
                #expect(values[group * newContext + position] == expected)
            }
        }
    }
}

// MARK: - Reset

@Suite("StaticStateStorage reset")
struct StaticStateStorageResetTests {
    @Test("Reset zeroes every state, so a restart can't see the previous sequence")
    func resetClearsPreviousSequence() {
        // Stand-ins for a leftover conversation: a global cache and a sliding ring,
        // in two of the scalar types a static state can have.
        var keyCache = NDArray(shape: [2, 1, 4, 1, 16], scalarType: .float16)
        fillNDArray(&keyCache, as: Float16.self, count: 2 * 4 * 16) { Float16($0 % 50 + 1) }
        var slidingCache = NDArray(shape: [2, 1, 4, 1, 8], scalarType: .float32)
        fillNDArray(&slidingCache, as: Float.self, count: 2 * 4 * 8) { Float($0 + 1) }
        let storage = StaticStateStorage(
            stateNames: ["key_cache", "sliding_key_cache"],
            arrays: ["key_cache": keyCache, "sliding_key_cache": slidingCache])

        storage.reset()

        let keys = readNDArray(storage.arrays["key_cache"]!, as: Float16.self, count: 2 * 4 * 16)
        let ring = readNDArray(storage.arrays["sliding_key_cache"]!, as: Float.self, count: 2 * 4 * 8)
        #expect(keys.allSatisfy { $0 == 0 })
        #expect(ring.allSatisfy { $0 == 0 })
    }
}

// MARK: - Fixed vs bucketed classification

/// Pins the rule that decides whether a state gets one shared buffer or one per
/// context bucket, using the real footprints of the two assets that sit on either
/// side of it.
@Suite("StaticStateFactory footprint classification")
struct StaticStateClassificationTests {
    typealias Layout = StaticStateFactory.StorageLayout

    @Test("A max-aligned cache is fixed: every bucket is the same buffer")
    func maxAlignedCacheIsFixed() {
        // qwen3_0_6b_mixed_4bit_8bit_static, key_cache. Buckets 256…4096 declare a
        // shrinking shape `[28, 1, 1024, 1, ctx]` over one 224 MiB allocation, so
        // byte count and strides are identical at every rung.
        let strides = [4_194_304, 4_194_304, 32_768, 32_768, 8]
        let layouts = [256, 512, 1024, 2048, 4096].map { _ in
            Layout(byteCount: 234_881_024, strides: strides)
        }
        #expect(StaticStateFactory.footprintVaries(layouts) == false)
    }

    @Test("A per-bucket ladder is bucketed: each bucket is its own buffer")
    func perBucketLadderIsBucketed() {
        // gemma-4-e2b-it_static, key_cache. Each bucket is allocated at its own
        // context length, so the sequence stride scales with the rung.
        let layouts = [
            Layout(byteCount: 3_145_728, strides: [524_288, 524_288, 8_192, 8_192, 8]),
            Layout(byteCount: 25_165_824, strides: [4_194_304, 4_194_304, 65_536, 65_536, 8]),
            Layout(byteCount: 100_663_296, strides: [16_777_216, 16_777_216, 262_144, 262_144, 8]),
            Layout(
                byteCount: 402_653_184,
                strides: [67_108_864, 67_108_864, 1_048_576, 1_048_576, 8]),
        ]
        #expect(StaticStateFactory.footprintVaries(layouts))
    }

    @Test("Equal byte counts still vary when the addressing differs")
    func stridesAloneDecide() {
        let layouts = [
            Layout(byteCount: 1_024, strides: [128, 1]),
            Layout(byteCount: 1_024, strides: [256, 1]),
        ]
        #expect(StaticStateFactory.footprintVaries(layouts))
    }

    @Test("A single bucket, or none, never varies")
    func degenerateLadders() {
        #expect(StaticStateFactory.footprintVaries([]) == false)
        #expect(
            StaticStateFactory.footprintVaries([Layout(byteCount: 64, strides: [8, 1])]) == false)
    }
}

// MARK: - Dual RoPE table

@Suite("DualRoPEInputHandler theta table")
struct DualRoPEThetaTests {
    private let rope = RoPEConfig(
        slidingHeadDim: 8,
        globalHeadDim: 8,
        slidingRopeTheta: 10_000,
        globalRopeTheta: 1_000_000,
        partialRotaryFactor: 0.5
    )

    @Test("Width is the sum of the two head dims")
    func combinedWidth() {
        #expect(DualRoPEInputHandler.buildTheta(rope).count == 16)
    }

    @Test("Sliding range is full-rotary and repeats its half twice")
    func slidingRangeRepeatsHalf() {
        let theta = DualRoPEInputHandler.buildTheta(rope)
        // GPT-NeoX layout: frequency k appears at both j and j + slidingHeadDim/2.
        for k in 0..<4 {
            #expect(abs(theta[k] - theta[k + 4]) < 1e-12)
            let expected = pow(10_000.0, -(Double(2 * k) / 8.0))
            #expect(abs(theta[k] - expected) < 1e-12)
        }
        // Full rotary: no zeros in the sliding range.
        #expect(theta[0..<8].allSatisfy { $0 > 0 })
    }

    @Test("Global range is partial-rotary: the tail frequencies are NoPE zeros")
    func globalRangeIsPartialRotary() {
        let theta = DualRoPEInputHandler.buildTheta(rope)
        // rotaryAngles = floor(0.5 * 8 / 2) = 2 of the 4 half-frequencies rotate.
        let global = Array(theta[8..<16])
        for m in 0..<2 {
            let expected = pow(1_000_000.0, -(Double(2 * m) / 8.0))
            #expect(abs(global[m] - expected) < 1e-12)
            #expect(abs(global[m + 4] - expected) < 1e-12)
        }
        for m in 2..<4 {
            #expect(global[m] == 0)
            #expect(global[m + 4] == 0)
        }
    }

    @Test("A zero partial-rotary factor makes the global range entirely NoPE")
    func zeroPartialRotaryFactorIsAllNoPE() {
        let noPE = RoPEConfig(
            slidingHeadDim: 4, globalHeadDim: 4,
            slidingRopeTheta: 10_000, globalRopeTheta: 1_000_000,
            partialRotaryFactor: 0
        )
        let theta = DualRoPEInputHandler.buildTheta(noPE)
        #expect(theta[0..<4].allSatisfy { $0 > 0 })
        #expect(theta[4..<8].allSatisfy { $0 == 0 })
    }
}
