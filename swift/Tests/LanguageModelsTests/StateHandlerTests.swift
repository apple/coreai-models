// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Zero-Fill Tests

@Suite("ZeroFill NDArray Tests")
struct ZeroFillNDArrayTests {
    @Test("Zero-fills a Float16 NDArray")
    func zeroFillFloat16() {
        var array = NDArray(shape: [2, 4], scalarType: .float16)
        fillNDArray(&array, as: Float16.self, count: 8) { Float16($0 + 1) }
        zeroFillNDArray(&array)
        let values = readNDArray(array, as: Float16.self, count: 8)
        for v in values {
            #expect(v == 0, "Expected 0, got \(v)")
        }
    }

    @Test("Zero-fills a Float32 NDArray")
    func zeroFillFloat32() {
        var array = NDArray(shape: [2, 4], scalarType: .float32)
        fillNDArray(&array, as: Float.self, count: 8) { Float($0 + 1) }
        zeroFillNDArray(&array)
        let values = readNDArray(array, as: Float.self, count: 8)
        for v in values {
            #expect(v == 0, "Expected 0, got \(v)")
        }
    }

    @Test("Zero-fills a BFloat16 NDArray")
    func zeroFillBFloat16() {
        // BFloat16 states (e.g. gpt-oss-20b) reach zeroFillNDArray through reset(). A typed
        // Float16 view traps on them, so this exercises the raw-view path.
        var array = NDArray(shape: [2, 4], scalarType: .bfloat16)
        array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
            let dst = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0..<8 { dst[i] = 0x3F80 }  // 1.0 in bf16
        }
        zeroFillNDArray(&array)
        array.rawView().withUnsafeBytes { ptr, _, _ in
            let src = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0..<8 {
                #expect(src[i] == 0, "Expected 0, got \(src[i])")
            }
        }
    }

    @Test("Zero-fills a high-rank NDArray")
    func zeroFillHighRank() {
        var array = NDArray(shape: [2, 4, 8, 16], scalarType: .float16)
        let count = 2 * 4 * 8 * 16
        fillNDArray(&array, as: Float16.self, count: count) { Float16($0 % 100) }
        zeroFillNDArray(&array)
        let values = readNDArray(array, as: Float16.self, count: count)
        #expect(values.allSatisfy { $0 == 0 })
    }
}

// MARK: - StateKind Tests

@Suite("StateKind Tests")
struct StateKindTests {
    @Test("StateKind raw values")
    func rawValues() {
        #expect(StateKind.kvCache.rawValue == "kv_cache")
        #expect(StateKind.slidingCache.rawValue == "sliding_cache")
        #expect(StateKind.fixed.rawValue == "fixed")
    }

    @Test("StateKind decodes from JSON")
    func decodable() throws {
        let json = """
            {"key": "kv_cache", "sliding": "sliding_cache", "fix": "fixed"}
            """
        struct Wrapper: Decodable {
            let key: StateKind
            let sliding: StateKind
            let fix: StateKind
        }
        let decoded = try JSONDecoder().decode(Wrapper.self, from: json.data(using: .utf8)!)
        #expect(decoded.key == StateKind.kvCache)
        #expect(decoded.sliding == StateKind.slidingCache)
        #expect(decoded.fix == StateKind.fixed)
    }
}

// MARK: - Protocol Conformance Tests

@Suite("StateHandler Conformance Tests")
struct StateHandlerConformanceTests {
    @Test("GrowingNDArrayState conforms to SyncStateHandler")
    func growingConformance() {
        let _: any SyncStateHandler.Type = GrowingNDArrayState.self
    }

    @Test("FixedNDArrayState conforms to SyncStateHandler")
    func fixedConformance() {
        let _: any SyncStateHandler.Type = FixedNDArrayState.self
    }
}

// MARK: - KV Cache Shape Resolution (batch-aware)

@Suite("KV cache dynamic-shape resolution")
struct KVCacheShapeResolutionTests {
    // Cache layout is [n_layers, batch, n_kv_heads, seq, head_dim]; seq is second-to-last (index 3).
    let seqDim = 3

    @Test("grows the seq dim and pins a dynamic batch dim to batchSize")
    func dynamicBatchAndSeq() {
        let resolved = GrowingNDArrayState.resolveKVShape(
            [2, -1, 4, -1, 8], sequenceDimIndex: seqDim, sequenceLength: 256, batchSize: 3)
        #expect(resolved == [2, 3, 4, 256, 8])
    }

    @Test("single dynamic seq dim leaves a static batch untouched")
    func onlySeqDynamic() {
        let resolved = GrowingNDArrayState.resolveKVShape(
            [2, 1, 4, -1, 8], sequenceDimIndex: seqDim, sequenceLength: 256, batchSize: 1)
        #expect(resolved == [2, 1, 4, 256, 8])
    }

    @Test("default batchSize=1 resolves a dynamic batch dim to 1")
    func dynamicBatchDefaultsToOne() {
        let resolved = GrowingNDArrayState.resolveKVShape(
            [2, -1, 4, -1, 8], sequenceDimIndex: seqDim, sequenceLength: 512, batchSize: 1)
        #expect(resolved == [2, 1, 4, 512, 8])
    }
}

// MARK: - withBoundStates Tests

/// Minimal state handler for testing the binding API.
final class MockStateHandler: SyncStateHandler {
    var stateNames: [String]
    var stateCount: Int { arrays.count }
    let currentCapacity: Int = .max
    let supportsTruncation: Bool = false

    private var arrays: [String: NDArray]

    init(names: [String], shape: [Int], scalarType: NDArray.ScalarType = .float16) {
        self.stateNames = names
        self.arrays = Dictionary(
            uniqueKeysWithValues: names.map { ($0, NDArray(shape: shape, scalarType: scalarType)) })
    }

    func ensureCapacity(forContextLength contextLength: Int) throws -> Bool { false }

    subscript(stateIndex index: Int) -> (name: String, array: NDArray) {
        get { (stateNames[index], arrays[stateNames[index]]!) }
        set { arrays[stateNames[index]] = newValue.array }
    }

    @_lifetime(views: borrow self)
    func bind(into views: inout InferenceFunction.MutableViews) {
        for name in stateNames {
            let view = _overrideLifetime(arrays[name]!.mutableRawView(), borrowing: Void())
            views.insert(view, for: name)
        }
    }

    func reset() {}
    func truncate(to tokenCount: Int) {}
}

@Suite("bind(into:) Tests")
struct BindTests {
    @Test("binds 1 through 4 states into MutableViews")
    func bindsVariousCounts() {
        for count in 1...4 {
            let names = (0..<count).map { "state_\($0)" }
            let handler = MockStateHandler(names: names, shape: [1, 4])
            var views = InferenceFunction.MutableViews()
            handler.bind(into: &views)
        }
    }

    @Test("preserves state data through bind")
    func preservesData() {
        let handler = MockStateHandler(names: ["s0"], shape: [1, 4], scalarType: .float32)
        var state = handler[stateIndex: 0]
        fillNDArray(&state.array, as: Float.self, count: 4) { Float($0 + 1) }
        handler[stateIndex: 0] = state

        var views = InferenceFunction.MutableViews()
        handler.bind(into: &views)

        let values = readNDArray(handler[stateIndex: 0].array, as: Float.self, count: 4)
        #expect(values == [1.0, 2.0, 3.0, 4.0])
    }

    @Test("multiple handlers compose into single MutableViews")
    func composesHandlers() {
        let primary = MockStateHandler(names: ["kv0", "kv1"], shape: [1, 4])
        let secondary = MockStateHandler(names: ["conv"], shape: [1, 4])
        var views = InferenceFunction.MutableViews()
        primary.bind(into: &views)
        secondary.bind(into: &views)
    }
}

// MARK: - Per-row blit (copyRegion)

@Suite("GrowingNDArrayState.copyRegion (per-row blit)")
struct CopyRegionTests {
    /// Row-major flat index into a [B, H, S, D] cache.
    private func idx(_ b: Int, _ h: Int, _ s: Int, _ d: Int, H: Int, S: Int, D: Int) -> Int {
        ((b * H + h) * S + s) * D + d
    }

    @Test("blits one row's seq slice (float16), leaving every other row/position untouched")
    func blitsFloat16() {
        let (B, H, S, D) = (3, 2, 6, 4)
        let count = B * H * S * D
        var array = NDArray(shape: [B, H, S, D], scalarType: .float16)
        fillNDArray(&array, as: Float16.self, count: count) { Float16($0) }

        // Row 1's new token was shared-written at the longest slot (seq 5); blit it to its cursor (2).
        GrowingNDArrayState.copyRegion(
            in: &array, sequenceDimIndex: 2, batchDimIndex: 0, row: 1, fromSeq: 5, toSeq: 2)

        let out = readNDArray(array, as: Float16.self, count: count)
        for b in 0..<B {
            for h in 0..<H {
                for s in 0..<S {
                    for d in 0..<D {
                        let flat = idx(b, h, s, d, H: H, S: S, D: D)
                        let expected =
                            (b == 1 && s == 2)
                            ? Float16(idx(1, h, 5, d, H: H, S: S, D: D)) : Float16(flat)
                        #expect(out[flat] == expected, "mismatch at [\(b),\(h),\(s),\(d)]")
                    }
                }
            }
        }
    }

    @Test("blits one row's seq slice (float32)")
    func blitsFloat32() {
        let (B, H, S, D) = (2, 1, 5, 3)
        let count = B * H * S * D
        var array = NDArray(shape: [B, H, S, D], scalarType: .float32)
        fillNDArray(&array, as: Float.self, count: count) { Float($0) }

        GrowingNDArrayState.copyRegion(
            in: &array, sequenceDimIndex: 2, batchDimIndex: 0, row: 0, fromSeq: 4, toSeq: 1)

        let out = readNDArray(array, as: Float.self, count: count)
        for b in 0..<B {
            for h in 0..<H {
                for s in 0..<S {
                    for d in 0..<D {
                        let flat = idx(b, h, s, d, H: H, S: S, D: D)
                        let expected =
                            (b == 0 && s == 1)
                            ? Float(idx(0, h, 4, d, H: H, S: S, D: D)) : Float(flat)
                        #expect(out[flat] == expected, "mismatch at [\(b),\(h),\(s),\(d)]")
                    }
                }
            }
        }
    }

    @Test("fromSeq == toSeq is a no-op (the longest row / equal-length cohort → zero blits)")
    func equalSlotIsNoOp() {
        let (B, H, S, D) = (2, 2, 4, 2)
        let count = B * H * S * D
        var array = NDArray(shape: [B, H, S, D], scalarType: .float16)
        fillNDArray(&array, as: Float16.self, count: count) { Float16($0) }

        GrowingNDArrayState.copyRegion(
            in: &array, sequenceDimIndex: 2, batchDimIndex: 0, row: 1, fromSeq: 3, toSeq: 3)

        let out = readNDArray(array, as: Float16.self, count: count)
        #expect(out == (0..<count).map { Float16($0) })
    }

    @Test("copyRowPrefix moves a batch-1 prefill into a slab across differing seq capacities")
    func copyRowPrefixIntoSlab() {
        // Source: batch-1 scratch [1, H, srcSeq, D]; destination: batch-N [N, H, dstSeq, D] with a
        // larger seq capacity. Prefill prefix of length P goes into destination slab `destRow`.
        let (H, D) = (2, 2)
        let (srcSeq, dstSeq, N, destRow, P) = (3, 5, 3, 1, 3)
        func sidx(_ h: Int, _ s: Int, _ d: Int) -> Int { ((0 * H + h) * srcSeq + s) * D + d }
        func didx(_ b: Int, _ h: Int, _ s: Int, _ d: Int) -> Int { ((b * H + h) * dstSeq + s) * D + d }

        var src = NDArray(shape: [1, H, srcSeq, D], scalarType: .float16)
        fillNDArray(&src, as: Float16.self, count: H * srcSeq * D) { Float16(100 + $0) }
        var dst = NDArray(shape: [N, H, dstSeq, D], scalarType: .float16)
        let dstCount = N * H * dstSeq * D
        fillNDArray(&dst, as: Float16.self, count: dstCount) { Float16($0) }

        GrowingNDArrayState.copyRowPrefix(
            from: src, sourceRow: 0, to: &dst, destRow: destRow, sequenceDimIndex: 2,
            batchDimIndex: 0, count: P)

        let out = readNDArray(dst, as: Float16.self, count: dstCount)
        for b in 0..<N {
            for h in 0..<H {
                for s in 0..<dstSeq {
                    for d in 0..<D {
                        let flat = didx(b, h, s, d)
                        let expected =
                            (b == destRow && s < P)
                            ? Float16(100 + sidx(h, s, d)) : Float16(flat)
                        #expect(out[flat] == expected, "mismatch at [\(b),\(h),\(s),\(d)]")
                    }
                }
            }
        }
    }

    // The real qwen3 KV layout is [G, B, H, seq, D] — batch at dim 1, NOT leading. These lock the
    // generalized blit to a non-leading batch dim; a leading-batch assumption crashes on device.

    @Test("copyRegion blits one row with a NON-leading batch dim ([G,B,H,S,D], batchDimIndex 1)")
    func blitNonLeadingBatchDim() {
        let (G, B, H, S, D) = (2, 3, 2, 5, 2)
        let count = G * B * H * S * D
        func idx(_ g: Int, _ b: Int, _ h: Int, _ s: Int, _ d: Int) -> Int {
            (((g * B + b) * H + h) * S + s) * D + d
        }
        var array = NDArray(shape: [G, B, H, S, D], scalarType: .float16)
        fillNDArray(&array, as: Float16.self, count: count) { Float16($0) }

        // Blit batch row 1's new token from the shared write slot (seq 4) down to its cursor (seq 1).
        GrowingNDArrayState.copyRegion(
            in: &array, sequenceDimIndex: 3, batchDimIndex: 1, row: 1, fromSeq: 4, toSeq: 1)

        let out = readNDArray(array, as: Float16.self, count: count)
        for g in 0..<G {
            for b in 0..<B {
                for h in 0..<H {
                    for s in 0..<S {
                        for d in 0..<D {
                            let flat = idx(g, b, h, s, d)
                            let expected =
                                (b == 1 && s == 1) ? Float16(idx(g, 1, h, 4, d)) : Float16(flat)
                            #expect(out[flat] == expected, "mismatch at [\(g),\(b),\(h),\(s),\(d)]")
                        }
                    }
                }
            }
        }
    }

    @Test("copyRowPrefix into a slab with a NON-leading batch dim ([G,B,H,S,D], batchDimIndex 1)")
    func prefixNonLeadingBatchDim() {
        let (G, H, D) = (2, 2, 2)
        let (srcSeq, dstSeq, N, destRow, P) = (3, 5, 3, 1, 3)
        func sidx(_ g: Int, _ h: Int, _ s: Int, _ d: Int) -> Int {
            (((g * 1 + 0) * H + h) * srcSeq + s) * D + d
        }
        func didx(_ g: Int, _ b: Int, _ h: Int, _ s: Int, _ d: Int) -> Int {
            (((g * N + b) * H + h) * dstSeq + s) * D + d
        }
        var src = NDArray(shape: [G, 1, H, srcSeq, D], scalarType: .float16)
        fillNDArray(&src, as: Float16.self, count: G * H * srcSeq * D) { Float16(100 + $0) }
        var dst = NDArray(shape: [G, N, H, dstSeq, D], scalarType: .float16)
        let dstCount = G * N * H * dstSeq * D
        fillNDArray(&dst, as: Float16.self, count: dstCount) { Float16($0) }

        GrowingNDArrayState.copyRowPrefix(
            from: src, sourceRow: 0, to: &dst, destRow: destRow, sequenceDimIndex: 3,
            batchDimIndex: 1, count: P)

        let out = readNDArray(dst, as: Float16.self, count: dstCount)
        for g in 0..<G {
            for b in 0..<N {
                for h in 0..<H {
                    for s in 0..<dstSeq {
                        for d in 0..<D {
                            let flat = didx(g, b, h, s, d)
                            let expected =
                                (b == destRow && s < P)
                                ? Float16(100 + sidx(g, h, s, d)) : Float16(flat)
                            #expect(out[flat] == expected, "mismatch at [\(g),\(b),\(h),\(s),\(d)]")
                        }
                    }
                }
            }
        }
    }
}
