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

// MARK: - Non-Float State Tests

/// 8-bit KV caches (and any other byte-addressable state) go through the same raw-byte
/// zero-fill and grow copy as fp16 states.
@Suite("Non-Float State Tests")
struct NonFloatStateTests {
    @Test(
        "Zero-fills non-float states",
        arguments: [NDArray.ScalarType.int8, .uint8, .float8e4m3fn, .int32])
    func zeroFill(_ type: NDArray.ScalarType) throws {
        var array = NDArray(shape: [2, 3, 4], scalarType: type)
        let byteCount = 24 * (try #require(stateElementByteWidth(type)))
        array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
            memset(ptr, 0x5A, byteCount)
        }
        zeroFillNDArray(&array)
        array.rawView().withUnsafeBytes { ptr, _, _ in
            let bytes = UnsafeRawBufferPointer(start: ptr, count: byteCount)
            #expect(bytes.allSatisfy { $0 == 0 })
        }
    }

    @Test("Copies block prefixes of an 8-bit state byte for byte")
    func copyBlockPrefixesInt8() {
        // Two blocks of 4 slots grown to 8 slots: the first 3 elements of each block move.
        var source = NDArray(shape: [2, 4], scalarType: .int8)
        source.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
            let dst = ptr.assumingMemoryBound(to: Int8.self)
            for i in 0..<8 { dst[i] = Int8(i + 1) }
        }
        var destination = NDArray(shape: [2, 8], scalarType: .int8)
        zeroFillNDArray(&destination)
        copyBlockPrefixes(
            from: source, to: &destination, blockCount: 2, sourceBlockStride: 4,
            destinationBlockStride: 8, runElements: 3)
        destination.rawView().withUnsafeBytes { ptr, _, _ in
            let got = Array(UnsafeBufferPointer(start: ptr.assumingMemoryBound(to: Int8.self), count: 16))
            #expect(got == [1, 2, 3, 0, 0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 0])
        }
    }

    @Test("Packed sub-byte types have no element width")
    func subByteWidth() {
        #expect(stateElementByteWidth(.int4) == nil)
        #expect(stateElementByteWidth(.int8) == 1)
        #expect(stateElementByteWidth(.float16) == 2)
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

// MARK: - State Classification Tests

@Suite("StateHandlerFactory.inferKind Tests")
struct InferKindTests {
    @Test(
        "states classify by shape and name",
        arguments: [
            ("keyCache", [28, 1, 8, -1, 128], StateKind.kvCache),
            // A 1024-context export pins the cache at the full context: still a KV cache.
            ("valueCache", [28, 1, 8, 1024, 128], .kvCache),
            ("slidingKeyCache", [20, 1, 8, 512, 128], .slidingCache),
            ("convState", [24, 1, 4, 2048], .fixed),
        ])
    func classify(state: (String, [Int], StateKind)) {
        #expect(StateHandlerFactory.inferKind(name: state.0, shape: state.1) == state.2)
    }
}

// MARK: - Sliding Ring Write-First Guard Tests

@Suite("Sliding ring write-first guard Tests")
struct WriteFirstGuardTests {
    @Test("ring capacity is the smallest sliding sequence dim")
    func ringCapacity() {
        #expect(StateHandlerFactory.ringCapacity(shapes: []) == nil)
        #expect(
            StateHandlerFactory.ringCapacity(shapes: [[20, 1, 8, 512, 128], [20, 1, 8, 256, 128]]) == 256)
    }

    @Test(
        "a step evicts only with more than one token ending past the ring",
        arguments: [
            (10_000, 1, false),  // single token: never
            (0, 512, false),  // exactly fills the ring
            (0, 513, true),  // one past
            (512, 2, true),
        ])
    func evicts(step: (offset: Int, queryLength: Int, evicts: Bool)) {
        #expect(
            StateHandlerFactory.writeFirstStepEvicts(
                offset: step.offset, queryLength: step.queryLength, ringCapacity: 512) == step.evicts)
    }

    @Test("check throws only with a ring and an evicting step")
    func check() throws {
        try StateHandlerFactory.checkWriteFirstStep(offset: 4096, queryLength: 64, ringCapacity: nil)
        try StateHandlerFactory.checkWriteFirstStep(offset: 0, queryLength: 64, ringCapacity: 512)
        #expect(throws: InferenceRuntimeError.self) {
            try StateHandlerFactory.checkWriteFirstStep(offset: 480, queryLength: 64, ringCapacity: 512)
        }
    }
}

@Suite("Sliding ring rewind Tests")
struct RingRewindTests {
    @Test("without a ring any rewind is safe; after a wrap the one-token re-feed is too")
    func examples() {
        #expect(StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 10, ringCapacity: nil))
        // The `commonPrefix -= 1` re-feed rewrites the newest position in its own slot.
        #expect(StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 4999, ringCapacity: 1024))
        #expect(!StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 4998, ringCapacity: 1024))
    }

    @Test("matches a slot-by-slot simulation")
    func simulation() {
        let ring = 8
        for processed in 0...24 {
            for target in 0...processed {
                // Slot s holds the newest position p < processed with p % ring == s.
                var slot = [Int](repeating: -1, count: ring)
                for p in 0..<processed { slot[p % ring] = p }
                // The query at `target` reads positions [target - ring + 1, target).
                let readsOK = (max(0, target - ring + 1)..<target).allSatisfy { slot[$0 % ring] == $0 }
                #expect(
                    StateHandlerFactory.ringRewindIsSafe(from: processed, to: target, ringCapacity: ring)
                        == readsOK, "processed=\(processed) target=\(target)")
            }
        }
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
    /// Layout `[layers, batch, kv_heads, seq, head_dim]`: the seq dim grows to `sequenceLength`
    /// and a dynamic batch dim resolves to `batchSize`.
    @Test(
        "resolves the seq dim and a dynamic batch dim",
        arguments: [
            ([2, -1, 4, -1, 8], 3, [2, 3, 4, 256, 8]),
            ([2, 1, 4, -1, 8], 1, [2, 1, 4, 256, 8]),
            ([2, -1, 4, -1, 8], 1, [2, 1, 4, 256, 8]),
        ] as [([Int], Int, [Int])])
    func resolves(shape: [Int], batchSize: Int, resolved: [Int]) {
        #expect(
            GrowingNDArrayState.resolveKVShape(
                shape, sequenceDimIndex: 3, sequenceLength: 256, batchSize: batchSize) == resolved)
    }
}

// MARK: - withBoundStates Tests

/// Minimal state handler for testing the binding API.
@Suite("State memory estimates")
struct StateMemoryEstimateTests {
    @Test(
        "A handler's estimate is element count times element size, summed over its states",
        arguments: [
            (NDArray.ScalarType.float16, 2 * 2 * 4 * 8 * 16 * 2),
            (.float32, 2 * 2 * 4 * 8 * 16 * 4),
            (.int8, 2 * 2 * 4 * 8 * 16),
        ] as [(NDArray.ScalarType, Int)])
    func handlerEstimate(scalarType: NDArray.ScalarType, expected: Int) {
        let handler = MockStateHandler(names: ["k", "v"], shape: [2, 4, 8, 16], scalarType: scalarType)
        #expect(handler.estimatedBytes == expected)
    }

    @Test("A session's estimate counts its KV cache, persistent states and logits buffer")
    func sessionEstimate() {
        let session = GenerationSessionState(
            kvCache: MockStateHandler(names: ["k", "v"], shape: [1, 4, 8, 16], scalarType: .float16),
            additionalStates: nil,
            hasNonTruncatableStates: false)
        #expect(session.estimatedStateBytes == 2 * 4 * 8 * 16 * 2)
        #expect(session.estimatedLogitsScratchBytes == 0)
        session.logitsScratch = NDArray(shape: [1, 1, 1000], scalarType: .float32)
        #expect(session.estimatedLogitsScratchBytes == 1000 * 4)
    }
}

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

// MARK: - State Kind Validation

@Suite("Explicit state kinds")
struct StateKindValidationTests {
    @Test("a dynamic-shaped state given a fixed-size kind is reported; valid kinds are not")
    func dynamicStateWithFixedSizeKind() {
        let isDynamic: (String) -> Bool = { $0 == "keyCache" || $0 == "valueCache" }
        let invalid: [(name: String, kind: StateKind)] = [
            ("keyCache", .slidingCache), ("valueCache", .fixed), ("convStates", .fixed),
        ]
        #expect(
            StateHandlerFactory.dynamicStatesWithFixedSizeKind(invalid, isDynamic: isDynamic)
                == ["keyCache", "valueCache"])
        let valid: [(name: String, kind: StateKind)] = [
            ("keyCache", .kvCache), ("valueCache", .kvCache), ("convStates", .fixed),
        ]
        #expect(StateHandlerFactory.dynamicStatesWithFixedSizeKind(valid, isDynamic: isDynamic).isEmpty)
    }
}

// MARK: - Per-row KV copies

/// Batch layouts the engine sees: batch leading (`[B, H, seq, D]`) and batch at dim 1
/// (`[G, B, H, seq, D]`), where code that assumes a leading batch dim copies the wrong slab.
enum RowCopyLayout: CaseIterable, Sendable {
    case leadingBatch, nonLeadingBatch

    var batchDimIndex: Int { self == .leadingBatch ? 0 : 1 }
    var sequenceDimIndex: Int { self == .leadingBatch ? 2 : 3 }

    /// Shape with `batch` rows and `seq` slots.
    func shape(batch: Int, seq: Int) -> [Int] {
        self == .leadingBatch ? [batch, 2, seq, 3] : [2, batch, 2, seq, 3]
    }
}

@Suite("Per-row KV copies")
struct RowCopyTests {
    /// An array whose bytes count up, so every element has a distinct bit pattern.
    private func countingArray(_ shape: [Int], _ type: NDArray.ScalarType, offset: Int = 0) -> NDArray {
        var array = NDArray(shape: shape, scalarType: type)
        let byteCount = shape.reduce(1, *) * type.byteSize
        array.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
            let bytes = ptr.assumingMemoryBound(to: UInt8.self)
            for i in 0..<byteCount { bytes[i] = UInt8((i + offset) % 251) }
        }
        return array
    }

    private func bytes(_ array: NDArray) -> [UInt8] {
        let byteCount = array.shape.reduce(1, *) * array.scalarType.byteSize
        var out = [UInt8](repeating: 0, count: byteCount)
        array.rawView().withUnsafeBytes { ptr, _, _ in
            let p = ptr.assumingMemoryBound(to: UInt8.self)
            for i in 0..<byteCount { out[i] = p[i] }
        }
        return out
    }

    /// Reference copy: every destination element for which `source(index)` returns a source index
    /// takes that source element's bytes.
    private func expected(
        destination: NDArray, from source: NDArray, source sourceIndex: ([Int]) -> [Int]?
    ) -> [UInt8] {
        var out = bytes(destination)
        let src = bytes(source)
        let width = destination.scalarType.byteSize
        func flat(_ index: [Int], _ shape: [Int]) -> Int {
            zip(index, shape).reduce(0) { $0 * $1.1 + $1.0 }
        }
        let shape = destination.shape
        for element in 0..<shape.reduce(1, *) {
            var index = [Int](repeating: 0, count: shape.count)
            var rest = element
            for dim in shape.indices.reversed() {
                index[dim] = rest % shape[dim]
                rest /= shape[dim]
            }
            guard let from = sourceIndex(index) else { continue }
            let s = flat(from, source.shape) * width
            out.replaceSubrange(element * width..<(element + 1) * width, with: src[s..<s + width])
        }
        return out
    }

    @Test(
        "copyRegion moves one row's slot and leaves everything else untouched",
        arguments: [NDArray.ScalarType.float16, .float32, .int8], RowCopyLayout.allCases)
    func copyRegion(_ type: NDArray.ScalarType, _ layout: RowCopyLayout) {
        var array = countingArray(layout.shape(batch: 3, seq: 6), type)
        let (b, seq) = (layout.batchDimIndex, layout.sequenceDimIndex)
        // Row 1's new token moves from the shared write slot 5 down to its cursor 2.
        let want = expected(destination: array, from: array) { index in
            guard index[b] == 1, index[seq] == 2 else { return nil }
            var from = index
            from[seq] = 5
            return from
        }

        GrowingNDArrayState.copyRegion(
            in: &array, sequenceDimIndex: seq, batchDimIndex: b, row: 1, fromSeq: 5, toSeq: 2)

        #expect(bytes(array) == want)
    }

    @Test("copyRegion with fromSeq == toSeq is a no-op")
    func copyRegionSameSlot() {
        var array = countingArray([2, 1, 4, 2], .float16)
        let before = bytes(array)
        GrowingNDArrayState.copyRegion(
            in: &array, sequenceDimIndex: 2, batchDimIndex: 0, row: 0, fromSeq: 3, toSeq: 3)
        #expect(bytes(array) == before)
    }

    @Test(
        "copyRowPrefix copies a batch-1 prefix into one row across different seq capacities",
        arguments: [NDArray.ScalarType.float16, .float32, .int8], RowCopyLayout.allCases)
    func copyRowPrefix(_ type: NDArray.ScalarType, _ layout: RowCopyLayout) {
        let source = countingArray(layout.shape(batch: 1, seq: 4), type, offset: 100)
        var destination = countingArray(layout.shape(batch: 3, seq: 6), type)
        let (b, seq) = (layout.batchDimIndex, layout.sequenceDimIndex)
        // Slots 0..<3 of row 2 come from the source; everything else keeps its bytes.
        let want = expected(destination: destination, from: source) { index in
            guard index[b] == 2, index[seq] < 3 else { return nil }
            var from = index
            from[b] = 0
            return from
        }

        GrowingNDArrayState.copyRowPrefix(
            from: source, sourceRow: 0, to: &destination, destRow: 2,
            sequenceDimIndex: seq, batchDimIndex: b, count: 3)

        #expect(bytes(destination) == want)
    }
}
