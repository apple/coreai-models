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

// MARK: - State Classification Tests

@Suite("StateHandlerFactory.inferKind Tests")
struct InferKindTests {
    @Test("dynamic-dim caches are KV caches")
    func dynamicCache() {
        #expect(StateHandlerFactory.inferKind(name: "keyCache", shape: [28, 1, 8, -1, 128]) == .kvCache)
    }

    @Test("static full-context caches stay KV caches")
    func staticFullContextCache() {
        // A 1024-context export pins the cache at the full context.
        #expect(StateHandlerFactory.inferKind(name: "keyCache", shape: [28, 1, 8, 1024, 128]) == .kvCache)
        #expect(StateHandlerFactory.inferKind(name: "valueCache", shape: [28, 1, 8, 1024, 128]) == .kvCache)
    }

    @Test("static sliding caches are ring buffers")
    func staticSlidingCache() {
        #expect(
            StateHandlerFactory.inferKind(name: "slidingKeyCache", shape: [20, 1, 8, 512, 128]) == .slidingCache)
        #expect(
            StateHandlerFactory.inferKind(name: "slidingValueCache", shape: [20, 1, 8, 512, 128]) == .slidingCache)
    }

    @Test("other static states are fixed")
    func staticOther() {
        #expect(StateHandlerFactory.inferKind(name: "convState", shape: [24, 1, 4, 2048]) == .fixed)
    }
}

// MARK: - Sliding Ring Write-First Guard Tests

@Suite("Sliding ring write-first guard Tests")
struct WriteFirstGuardTests {
    @Test("ring capacity is the sequence dim of the sliding states")
    func ringCapacity() {
        #expect(StateHandlerFactory.ringCapacity(shapes: []) == nil)
        #expect(StateHandlerFactory.ringCapacity(shapes: [[20, 1, 8, 512, 128]]) == 512)
        #expect(
            StateHandlerFactory.ringCapacity(shapes: [[20, 1, 8, 512, 128], [20, 1, 8, 256, 128]]) == 256)
    }

    @Test("single-token steps never evict")
    func singleToken() {
        for offset in [0, 511, 512, 513, 10_000] {
            #expect(!StateHandlerFactory.writeFirstStepEvicts(offset: offset, queryLength: 1, ringCapacity: 512))
        }
    }

    @Test("multi-token steps that stay inside the ring are fine")
    func insideRing() {
        #expect(!StateHandlerFactory.writeFirstStepEvicts(offset: 0, queryLength: 100, ringCapacity: 512))
        // Exactly fills the ring: nothing is overwritten yet.
        #expect(!StateHandlerFactory.writeFirstStepEvicts(offset: 0, queryLength: 512, ringCapacity: 512))
        #expect(!StateHandlerFactory.writeFirstStepEvicts(offset: 500, queryLength: 12, ringCapacity: 512))
    }

    @Test("multi-token steps past the ring evict")
    func pastRing() {
        #expect(StateHandlerFactory.writeFirstStepEvicts(offset: 0, queryLength: 513, ringCapacity: 512))
        #expect(StateHandlerFactory.writeFirstStepEvicts(offset: 500, queryLength: 13, ringCapacity: 512))
        #expect(StateHandlerFactory.writeFirstStepEvicts(offset: 512, queryLength: 2, ringCapacity: 512))
    }

    @Test("check throws only with a ring and an evicting step")
    func check() throws {
        try StateHandlerFactory.checkWriteFirstStep(offset: 4096, queryLength: 64, ringCapacity: nil)
        try StateHandlerFactory.checkWriteFirstStep(offset: 0, queryLength: 64, ringCapacity: 512)
        #expect(throws: InferenceRuntimeError.self) {
            try StateHandlerFactory.checkWriteFirstStep(offset: 480, queryLength: 64, ringCapacity: 512)
        }
    }

    @Test("write-first steps fill up to the ring, then go one token at a time")
    func writeFirstStepSizes() {
        #expect(
            mainGraphStepSizes(tokenCount: 1300, chunkSize: 512, startOffset: 0, ringCapacity: 1024)
                == [512, 512] + Array(repeating: 1, count: 276))
        // Turn 2 starting mid-ring: the first step stops at the ring's last slot.
        #expect(
            mainGraphStepSizes(tokenCount: 10, chunkSize: 512, startOffset: 1020, ringCapacity: 1024)
                == [4, 1, 1, 1, 1, 1, 1])
        #expect(
            mainGraphStepSizes(tokenCount: 3, chunkSize: 512, startOffset: 4096, ringCapacity: 1024) == [1, 1, 1])
        #expect(
            mainGraphStepSizes(tokenCount: 0, chunkSize: 512, startOffset: 0, ringCapacity: 1024) == [])
    }

    @Test("write-first chunk plan never trips the guard")
    func writeFirstStepSizesPassGuard() {
        let ring = 64
        for start in [0, 1, 31, 63, 64, 65, 200] {
            for count in [1, 2, 63, 64, 65, 130] {
                for chunk in [1, 7, 64, 100] {
                    let plan = mainGraphStepSizes(
                        tokenCount: count, chunkSize: chunk, startOffset: start, ringCapacity: ring)
                    #expect(plan.reduce(0, +) == count)
                    var position = start
                    for size in plan {
                        #expect(size <= chunk)
                        #expect(
                            !StateHandlerFactory.writeFirstStepEvicts(
                                offset: position, queryLength: size, ringCapacity: ring))
                        position += size
                    }
                }
            }
        }
    }
}

@Suite("Sliding ring rewind Tests")
struct RingRewindTests {
    @Test("rewinds without a ring, before it wraps, or to 0 are safe")
    func safeRewinds() {
        #expect(StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 10, ringCapacity: nil))
        #expect(StateHandlerFactory.ringRewindIsSafe(from: 1024, to: 10, ringCapacity: 1024))
        #expect(StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 0, ringCapacity: 1024))
    }

    @Test("a one-token rewind after a wrap is safe")
    func oneTokenRewind() {
        // The `commonPrefix -= 1` re-feed rewrites the newest position in its own slot.
        #expect(StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 4999, ringCapacity: 1024))
        #expect(StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 5000, ringCapacity: 1024))
    }

    @Test("a longer rewind after a wrap is unsafe")
    func wrappedRewind() {
        #expect(!StateHandlerFactory.ringRewindIsSafe(from: 1025, to: 1023, ringCapacity: 1024))
        #expect(!StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 4998, ringCapacity: 1024))
        #expect(!StateHandlerFactory.ringRewindIsSafe(from: 5000, to: 1, ringCapacity: 1024))
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
