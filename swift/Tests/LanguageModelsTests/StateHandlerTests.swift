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
