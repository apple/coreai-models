// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("StaticInputHandler Tests")
struct StaticInputHandlerTests {
    // MARK: - InputBuffers Basic Tests

    @Test("InputBuffers register and access")
    func buffersRegisterAccess() {
        var buffers = InputBuffers()
        // Can't use register() without NDArrayDescriptor, so test raw subscript
        var arr = NDArray(shape: [1, 5], scalarType: .int32)
        fillNDArray(&arr, as: Int32.self, with: [10, 20, 30, 40, 50])
        buffers["input_ids"] = arr

        #expect(buffers["input_ids"] != nil)
        #expect(buffers["nonexistent"] == nil)

        let readBack = readNDArray(buffers["input_ids"]!, as: Int32.self, count: 5)
        #expect(readBack == [10, 20, 30, 40, 50])
    }

    @Test("InputBuffers withBuffer fills in-place")
    func buffersWithBuffer() {
        var buffers = InputBuffers()
        buffers["test"] = NDArray(shape: [1, 4], scalarType: .int32)

        buffers.withBuffer("test") { array in
            fillNDArray(&array, as: Int32.self, with: [1, 2, 3, 4])
        }

        let result = readNDArray(buffers["test"]!, as: Int32.self, count: 4)
        #expect(result == [1, 2, 3, 4])
    }

    @Test("InputBuffers borrowedInputs returns all entries")
    func buffersBorrowedInputs() {
        var buffers = InputBuffers()
        buffers["a"] = NDArray(shape: [1, 2], scalarType: .int32)
        buffers["b"] = NDArray(shape: [1, 3], scalarType: .int32)

        let dict = buffers.borrowedInputs()
        #expect(dict.count == 2)
        #expect(dict["a"] != nil)
        #expect(dict["b"] != nil)
    }

    @Test("InputBuffers repeated fill reuses allocation (no shape change)")
    func buffersReusesAllocation() {
        var buffers = InputBuffers()
        buffers["pos"] = NDArray(shape: [1, 10], scalarType: .int32)

        // Fill multiple times — same shape, should be stable
        for i in 0..<100 {
            buffers.withBuffer("pos") { array in
                fillNDArray(&array, as: Int32.self, count: 10) { Int32(i + $0) }
            }
        }

        let result = readNDArray(buffers["pos"]!, as: Int32.self, count: 10)
        #expect(result == [99, 100, 101, 102, 103, 104, 105, 106, 107, 108])
    }

    // MARK: - Benchmarks

    @Test("Benchmark: InputBuffers fill vs NDArray allocation (decode step)")
    func benchmarkDecodeStep() {
        let iterations = 5000

        // --- Old path: allocate NDArray each step ---
        var oldResults: [[Int32]] = []
        let oldStart = ContinuousClock.now
        for i in 0..<iterations {
            let totalPos = i + 1
            var posIds = NDArray(shape: [1, totalPos], scalarType: .int32)
            fillNDArray(&posIds, as: Int32.self, count: totalPos) { Int32($0) }
            oldResults.append([Int32(totalPos)])  // prevent optimization
        }
        let oldElapsed = ContinuousClock.now - oldStart
        let oldUs =
            (Double(oldElapsed.components.attoseconds) / 1e12
                + Double(oldElapsed.components.seconds) * 1e6) / Double(iterations)

        // --- New path: reuse pre-allocated buffer ---
        var buffers = InputBuffers()
        // Pre-allocate at max expected size
        buffers["position_ids"] = NDArray(shape: [1, iterations + 1], scalarType: .int32)

        var newResults: [[Int32]] = []
        let newStart = ContinuousClock.now
        for i in 0..<iterations {
            let totalPos = i + 1
            buffers.withBuffer("position_ids") { array in
                fillNDArray(&array, as: Int32.self, count: totalPos) { Int32($0) }
            }
            newResults.append([Int32(totalPos)])
        }
        let newElapsed = ContinuousClock.now - newStart
        let newUs =
            (Double(newElapsed.components.attoseconds) / 1e12
                + Double(newElapsed.components.seconds) * 1e6) / Double(iterations)

        let speedup = oldUs / newUs
        print("  old (alloc each step): \(String(format: "%.1f", oldUs))us/call")
        print("  new (reuse buffer):    \(String(format: "%.1f", newUs))us/call")
        print("  speedup:               \(String(format: "%.1f", speedup))x")
        #expect(oldResults.count == newResults.count)  // prevent dead code elimination
    }

    @Test("Benchmark: InputBuffers fill vs NDArray allocation (prefill 512 tokens)")
    func benchmarkPrefill() {
        let iterations = 1000
        let batchSize = 512
        let tokens: [Int32] = Array(0..<Int32(batchSize))

        // --- Old path ---
        let oldStart = ContinuousClock.now
        for _ in 0..<iterations {
            var inputIds = NDArray(shape: [1, batchSize], scalarType: .int32)
            fillNDArray(&inputIds, as: Int32.self, with: tokens)
            var posIds = NDArray(shape: [1, batchSize], scalarType: .int32)
            fillNDArray(&posIds, as: Int32.self, count: batchSize) { Int32($0) }
            _ = (inputIds, posIds)  // prevent optimization
        }
        let oldUs =
            (Double((ContinuousClock.now - oldStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - oldStart).components.seconds) * 1e6) / Double(iterations)

        // --- New path ---
        var buffers = InputBuffers()
        buffers["input_ids"] = NDArray(shape: [1, batchSize], scalarType: .int32)
        buffers["position_ids"] = NDArray(shape: [1, batchSize], scalarType: .int32)

        let newStart = ContinuousClock.now
        for _ in 0..<iterations {
            buffers.withBuffer("input_ids") { array in
                fillNDArray(&array, as: Int32.self, with: tokens)
            }
            buffers.withBuffer("position_ids") { array in
                fillNDArray(&array, as: Int32.self, count: batchSize) { Int32($0) }
            }
        }
        let newUs =
            (Double((ContinuousClock.now - newStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - newStart).components.seconds) * 1e6) / Double(iterations)

        print("  old (alloc, 512 tok): \(String(format: "%.1f", oldUs))us/call")
        print("  new (reuse, 512 tok): \(String(format: "%.1f", newUs))us/call")
        print("  speedup:              \(String(format: "%.1f", oldUs / newUs))x")
    }
}

// MARK: - Detailed Benchmarks (matching RFC Performance Comparison table)

extension StaticInputHandlerTests {
    @Test("Benchmark: Position IDs — alloc+fill vs fill-only")
    func benchmarkPositionIdsIsolated() {
        let iterations = 5000
        let seqLen = 128  // typical decode position

        // --- Alloc + fill (current engine behavior) ---
        let allocStart = ContinuousClock.now
        for i in 0..<iterations {
            let total = seqLen + i % 100  // vary slightly to prevent optimization
            var arr = NDArray(shape: [1, total], scalarType: .int32)
            fillNDArray(&arr, as: Int32.self, count: total) { Int32($0) }
        }
        let allocUs =
            (Double((ContinuousClock.now - allocStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - allocStart).components.seconds) * 1e6) / Double(iterations)

        // --- Fill-only (pre-allocated buffer, same shape) ---
        var buffer = NDArray(shape: [1, seqLen + 100], scalarType: .int32)
        let fillStart = ContinuousClock.now
        for i in 0..<iterations {
            let total = seqLen + i % 100
            fillNDArray(&buffer, as: Int32.self, count: total) { Int32($0) }
        }
        let fillUs =
            (Double((ContinuousClock.now - fillStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - fillStart).components.seconds) * 1e6) / Double(iterations)

        // --- InputBuffers path ---
        var buffers = InputBuffers()
        buffers["pos"] = NDArray(shape: [1, seqLen + 100], scalarType: .int32)
        let ibStart = ContinuousClock.now
        for i in 0..<iterations {
            let total = seqLen + i % 100
            buffers.withBuffer("pos") { arr in
                fillNDArray(&arr, as: Int32.self, count: total) { Int32($0) }
            }
        }
        let ibUs =
            (Double((ContinuousClock.now - ibStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - ibStart).components.seconds) * 1e6) / Double(iterations)

        print("  Position IDs (seqLen=\(seqLen)):")
        print("    alloc + fill:     \(String(format: "%.1f", allocUs))us")
        print("    fill only (raw):  \(String(format: "%.1f", fillUs))us")
        print("    InputBuffers:     \(String(format: "%.1f", ibUs))us")
        print("    speedup (alloc→IB): \(String(format: "%.1f", allocUs / ibUs))x")
        print("    speedup (alloc→raw): \(String(format: "%.1f", allocUs / fillUs))x")
    }

    @Test("Benchmark: Step scalar — alloc+fill vs fill-only")
    func benchmarkStepScalar() {
        let iterations = 10000

        // --- Alloc + fill (scalar NDArray each step) ---
        let allocStart = ContinuousClock.now
        for i in 0..<iterations {
            var arr = NDArray(shape: [1], scalarType: .int32)
            fillNDArray(&arr, as: Int32.self, with: [Int32(i)])
        }
        let allocUs =
            (Double((ContinuousClock.now - allocStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - allocStart).components.seconds) * 1e6) / Double(iterations)

        // --- Fill-only ---
        var buffer = NDArray(shape: [1], scalarType: .int32)
        let fillStart = ContinuousClock.now
        for i in 0..<iterations {
            fillNDArray(&buffer, as: Int32.self, with: [Int32(i)])
        }
        let fillUs =
            (Double((ContinuousClock.now - fillStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - fillStart).components.seconds) * 1e6) / Double(iterations)

        // --- InputBuffers ---
        var buffers = InputBuffers()
        buffers["step"] = NDArray(shape: [1], scalarType: .int32)
        let ibStart = ContinuousClock.now
        for i in 0..<iterations {
            buffers.withBuffer("step") { arr in
                fillNDArray(&arr, as: Int32.self, with: [Int32(i)])
            }
        }
        let ibUs =
            (Double((ContinuousClock.now - ibStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - ibStart).components.seconds) * 1e6) / Double(iterations)

        print("  Step scalar:")
        print("    alloc + fill:     \(String(format: "%.1f", allocUs))us")
        print("    fill only (raw):  \(String(format: "%.1f", fillUs))us")
        print("    InputBuffers:     \(String(format: "%.1f", ibUs))us")
        print("    speedup (alloc→IB): \(String(format: "%.1f", allocUs / ibUs))x")
    }

    @Test("Benchmark: borrowedInputs() overhead")
    func benchmarkAsDict() {
        let iterations = 10000
        var buffers = InputBuffers()
        buffers["input_ids"] = NDArray(shape: [1, 1], scalarType: .int32)
        buffers["position_ids"] = NDArray(shape: [1, 128], scalarType: .int32)

        let start = ContinuousClock.now
        for _ in 0..<iterations {
            let dict = buffers.borrowedInputs()
            _ = dict.count  // prevent optimization
        }
        let us =
            (Double((ContinuousClock.now - start).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - start).components.seconds) * 1e6) / Double(iterations)
        print("  borrowedInputs() (2 entries): \(String(format: "%.2f", us))us")
    }
}

extension StaticInputHandlerTests {
    @Test("Benchmark: _modify subscript vs withBuffer vs alloc")
    func benchmarkModifySubscript() {
        let iterations = 5000
        let seqLen = 128

        // --- Alloc each step (baseline) ---
        let allocStart = ContinuousClock.now
        for i in 0..<iterations {
            let total = seqLen + i % 100
            var arr = NDArray(shape: [1, total], scalarType: .int32)
            fillNDArray(&arr, as: Int32.self, count: total) { Int32($0) }
        }
        let allocUs =
            (Double((ContinuousClock.now - allocStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - allocStart).components.seconds) * 1e6) / Double(iterations)

        // --- withBuffer (closure, get/set) ---
        var buffers1 = InputBuffers()
        buffers1["pos"] = NDArray(shape: [1, seqLen + 100], scalarType: .int32)
        let wbStart = ContinuousClock.now
        for i in 0..<iterations {
            let total = seqLen + i % 100
            buffers1.withBuffer("pos") { arr in
                fillNDArray(&arr, as: Int32.self, count: total) { Int32($0) }
            }
        }
        let wbUs =
            (Double((ContinuousClock.now - wbStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - wbStart).components.seconds) * 1e6) / Double(iterations)

        // --- _modify subscript (direct inout, no closure) ---
        var buffers2 = InputBuffers()
        buffers2["pos"] = NDArray(shape: [1, seqLen + 100], scalarType: .int32)
        let modStart = ContinuousClock.now
        for i in 0..<iterations {
            let total = seqLen + i % 100
            fillNDArray(&buffers2["pos"]!, as: Int32.self, count: total) { Int32($0) }
        }
        let modUs =
            (Double((ContinuousClock.now - modStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - modStart).components.seconds) * 1e6) / Double(iterations)

        // --- MutableRawView subscript (consuming view, zero-copy) ---
        var buffers3 = InputBuffers()
        buffers3["pos"] = NDArray(shape: [1, seqLen + 100], scalarType: .int32)
        let viewStart = ContinuousClock.now
        for i in 0..<iterations {
            let total = seqLen + i % 100
            fillNDArray(buffers3[view: "pos"]!, as: Int32.self, count: total) { Int32($0) }
        }
        let viewUs =
            (Double((ContinuousClock.now - viewStart).components.attoseconds) / 1e12
                + Double((ContinuousClock.now - viewStart).components.seconds) * 1e6) / Double(iterations)

        print("  Position IDs (seq=\(seqLen), _modify benchmark):")
        print("    alloc + fill:       \(String(format: "%.1f", allocUs))us")
        print("    withBuffer:         \(String(format: "%.1f", wbUs))us")
        print("    _modify subscript:  \(String(format: "%.1f", modUs))us")
        print("    view subscript:     \(String(format: "%.1f", viewUs))us")
        print("    speedup (alloc→view): \(String(format: "%.1f", allocUs / viewUs))x")
        print("    speedup (mod→view):   \(String(format: "%.1f", modUs / viewUs))x")
    }
}
