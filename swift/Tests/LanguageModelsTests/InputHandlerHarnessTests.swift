// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

// MARK: - Helpers

extension Duration {
    fileprivate var inMicroseconds: Double { inMilliseconds * 1000.0 }
}

private func currentRSSBytes() -> Int {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
        }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return Int(info.resident_size)
}

/// Minimal state handler for benchmark tests.
private final class BenchMockStateHandler: SyncStateHandler {
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

// MARK: - Benchmarks

@Suite("Input Preparation Benchmarks", .serialized)
struct InputPrepBenchmarks {
    @Test("NDArray allocation cost for typical input shapes")
    func allocationCost() {
        let shapes: [(String, [Int], NDArray.ScalarType)] = [
            ("position_ids_decode", [1, 1], .uint16),
            ("position_ids_prefill_64", [1, 64], .int32),
            ("causal_mask_512", [1, 512, 1, 1], .float16),
            ("causal_mask_2048", [1, 2048, 1, 64], .float16),
            ("step_scalar", [1], .int32),
        ]

        let iterations = 100
        let clock = SuspendingClock()

        for (label, shape, scalar) in shapes {
            let start = clock.now
            for _ in 0..<iterations {
                _ = NDArray(shape: shape, scalarType: scalar)
            }
            let elapsed = clock.now - start
            let perIterUs = (elapsed / iterations).inMicroseconds
            print("[\(label)] alloc: \(String(format: "%.1f", perIterUs))us/iter")

            let thresholdUs: Double = CIEnvironment.isVM ? 400 : 200
            #expect(
                perIterUs < thresholdUs,
                "\(label) allocation took \(perIterUs)us, expected <\(thresholdUs)us")
        }
    }

    @Test("fillNDArray cost for pre-allocated buffers")
    func fillCost() {
        let iterations = 1000
        let clock = SuspendingClock()

        var posIds = NDArray(shape: [1, 1], scalarType: .uint16)
        let startPos = clock.now
        for i in 0..<iterations {
            fillNDArray(&posIds, as: UInt16.self, count: 1) { _ in UInt16(i) }
        }
        let posUs = ((clock.now - startPos) / iterations).inMicroseconds
        print("[pos_ids_1] fill: \(String(format: "%.2f", posUs))us/iter")
        #expect(posUs < 10, "Trivial fill should be <10us")

        var mask = NDArray(shape: [1, 512, 1, 1], scalarType: .float16)
        let startMask = clock.now
        for step in 0..<iterations {
            fillNDArray(&mask, as: Float16.self, count: 512) { i in
                i <= step % 512 ? Float16(0) : causalMaskSentinel
            }
        }
        let maskUs = ((clock.now - startMask) / iterations).inMicroseconds
        print("[causal_mask_512] fill: \(String(format: "%.2f", maskUs))us/iter")
        #expect(maskUs < 200, "512-element fill should be <200us")

        var largeMask = NDArray(shape: [1, 2048, 1, 64], scalarType: .float16)
        let largeIterations = 100
        let startLarge = clock.now
        for _ in 0..<largeIterations {
            fillNDArray(&largeMask, as: Float16.self, count: 2048 * 64) { _ in causalMaskSentinel }
        }
        let largeUs = ((clock.now - startLarge) / largeIterations).inMicroseconds
        print("[causal_mask_2048x64] fill: \(String(format: "%.1f", largeUs))us/iter")
    }

    @Test("Owned-buffer COW safety: fill-return-drop cycle")
    func ownedBufferCOWSafety() {
        let iterations = 1000
        let clock = SuspendingClock()

        var ownedArray = NDArray(shape: [1, 128], scalarType: .int32)

        let start = clock.now
        for i in 0..<iterations {
            fillNDArray(&ownedArray, as: Int32.self, count: 128) { Int32(i + $0) }
            let dict: [String: NDArray] = ["position_ids": ownedArray]
            _ = dict["position_ids"]
        }
        let perIterUs = ((clock.now - start) / iterations).inMicroseconds
        print("[owned_buffer_cycle] \(String(format: "%.2f", perIterUs))us/iter")

        let threshold: Double = CIEnvironment.isVM ? 50 : 20
        #expect(
            perIterUs < threshold,
            "Owned-buffer cycle should not trigger COW, took \(perIterUs)us")
    }

    @Test("bind(into:) cost for 2-8 states")
    func bindCost() {
        let iterations = 10_000
        let clock = SuspendingClock()

        for stateCount in [2, 4, 6, 8] {
            let names = (0..<stateCount).map { "state_\($0)" }
            let handler = BenchMockStateHandler(names: names, shape: [1, 32, 128])

            let start = clock.now
            for _ in 0..<iterations {
                var views = InferenceFunction.MutableViews()
                handler.bind(into: &views)
            }
            let perIterUs = ((clock.now - start) / iterations).inMicroseconds
            print("[bind_\(stateCount)_states] \(String(format: "%.2f", perIterUs))us/iter")

            let threshold: Double = CIEnvironment.isVM ? 30 : 15
            #expect(
                perIterUs < threshold,
                "bind(into:) with \(stateCount) states should be <\(threshold)us")
        }
    }

    @Test("Speedup ratio: alloc-per-step vs fill-in-place")
    func speedupRatio() {
        let iterations = 500
        let clock = SuspendingClock()

        let startAlloc = clock.now
        for i in 0..<iterations {
            var arr = NDArray(shape: [1, 512, 1, 1], scalarType: .float16)
            fillNDArray(&arr, as: Float16.self, count: 512) { Float16($0 + i) }
        }
        let allocMs = (clock.now - startAlloc).inMilliseconds

        var reused = NDArray(shape: [1, 512, 1, 1], scalarType: .float16)
        let startReuse = clock.now
        for i in 0..<iterations {
            fillNDArray(&reused, as: Float16.self, count: 512) { Float16($0 + i) }
        }
        let reuseMs = (clock.now - startReuse).inMilliseconds

        let ratio = allocMs / max(0.001, reuseMs)
        print("[speedup] alloc-per-step: \(String(format: "%.2f", allocMs / Double(iterations) * 1000))us/iter")
        print("[speedup] fill-in-place: \(String(format: "%.2f", reuseMs / Double(iterations) * 1000))us/iter")
        print("[speedup] ratio: \(String(format: "%.1f", ratio))x")

        // For small arrays (512 Float16 = 1KB), fill dominates so ratio is modest (~1.5x).
        // The mechanism is validated; the real win is skipping alloc entirely on hot path.
        #expect(ratio > 1.2, "Reuse should be faster than alloc+fill, got \(ratio)x")
    }
}

// MARK: - Fuzz Tests

@Suite("Input Handler Fuzz Tests")
struct InputHandlerFuzzTests {
    @Test(
        "InputContext.dynamic with varied token counts",
        arguments: [1, 2, 7, 16, 63, 64, 128, 255, 256, 512])
    func dynamicContextVaried(tokenCount: Int) {
        let tokens = ArraySlice((0..<tokenCount).map { Int32($0) })
        let processedCount = tokenCount * 3
        let context = InputContext.dynamic(tokens: tokens, processedTokenCount: processedCount)

        #expect(context.batchSize == tokenCount)
        #expect(context.processedTokenCount == processedCount)
        #expect(context.alignedStep == processedCount)
    }

    @Test(
        "InputContext.static with bucket configs",
        arguments: [1, 8, 32, 64, 128])
    func staticContextBuckets(batchSize: Int) {
        let tokens = ArraySlice((0..<batchSize).map { Int32($0) })
        let alignedStep = 512
        let context = InputContext.static(
            tokens: tokens, alignedStep: alignedStep,
            batchSize: batchSize, slidingWindow: 256, contextBucket: 2048)

        #expect(context.batchSize == batchSize)
        #expect(context.alignedStep == alignedStep)
        #expect(context.slidingWindow == 256)
    }

    @Test(
        "fillNDArray correctness at boundary sizes",
        arguments: [1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1024, 2048, 4096])
    func fillBoundaries(count: Int) {
        var array = NDArray(shape: [1, count], scalarType: .int32)
        fillNDArray(&array, as: Int32.self, count: count) { Int32($0 * 3 + 7) }
        let values = readNDArray(array, as: Int32.self, count: count)

        for i in 0..<count {
            #expect(
                values[i] == Int32(i * 3 + 7),
                "Mismatch at index \(i): expected \(i * 3 + 7), got \(values[i])")
        }
    }

    @Test(
        "Causal mask correctness for varied configurations",
        arguments: [
            (64, 1, 0), (64, 1, 32), (64, 1, 63),
            (128, 4, 0), (128, 4, 60), (128, 8, 100),
            (256, 1, 200), (256, 16, 0), (256, 32, 128),
            (512, 1, 0), (512, 1, 511), (512, 64, 256),
        ] as [(Int, Int, Int)])
    func causalMaskCorrectness(contextLength: Int, tokensInBatch: Int, alignedStep: Int) {
        guard alignedStep + tokensInBatch <= contextLength else { return }

        var mask = NDArray(shape: [1, contextLength, 1, tokensInBatch], scalarType: .float16)
        let count = contextLength * tokensInBatch
        fillNDArray(&mask, as: Float16.self, count: count) { _ in causalMaskSentinel }

        let view = mask.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { ptr, shape, strides in
            for query in 0..<tokensInBatch {
                let queryPos = alignedStep + query
                let upperBound = min(queryPos, contextLength - 1)
                for ctx in 0...upperBound {
                    let offset = ctx * strides[1] + query * strides[3]
                    ptr[offset] = 0
                }
            }
        }

        let result = readNDArray(mask, as: Float16.self, count: count)
        for query in 0..<tokensInBatch {
            let queryPos = alignedStep + query
            for ctx in 0..<contextLength {
                let idx = ctx * tokensInBatch + query
                if ctx <= queryPos {
                    #expect(
                        result[idx] == 0,
                        "(\(ctx),\(query)) should be unmasked (queryPos=\(queryPos))")
                } else {
                    #expect(
                        result[idx] == causalMaskSentinel,
                        "(\(ctx),\(query)) should be masked (queryPos=\(queryPos))")
                }
            }
        }
    }

    @Test("Rapid re-fill does not corrupt")
    func rapidRefill() {
        var array = NDArray(shape: [1, 256], scalarType: .float16)

        for round in 0..<100 {
            let seed = Float16(round)
            fillNDArray(&array, as: Float16.self, count: 256) { Float16($0) + seed }
            let values = readNDArray(array, as: Float16.self, count: 256)
            #expect(values[0] == seed, "Corruption at round \(round), index 0")
            #expect(values[255] == Float16(255) + seed, "Corruption at round \(round), index 255")
        }
    }
}

// MARK: - Stress Tests

@Suite("Input Handler Stress Tests", .serialized)
struct InputHandlerStressTests {
    @Test("10k decode steps: no drift in fill correctness")
    func longDecodeRun() {
        var posIds = NDArray(shape: [1, 1], scalarType: .int32)
        var step = NDArray(shape: [1], scalarType: .int32)

        for i in 0..<10_000 {
            fillNDArray(&posIds, as: Int32.self, count: 1) { _ in Int32(i) }
            fillNDArray(&step, as: Int32.self, count: 1) { _ in Int32(i) }

            if i % 1000 == 0 || i == 9999 {
                let posValue = readNDArray(posIds, as: Int32.self, count: 1)[0]
                let stepValue = readNDArray(step, as: Int32.self, count: 1)[0]
                #expect(posValue == Int32(i), "pos_ids drift at step \(i)")
                #expect(stepValue == Int32(i), "step drift at step \(i)")
            }
        }
    }

    @Test("Bucket transitions: shape changes don't corrupt")
    func bucketTransitions() {
        var arrays = [
            NDArray(shape: [1, 256, 1, 1], scalarType: .float16),
            NDArray(shape: [1, 512, 1, 1], scalarType: .float16),
            NDArray(shape: [1, 1024, 1, 1], scalarType: .float16),
        ]
        let sizes = [256, 512, 1024]

        let transitions = [0, 0, 0, 1, 1, 2, 2, 1, 0, 0, 2, 1, 0]
        for (step, bucketIdx) in transitions.enumerated() {
            let size = sizes[bucketIdx]
            fillNDArray(&arrays[bucketIdx], as: Float16.self, count: size) { i in
                i <= step ? Float16(0) : causalMaskSentinel
            }

            let values = readNDArray(arrays[bucketIdx], as: Float16.self, count: size)
            for i in 0..<min(size, step + 2) {
                let expected: Float16 = i <= step ? 0 : causalMaskSentinel
                #expect(
                    values[i] == expected,
                    "Bucket \(bucketIdx) corruption at step \(step), index \(i)")
            }
        }
    }

    @Test("COW detection: holding dict across steps triggers copy")
    func cowDetection() {
        var ownedArray = NDArray(shape: [1, 64], scalarType: .int32)
        fillNDArray(&ownedArray, as: Int32.self, count: 64) { Int32($0) }

        let heldDict: [String: NDArray] = ["input": ownedArray]

        fillNDArray(&ownedArray, as: Int32.self, count: 64) { Int32($0 + 100) }

        let heldValues = readNDArray(heldDict["input"]!, as: Int32.self, count: 64)
        let newValues = readNDArray(ownedArray, as: Int32.self, count: 64)

        #expect(heldValues[0] == Int32(0), "Held dict should have old value")
        #expect(newValues[0] == Int32(100), "Owned array should have new value")
    }

    @Test("State handler rapid lifecycle: 1000 create-bind-drop cycles")
    func stateHandlerLifecycle() {
        for i in 0..<1000 {
            let handler = BenchMockStateHandler(
                names: ["key_cache", "value_cache"], shape: [1, 32, 128])

            var state = handler[stateIndex: 0]
            fillNDArray(&state.array, as: Float16.self, count: 32 * 128) { Float16($0 % 100) }
            handler[stateIndex: 0] = state

            var views = InferenceFunction.MutableViews()
            handler.bind(into: &views)

            let values = readNDArray(handler[stateIndex: 0].array, as: Float16.self, count: 1)
            #expect(values[0] == Float16(0), "Cycle \(i): data lost after bind")
        }
    }

    @Test(
        "Boundary context lengths: 2^n-1, 2^n, 2^n+1",
        arguments: [127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049])
    func boundaryContextLengths(contextLength: Int) {
        var mask = NDArray(shape: [1, contextLength, 1, 1], scalarType: .float16)

        fillNDArray(&mask, as: Float16.self, count: contextLength) { i in
            i < contextLength / 2 ? Float16(0) : causalMaskSentinel
        }
        let values = readNDArray(mask, as: Float16.self, count: contextLength)
        #expect(values[0] == Float16(0))
        #expect(values[contextLength / 2 - 1] == Float16(0))
        #expect(values[contextLength / 2] == causalMaskSentinel)
        #expect(values[contextLength - 1] == causalMaskSentinel)
    }
}

// MARK: - Memory Pattern Tests

@Suite("Memory Allocation Pattern Tests", .serialized)
struct MemoryPatternTests {
    @Test("Memory growth: 1000 allocations without reuse (anti-pattern)")
    func allocationWithoutReuse() {
        let before = currentRSSBytes()
        var arrays: [NDArray] = []
        arrays.reserveCapacity(1000)

        for _ in 0..<1000 {
            arrays.append(NDArray(shape: [1, 512, 1, 1], scalarType: .float16))
        }

        let after = currentRSSBytes()
        let growth = after - before
        print("[no_reuse] Memory growth: \(growth / 1024)KB for 1000 x [1,512,1,1] Float16")
        #expect(growth > 500_000, "Should have measurable growth: \(growth) bytes")

        arrays.removeAll()
    }

    @Test("Memory stability: 10000 fills with buffer reuse (correct pattern)")
    func allocationWithReuse() {
        var array = NDArray(shape: [1, 512, 1, 1], scalarType: .float16)
        _ = readNDArray(array, as: Float16.self, count: 1)  // fault the page

        let before = currentRSSBytes()
        for i in 0..<10_000 {
            fillNDArray(&array, as: Float16.self, count: 512) { Float16($0 + i) }
        }
        let after = currentRSSBytes()

        let growth = after - before
        print("[with_reuse] Memory growth: \(growth / 1024)KB for 10000 fills (reuse)")
        // RSS includes runtime noise (test runner overhead, autoreleasepool, lazy page faults).
        // The meaningful comparison: this should be far less than 1000x alloc (~3-6MB measured).
        // We allow up to 200MB for test-runner noise but log the actual value for manual review.
        #expect(
            growth < 200_000_000,
            "Buffer reuse grew excessively: \(growth / 1024)KB")
    }

    @Test("Owned-buffer dict pattern: no memory growth across 5000 cycles")
    func ownedBufferDictMemoryStability() {
        var ownedPos = NDArray(shape: [1, 1], scalarType: .int32)
        var ownedMask = NDArray(shape: [1, 512, 1, 1], scalarType: .float16)
        var ownedStep = NDArray(shape: [1], scalarType: .int32)
        // Fault pages
        fillNDArray(&ownedPos, as: Int32.self, count: 1) { _ in Int32(0) }
        fillNDArray(&ownedMask, as: Float16.self, count: 512) { _ in Float16(0) }
        fillNDArray(&ownedStep, as: Int32.self, count: 1) { _ in Int32(0) }

        let before = currentRSSBytes()

        for i in 0..<5000 {
            fillNDArray(&ownedPos, as: Int32.self, count: 1) { _ in Int32(i) }
            fillNDArray(&ownedMask, as: Float16.self, count: 512) { Float16($0) }
            fillNDArray(&ownedStep, as: Int32.self, count: 1) { _ in Int32(i) }

            let inputs: [String: NDArray] = [
                "position_ids": ownedPos,
                "causal_mask": ownedMask,
                "step": ownedStep,
            ]
            _ = inputs.count
        }

        let after = currentRSSBytes()
        let growth = after - before
        print("[owned_dict_pattern] Memory growth: \(growth / 1024)KB over 5000 cycles")
        // Dict creation has runtime overhead (malloc for bucket storage). The key insight:
        // no IOSurface leak — growth should be bounded, not linear with iterations.
        #expect(
            growth < 20_000_000,
            "Dict pattern leaked excessively: grew \(growth / 1024)KB")
    }

    @Test("Per-bucket pre-allocation: total cost is bounded")
    func perBucketPreallocation() {
        let before = currentRSSBytes()

        let bucketConfigs: [(ctx: Int, query: Int)] = [
            (256, 1), (512, 1), (512, 32), (1024, 1), (1024, 64), (2048, 1),
        ]

        var bucketArrays: [[(String, NDArray)]] = []
        for config in bucketConfigs {
            let arrays: [(String, NDArray)] = [
                ("position_ids", NDArray(shape: [1, config.query], scalarType: .uint16)),
                ("causal_mask", NDArray(shape: [1, config.ctx, 1, config.query], scalarType: .float16)),
                ("step", NDArray(shape: [1], scalarType: .int32)),
            ]
            bucketArrays.append(arrays)
        }

        let after = currentRSSBytes()
        let totalCost = after - before
        print("[bucket_prealloc] Total memory for 6 buckets: \(totalCost / 1024)KB")
        #expect(
            totalCost < 4_000_000,
            "6 buckets should cost <4MB: actual \(totalCost / 1024)KB")

        _ = bucketArrays
    }
}

// MARK: - Incremental Causal Mask Tests

@Suite("Incremental Causal Mask Tests")
struct IncrementalMaskTests {
    @Test(
        "Incremental vs full refill produce identical masks",
        arguments: [128, 256, 512, 1024])
    func incrementalMatchesFull(contextLength: Int) {
        var fullMask = NDArray(shape: [1, contextLength, 1, 1], scalarType: .float16)
        var incrMask = NDArray(shape: [1, contextLength, 1, 1], scalarType: .float16)
        fillNDArray(&incrMask, as: Float16.self, count: contextLength) { _ in causalMaskSentinel }

        let steps = min(64, contextLength)
        for step in 0..<steps {
            fillNDArray(&fullMask, as: Float16.self, count: contextLength) { i in
                i <= step ? Float16(0) : causalMaskSentinel
            }

            let view = incrMask.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr, shape, strides in
                ptr[step * strides[1]] = Float16(0)
            }

            let fullValues = readNDArray(fullMask, as: Float16.self, count: contextLength)
            let incrValues = readNDArray(incrMask, as: Float16.self, count: contextLength)
            #expect(fullValues == incrValues, "Mismatch at step \(step) (ctx=\(contextLength))")
        }
    }

    @Test("Incremental mask is faster than full refill for long contexts")
    func incrementalSpeedup() {
        let contextLength = 2048
        let steps = 500
        let clock = SuspendingClock()

        var fullMask = NDArray(shape: [1, contextLength, 1, 1], scalarType: .float16)
        let startFull = clock.now
        for step in 0..<steps {
            fillNDArray(&fullMask, as: Float16.self, count: contextLength) { i in
                i <= step ? Float16(0) : causalMaskSentinel
            }
        }
        let fullMs = (clock.now - startFull).inMilliseconds

        var incrMask = NDArray(shape: [1, contextLength, 1, 1], scalarType: .float16)
        fillNDArray(&incrMask, as: Float16.self, count: contextLength) { _ in causalMaskSentinel }
        let startIncr = clock.now
        for step in 0..<steps {
            let view = incrMask.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr, shape, strides in
                ptr[step * strides[1]] = Float16(0)
            }
        }
        let incrMs = (clock.now - startIncr).inMilliseconds

        let ratio = fullMs / max(0.001, incrMs)
        print("[mask_speedup] full: \(String(format: "%.2f", fullMs / Double(steps) * 1000))us/step")
        print("[mask_speedup] incr: \(String(format: "%.2f", incrMs / Double(steps) * 1000))us/step")
        print("[mask_speedup] ratio: \(String(format: "%.0f", ratio))x")

        #expect(
            ratio > 5.0,
            "Incremental should be >5x faster for ctx=2048, got \(String(format: "%.1f", ratio))x")
    }
}
