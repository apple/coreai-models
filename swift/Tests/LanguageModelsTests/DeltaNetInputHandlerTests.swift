// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Update flag (the re-send gate)

@Suite("DeltaNetInputHandler update flag")
struct DeltaNetUpdateFlagTests {
    @Test("A fresh prefill marks every real slot new")
    func freshPrefill() {
        for slot in 0..<8 {
            #expect(
                DeltaNetInputHandler.isNewToken(
                    slot: slot, tokensInBatch: 8, alignedStep: 0, processedTokenCount: 0))
        }
    }

    @Test("Padding slots past the real token count are not new")
    func paddingIsNotNew() {
        // A 5-token tail in an 8-wide block: slots 5..7 are padding.
        for slot in 5..<8 {
            #expect(
                !DeltaNetInputHandler.isNewToken(
                    slot: slot, tokensInBatch: 5, alignedStep: 0, processedTokenCount: 0))
        }
    }

    @Test("Re-sent positions are not new, the appended one is")
    func resendGate() {
        // 21 tokens processed; the decode block's start aligns down to 16 and is 8 wide, so
        // slots 0..4 re-send positions 16..20 and slot 5 carries the new position 21.
        let alignedStep = 16
        let processed = 21
        let tokensInBatch = 6
        let isNew = (0..<8).map {
            DeltaNetInputHandler.isNewToken(
                slot: $0, tokensInBatch: tokensInBatch, alignedStep: alignedStep,
                processedTokenCount: processed)
        }
        #expect(isNew == [false, false, false, false, false, true, false, false])
    }

    @Test("Exactly one slot is new on a steady-state decode step")
    func steadyStateDecode() {
        // Each decode step advances `processed` by one; the block start only moves every
        // `batchSize` steps, so the number of re-sent slots grows then resets.
        for processed in 17...23 {
            let alignedStep = (processed / 8) * 8
            let tokensInBatch = processed + 1 - alignedStep
            let newCount = (0..<8).filter {
                DeltaNetInputHandler.isNewToken(
                    slot: $0, tokensInBatch: tokensInBatch, alignedStep: alignedStep,
                    processedTokenCount: processed)
            }.count
            #expect(newCount == 1, "processed=\(processed) marked \(newCount) slots new")
        }
    }

    // MARK: - Plumbing
    //
    // `processedTokenCount` is what separates a re-sent token from a new one; without it the
    // gate collapses to `slot < tokensInBatch` and marks every re-sent token new. This pins that
    // `InputContext.static` carries it through rather than substituting `alignedStep`.

    @Test("A context carrying the real processed count gates the re-sent tail")
    func contextPreservesProcessedCount() {
        // The prefill->decode seam: 21 tokens processed, decode block starts at 16.
        let context = InputContext.static(
            tokens: [1, 2, 3, 4, 5, 6][...], alignedStep: 16, batchSize: 8,
            slidingWindow: nil, contextBucket: 256, processedTokenCount: 21)
        let isNew = (0..<context.batchSize).map {
            DeltaNetInputHandler.isNewToken(
                slot: $0, tokensInBatch: context.tokens.count, alignedStep: context.alignedStep,
                processedTokenCount: context.processedTokenCount)
        }
        #expect(isNew == [false, false, false, false, false, true, false, false])
    }
}

// MARK: - Conv ring write offset

@Suite("DeltaNetInputHandler conv step")
struct DeltaNetConvStepTests {
    @Test("The write offset is the block start modulo the ring width")
    func wrapsAtRingWidth() {
        #expect(DeltaNetInputHandler.convStep(alignedStep: 0, ringWidth: 128) == 0)
        #expect(DeltaNetInputHandler.convStep(alignedStep: 64, ringWidth: 128) == 64)
        #expect(DeltaNetInputHandler.convStep(alignedStep: 128, ringWidth: 128) == 0)
        #expect(DeltaNetInputHandler.convStep(alignedStep: 200, ringWidth: 128) == 72)
    }

    @Test("A block never straddles the wrap")
    func blockNeverStraddles() {
        // The exporter sizes the ring as a multiple of the max query length, so a block of
        // width b starting at a multiple of b always fits between the offset and the end.
        let ringWidth = 128
        let blockWidth = 64
        for step in stride(from: 0, to: 4 * ringWidth, by: blockWidth) {
            let offset = DeltaNetInputHandler.convStep(alignedStep: step, ringWidth: ringWidth)
            #expect(offset + blockWidth <= ringWidth, "block at \(step) straddles the wrap")
        }
    }
}

// MARK: - Seed selector rows

@Suite("DeltaNetInputHandler seed selector")
struct DeltaNetSelectorTests {
    let ringWidth = 16
    let kernelWindow = 3

    private func rows(alignedStep: Int, batchSize: Int) -> [Int] {
        (0..<(kernelWindow + batchSize)).map {
            DeltaNetInputHandler.selectorRow(
                column: $0, alignedStep: alignedStep, kernelWindow: kernelWindow,
                ringWidth: ringWidth)
        }
    }

    @Test("Columns walk the window in order: left context, then the block")
    func windowOrder() {
        #expect(rows(alignedStep: 5, batchSize: 8) == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    }

    @Test("The block's own columns start after the kernel window")
    func blockStartsAfterLeftContext() {
        let r = rows(alignedStep: 8, batchSize: 4)
        #expect(Array(r[kernelWindow...]) == [8, 9, 10, 11])
    }

    @Test("Negative positions select the ring tail, not a negative index")
    func negativePositionsWrapToTail() {
        // At block start 0 the left context is positions -3, -2, -1.
        let r = rows(alignedStep: 0, batchSize: 4)
        #expect(Array(r[0..<kernelWindow]) == [13, 14, 15])
        #expect(r.allSatisfy { $0 >= 0 && $0 < ringWidth })
    }

    @Test("A window straddling the ring end wraps to the head")
    func wrapsPastTheEnd() {
        let r = rows(alignedStep: ringWidth - 2, batchSize: 4)
        #expect(r == [11, 12, 13, 14, 15, 0, 1])
    }

    @Test("Every row stays in range across many wraps")
    func alwaysInRange() {
        for step in 0..<(5 * ringWidth) {
            #expect(rows(alignedStep: step, batchSize: 8).allSatisfy { $0 >= 0 && $0 < ringWidth })
        }
    }

    @Test("Rows are distinct, so no two columns collide on one ring slot")
    func noCollisions() {
        // The exporter sizes the ring at >= maxQueryLen + kernel - 1 precisely so a whole
        // window fits without two columns landing on the same slot.
        let batchSize = 8
        #expect(ringWidth >= batchSize + kernelWindow)
        for step in 0..<(3 * ringWidth) {
            let r = rows(alignedStep: step, batchSize: batchSize)
            #expect(Set(r).count == r.count, "collision at step \(step): \(r)")
        }
    }
}

// MARK: - Rollback capability

@Suite("DeltaNet rollback capability")
struct DeltaNetRollbackTests {
    @Test("A declared `fixed` state makes the set rewind only to the start")
    func fixedStateIsStartOnly() {
        let kinds: [String: StateKind] = [
            "key_cache": .kvCache, "conv_state": .fixed, "recurrent_state": .fixed,
        ]
        #expect(
            StaticStateFactory.rollback(
                stateNames: ["key_cache", "conv_state", "recurrent_state"], stateKinds: kinds)
                == .startOnly)
    }

    @Test("KV and sliding caches alone rewind to any position")
    func cachesAreAnyPosition() {
        let kinds: [String: StateKind] = ["key_cache": .kvCache, "sliding_k": .slidingCache]
        #expect(
            StaticStateFactory.rollback(stateNames: ["key_cache", "sliding_k"], stateKinds: kinds)
                == .anyPosition)
        #expect(
            StaticStateFactory.rollback(stateNames: ["key_cache"], stateKinds: nil)
                == .anyPosition)
    }

    @Test("State kinds survive the encoded config the engine factory re-parses")
    func stateKindsRoundTrip() throws {
        let config = ModelConfig(
            name: "m", tokenizer: "t", vocabSize: 8, maxContextLength: 256,
            serializedModel: ["m.aimodel"], function: "main",
            states: ["conv_state": .fixed, "recurrent_state": .fixed])
        let parsed = try ModelConfig(parsing: JSONEncoder().encode(config))
        #expect(parsed.states == config.states)
    }

    @Test("A start-only set continues an extension and replays anything else from 0")
    func startOnlyResume() {
        let states = StaticStateSet(
            bucketed: nil, fixed: nil, slidingRing: nil, rollback: .startOnly)
        let canRewind = { states.canTruncate(processed: 10, to: $0) }
        #expect(states.canTruncate(processed: 10, to: 0))
        #expect(states.canTruncate(processed: 10, to: 10))
        #expect(!states.canTruncate(processed: 10, to: 9))
        // Extension past what's processed: continue as is.
        #expect(
            StaticShapeEngine.resumePosition(
                commonPrefix: 10, inputCount: 14, historyCount: 10, processed: 10,
                canRewind: canRewind) == nil)
        // Re-sent prompt the history already covers: would rewind one token, so restart.
        #expect(
            StaticShapeEngine.resumePosition(
                commonPrefix: 10, inputCount: 10, historyCount: 10, processed: 10,
                canRewind: canRewind) == 0)
        // Divergence: restart.
        #expect(
            StaticShapeEngine.resumePosition(
                commonPrefix: 6, inputCount: 12, historyCount: 10, processed: 10,
                canRewind: canRewind) == 0)
    }

    @Test("DeltaNet inputs without `fixed` state metadata are rejected")
    func missingMetadataThrows() {
        #expect(throws: InferenceRuntimeError.self) {
            try DeltaNetInputHandler.validate(rollback: .anyPosition)
        }
        #expect(throws: Never.self) {
            try DeltaNetInputHandler.validate(rollback: .startOnly)
        }
    }
}

// MARK: - Selector write (what `fill` runs) and load-time validation

@Suite("DeltaNetInputHandler selector write")
struct DeltaNetSelectorWriteTests {
    static let ringWidth = 16
    static let kernelWindow = 3
    static let batchSize = 8

    /// The selector as a `ringWidth x (kernelWindow + batchSize)` matrix of 0/1.
    static func matrix(_ array: NDArray) -> [[Float]] {
        let columns = kernelWindow + batchSize
        let flat = readNDArray(array, as: Float16.self, count: ringWidth * columns).map { Float($0) }
        return (0..<ringWidth).map { Array(flat[($0 * columns)..<(($0 + 1) * columns)]) }
    }

    @Test("Each step leaves exactly one 1 per column, at the expected row, and clears the last")
    func oneHotPerStepWithNoStaleEntries() throws {
        var selector = NDArray(
            shape: [1, Self.ringWidth, Self.kernelWindow + Self.batchSize], scalarType: .float16)
        // Two consecutive blocks; the second wraps the ring. The buffer is reused, as in `fill`.
        for alignedStep in [8, 16] {
            DeltaNetInputHandler.writeSeedSelector(
                &selector, alignedStep: alignedStep, kernelWindow: Self.kernelWindow,
                ringWidth: Self.ringWidth)
            let m = Self.matrix(selector)
            #expect(m.flatMap { $0 }.reduce(0, +) == Float(Self.kernelWindow + Self.batchSize))
            for column in 0..<(Self.kernelWindow + Self.batchSize) {
                let row = DeltaNetInputHandler.selectorRow(
                    column: column, alignedStep: alignedStep, kernelWindow: Self.kernelWindow,
                    ringWidth: Self.ringWidth)
                let ones = (0..<Self.ringWidth).filter { m[$0][column] == 1 }
                #expect(ones == [row], "column \(column) at step \(alignedStep)")
            }
        }
    }

    @Test("Ring geometry comes from the selector's shape")
    func geometryFromShape() throws {
        let geometry = try DeltaNetInputHandler.ringGeometry(
            selectorShape: [1, Self.ringWidth, Self.kernelWindow + Self.batchSize],
            batchSize: Self.batchSize)
        #expect(geometry.ringWidth == Self.ringWidth)
        #expect(geometry.kernelWindow == Self.kernelWindow)
    }

    @Test(
        "Selector shapes that cannot address a block are rejected",
        arguments: [
            [Self.ringWidth, Self.kernelWindow + Self.batchSize],  // rank 2
            [2, Self.ringWidth, Self.kernelWindow + Self.batchSize],  // leading dim
            [1, Self.ringWidth, Self.batchSize],  // no kernel window
            [1, Self.batchSize, Self.kernelWindow + Self.batchSize],  // ring too narrow
        ])
    func badGeometryThrows(shape: [Int]) {
        #expect(throws: InferenceRuntimeError.self) {
            _ = try DeltaNetInputHandler.ringGeometry(selectorShape: shape, batchSize: Self.batchSize)
        }
    }
}

@Suite("DeltaNetInputHandler load-time validation")
struct DeltaNetLoadValidationTests {
    static let key = StaticBucketKey(batchSize: 8, contextBucket: 256)
    static let selector = [1, 16, 11]

    @Test("A well-formed bucket passes")
    func wellFormed() {
        #expect(throws: Never.self) {
            try DeltaNetInputHandler.validate(
                updateFlagShapes: [Self.key: [1, 8]], convStepShapes: [Self.key: [1]],
                seedSelectorShapes: [Self.key: Self.selector])
        }
    }

    @Test("conv_step without a seed selector is rejected at load")
    func convStepNeedsSelector() {
        #expect(throws: InferenceRuntimeError.self) {
            try DeltaNetInputHandler.validate(
                updateFlagShapes: [Self.key: [1, 8]], convStepShapes: [Self.key: [1]],
                seedSelectorShapes: [:])
        }
    }

    @Test("An update flag that is not batchSize wide is rejected at load")
    func flagWidth() {
        #expect(throws: InferenceRuntimeError.self) {
            try DeltaNetInputHandler.validate(
                updateFlagShapes: [Self.key: [1, 4]], convStepShapes: [:],
                seedSelectorShapes: [Self.key: Self.selector])
        }
    }

    @Test("Every bucket's selector is checked, not only the first")
    func everyBucket() {
        let other = StaticBucketKey(batchSize: 64, contextBucket: 256)
        #expect(throws: InferenceRuntimeError.self) {
            try DeltaNetInputHandler.validate(
                updateFlagShapes: [:], convStepShapes: [:],
                seedSelectorShapes: [Self.key: Self.selector, other: Self.selector])
        }
    }
}

// MARK: - Step write (what `fill` runs once the buffers are sized)

@Suite("DeltaNetInputHandler step write")
struct DeltaNetStepWriteTests {
    static let ringWidth = 8
    static let kernelWindow = 3
    static let batchSize = 4
    static let columns = kernelWindow + batchSize

    /// The buffers as `fill` leaves them after sizing. With `padded`, each selector row is 16
    /// elements wide in memory, so a write that ignores the strides lands in the wrong place.
    static func buffers(padded: Bool) -> InputBuffers {
        var buffers = InputBuffers()
        buffers[DeltaNetInputHandler.updateFlagInputName] = NDArray(
            shape: [1, batchSize], scalarType: .float16)
        buffers[DeltaNetInputHandler.convStepInputName] = NDArray(shape: [1], scalarType: .int32)
        let shape = [1, ringWidth, columns]
        buffers[DeltaNetInputHandler.seedSelectorInputName] =
            padded
            ? NDArray(shape: shape, scalarType: .float16, strides: [ringWidth * 16, 16, 1])
            : NDArray(shape: shape, scalarType: .float16)
        return buffers
    }

    static func context(tokens: Int, alignedStep: Int, processedTokenCount: Int) -> InputContext {
        InputContext.static(
            tokens: ArraySlice((0..<tokens).map { Int32($0) }), alignedStep: alignedStep,
            batchSize: batchSize, slidingWindow: nil, contextBucket: 256,
            processedTokenCount: processedTokenCount)
    }

    static func write(_ context: InputContext, into buffers: inout InputBuffers) throws {
        try DeltaNetInputHandler.write(
            context, into: &buffers, updateFlag: true, convStep: true, seedSelector: true)
    }

    static func flag(_ buffers: InputBuffers) -> [Float] {
        readNDArray(
            buffers[DeltaNetInputHandler.updateFlagInputName]!, as: Float16.self, count: batchSize
        ).map { Float($0) }
    }

    static func convStep(_ buffers: InputBuffers) -> Int32 {
        readNDArray(buffers[DeltaNetInputHandler.convStepInputName]!, as: Int32.self, count: 1)[0]
    }

    /// The selector as a `ringWidth x columns` matrix, read element by element through its strides.
    static func selector(_ buffers: InputBuffers) -> [[Float]] {
        let flat = readNDArray(
            buffers[DeltaNetInputHandler.seedSelectorInputName]!, as: Float16.self,
            count: ringWidth * columns
        ).map { Float($0) }
        return (0..<ringWidth).map { Array(flat[($0 * columns)..<(($0 + 1) * columns)]) }
    }

    /// The one-hot matrix with a 1 at `(rows[column], column)` and 0 everywhere else.
    static func oneHot(rows: [Int]) -> [[Float]] {
        (0..<ringWidth).map { row in (0..<columns).map { rows[$0] == row ? 1 : 0 } }
    }

    @Test("Two steps write the flag, conv step and selector, leaving nothing of the first", arguments: [false, true])
    func twoSteps(padded: Bool) throws {
        var buffers = Self.buffers(padded: padded)

        // A full prefill block at 4: every slot new; the window covers positions 1...7.
        try Self.write(
            Self.context(tokens: 4, alignedStep: 4, processedTokenCount: 4), into: &buffers)
        #expect(Self.flag(buffers) == [1, 1, 1, 1])
        #expect(Self.convStep(buffers) == 4)
        #expect(Self.selector(buffers) == Self.oneHot(rows: [1, 2, 3, 4, 5, 6, 7]))

        // A decode block at 8 with 10 processed: slots 0 and 1 re-send positions 8 and 9, slot 2
        // carries the new position 10, slot 3 is padding. The window covers positions 5...11 and
        // wraps the ring.
        try Self.write(
            Self.context(tokens: 3, alignedStep: 8, processedTokenCount: 10), into: &buffers)
        #expect(Self.flag(buffers) == [0, 0, 1, 0])
        #expect(Self.convStep(buffers) == 0)
        let selector = Self.selector(buffers)
        #expect(selector == Self.oneHot(rows: [5, 6, 7, 0, 1, 2, 3]))
        for column in 0..<Self.columns {
            #expect(selector.map { $0[column] }.reduce(0, +) == 1, "column \(column)")
        }
    }

    @Test("Only the declared inputs are written")
    func onlyDeclaredInputs() throws {
        var buffers = InputBuffers()
        buffers[DeltaNetInputHandler.updateFlagInputName] = NDArray(
            shape: [1, Self.batchSize], scalarType: .float16)
        try DeltaNetInputHandler.write(
            Self.context(tokens: 2, alignedStep: 0, processedTokenCount: 0), into: &buffers,
            updateFlag: true, convStep: false, seedSelector: false)
        #expect(Self.flag(buffers) == [1, 1, 0, 0])
    }

    @Test("conv_step without a seed selector throws")
    func convStepNeedsSelector() {
        var buffers = Self.buffers(padded: false)
        #expect(throws: InferenceRuntimeError.self) {
            try DeltaNetInputHandler.write(
                Self.context(tokens: 1, alignedStep: 0, processedTokenCount: 0), into: &buffers,
                updateFlag: true, convStep: true, seedSelector: false)
        }
    }

    @Test("A declared input with no buffer throws")
    func missingBufferThrows() {
        var buffers = InputBuffers()
        #expect(throws: InferenceRuntimeError.self) {
            try Self.write(
                Self.context(tokens: 1, alignedStep: 0, processedTokenCount: 0), into: &buffers)
        }
    }

    @Test("A selector buffer that cannot address the block throws")
    func badSelectorThrows() {
        var buffers = Self.buffers(padded: false)
        // Ring narrower than the window: (1, batchSize, kernelWindow + batchSize).
        buffers[DeltaNetInputHandler.seedSelectorInputName] = NDArray(
            shape: [1, Self.batchSize, Self.columns], scalarType: .float16)
        #expect(throws: InferenceRuntimeError.self) {
            try Self.write(
                Self.context(tokens: 1, alignedStep: 0, processedTokenCount: 0), into: &buffers)
        }
    }
}
