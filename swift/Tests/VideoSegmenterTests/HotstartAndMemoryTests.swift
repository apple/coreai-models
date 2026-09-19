// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Testing

@testable import CoreAIShared
@testable import CoreAIVideoSegmenter

@Suite("HotstartHeuristics")
@VideoSegmentationActor
struct HotstartTests {
    private func session() -> VideoInferenceSession {
        VideoInferenceSession(videoWidth: 64, videoHeight: 64)
    }

    private func register(_ session: VideoInferenceSession, ids: [Int], firstSeen: Int) {
        for id in ids {
            session.index(ofObject: id)
            session.firstFrameByObjectID[id] = firstSeen
            session.promptIDByObjectID[id] = 0
        }
    }

    @discardableResult
    private func run(
        _ session: VideoInferenceSession,
        frame: Int,
        matched: [Int: [Int]] = [:],
        new: [Int] = [],
        unmatched: [Int] = [],
        empty: [Int] = [],
        parameters: VideoSegmentationParameters = .default
    ) -> Set<Int> {
        HotstartHeuristics.process(
            session: session, frameIndex: frame, reverse: false,
            detectionToMatchedTrackIDs: matched, newObjectIDs: new,
            emptyTrackIDs: empty, unmatchedTrackIDs: unmatched, parameters: parameters)
    }

    @Test("A new object records its first frame and starts at the initial keep-alive")
    func seeding() {
        let session = session()
        run(session, frame: 3, new: [5])
        #expect(session.firstFrameByObjectID[5] == 3)
        #expect(session.keepAliveByObjectID[5] == VideoSegmentationParameters.default.initTrkKeepAlive)
    }

    @Test("Keep-alive rises on a match and falls on a miss, within its bounds")
    func keepAliveBounds() {
        var parameters = VideoSegmentationParameters.default
        parameters.initTrkKeepAlive = 2
        parameters.maxTrkKeepAlive = 3
        parameters.minTrkKeepAlive = -1

        let session = session()
        register(session, ids: [5], firstSeen: 0)
        run(session, frame: 1, new: [5], parameters: parameters)
        #expect(session.keepAliveByObjectID[5] == 2)

        run(session, frame: 2, matched: [0: [5]], parameters: parameters)
        #expect(session.keepAliveByObjectID[5] == 3)
        run(session, frame: 3, matched: [0: [5]], parameters: parameters)
        #expect(session.keepAliveByObjectID[5] == 3, "clamped at maxTrkKeepAlive")

        for frame in 4...9 { run(session, frame: frame, unmatched: [5], parameters: parameters) }
        #expect(session.keepAliveByObjectID[5] == -1, "clamped at minTrkKeepAlive")
    }

    @Test("A track unmatched for long enough inside the window is removed")
    func removesUnmatchedInsideWindow() {
        var parameters = VideoSegmentationParameters.default
        parameters.hotstartDelay = 15
        parameters.hotstartUnmatchThresh = 3

        let session = session()
        register(session, ids: [5], firstSeen: 2)
        // Frames 3 and 4 accumulate misses; frame 5 crosses the threshold. Object 5 first
        // appeared at frame 2, which is after `5 - 15`, so it is still inside the window.
        var removed: Set<Int> = []
        for frame in 3...5 {
            removed = run(session, frame: frame, unmatched: [5], parameters: parameters)
        }
        #expect(removed == [5])
        #expect(session.removedObjectIDs.contains(5))
    }

    @Test("A long-established track is never removed by the unmatch rule")
    func sparesEstablishedTracks() {
        // The point of the hotstart window: a track that predates it has earned the benefit
        // of the doubt, however long the detector loses sight of it.
        var parameters = VideoSegmentationParameters.default
        parameters.hotstartDelay = 5
        parameters.hotstartUnmatchThresh = 3

        let session = session()
        register(session, ids: [5], firstSeen: 0)
        var removed: Set<Int> = []
        for frame in 20...30 {
            removed = run(session, frame: frame, unmatched: [5], parameters: parameters)
        }
        #expect(removed.isEmpty)
        #expect(session.unmatchedFramesByObjectID[5]?.count == 11)
    }

    @Test("A duplicate that overlaps an earlier track for long enough is removed")
    func removesDuplicates() {
        var parameters = VideoSegmentationParameters.default
        parameters.hotstartDelay = 20
        parameters.hotstartDupThresh = 2

        let session = session()
        register(session, ids: [1], firstSeen: 1)
        register(session, ids: [2], firstSeen: 4)

        // The same detection claims both tracks. Object 2 appeared later, so it is the
        // candidate duplicate; object 1 is never at risk.
        var removed: Set<Int> = []
        for frame in 5...6 {
            removed = run(session, frame: frame, matched: [0: [1, 2]], parameters: parameters)
        }
        #expect(removed == [2])
    }

    @Test("A single track matching a detection is not an overlap")
    func singleMatchIsNotOverlap() {
        var parameters = VideoSegmentationParameters.default
        parameters.hotstartDupThresh = 1

        let session = session()
        register(session, ids: [1], firstSeen: 0)
        let removed = run(session, frame: 5, matched: [0: [1]], parameters: parameters)
        #expect(removed.isEmpty)
        #expect(session.overlapFramesByPair.isEmpty)
    }

    @Test("Keep-alive suppression only applies when the hotstart restriction is lifted")
    func keepAliveSuppression() {
        var parameters = VideoSegmentationParameters.default
        parameters.suppressUnmatchedOnlyWithinHotstart = true
        parameters.initTrkKeepAlive = 1
        parameters.minTrkKeepAlive = -1
        parameters.hotstartUnmatchThresh = 100  // keep the removal rule out of the way

        let session = session()
        register(session, ids: [5], firstSeen: 0)
        session.keepAliveByObjectID[5] = 0
        run(session, frame: 10, unmatched: [5], parameters: parameters)
        #expect(session.suppressedObjectIDsByFrame[10]?.isEmpty ?? true)

        parameters.suppressUnmatchedOnlyWithinHotstart = false
        run(session, frame: 11, unmatched: [5], parameters: parameters)
        #expect(session.suppressedObjectIDsByFrame[11]?.contains(5) == true)
    }
}

@Suite("MemoryBankPacker selection")
@VideoSegmentationActor
struct MemorySelectionTests {
    /// A history with the given conditioning and non-conditioning frames, each carrying a
    /// payload so the packer treats it as usable. The selection code never dereferences
    /// `objectPointer`, so a bare zero-filled array stands in and no live asset is needed.
    private func history(conditioning: [Int], nonConditioning: [Int]) -> ObjectOutputHistory {
        func stub() -> StoredFrameOutput {
            StoredFrameOutput(
                predictedMasks: nil,
                objectPointer: NDArray(shape: [1, 1, 4], scalarType: .float16),
                objectScoreLogit: 0)
        }
        var history = ObjectOutputHistory()
        for frame in conditioning {
            history.conditioningOrder.append(frame)
            history.conditioning[frame] = stub()
        }
        for frame in nonConditioning {
            history.nonConditioning[frame] = stub()
        }
        return history
    }

    @Test("Under the cap, every conditioning frame is selected")
    func selectsAllUnderCap() {
        let (selected, unselected) = MemoryBankPacker.selectClosestConditioningFrames(
            history: history(conditioning: [0, 5, 9], nonConditioning: []),
            frameIndex: 12, limit: 4)
        #expect(selected == [0, 5, 9])
        #expect(unselected.isEmpty)
    }

    @Test("Over the cap, the nearest before and after come first, then the next closest")
    func selectsClosest() {
        // Port of `_select_closest_cond_frames`: frame 9 is the nearest before 10, frame
        // 11 the nearest at-or-after, then 5 as the next closest. Frame 0 is dropped.
        let (selected, unselected) = MemoryBankPacker.selectClosestConditioningFrames(
            history: history(conditioning: [0, 5, 9, 11, 30], nonConditioning: []),
            frameIndex: 10, limit: 3)
        #expect(Set(selected) == [9, 11, 5])
        #expect(unselected == [0, 30])
    }

    @Test("Recent frames are gathered newest-last, with gaps preserved")
    func gathersRecentFrames() {
        var parameters = VideoSegmentationParameters.default
        parameters.numMaskmem = 4  // offsets 3, 2, 1
        parameters.maxCondFrameNum = 4

        let entries = MemoryBankPacker.gatherMemoryFrames(
            history: history(conditioning: [0], nonConditioning: [8, 9]),
            frameIndex: 10, reverse: false, parameters: parameters)

        #expect(entries.map(\.offset) == [0, 3, 2, 1])
        // Frame 7 is missing, so offset 3 is a gap. It stays in the list rather than being
        // filtered, because the offset has to stay attached to the right entry.
        #expect(entries[1].output == nil)
        #expect(entries[2].output != nil)
        #expect(entries[3].output != nil)
    }

    @Test("Tracking backwards looks forward for recent frames")
    func gathersInReverse() {
        var parameters = VideoSegmentationParameters.default
        parameters.numMaskmem = 3  // offsets 2, 1

        let entries = MemoryBankPacker.gatherMemoryFrames(
            history: history(conditioning: [20], nonConditioning: [11, 12]),
            frameIndex: 10, reverse: true, parameters: parameters)
        #expect(entries.map(\.offset) == [0, 2, 1])
        #expect(entries[1].output != nil, "frame 12 is two ahead")
        #expect(entries[2].output != nil, "frame 11 is one ahead")
    }

    @Test("An unselected conditioning frame can still be picked up as a recent memory")
    func unselectedConditioningIsReachable() {
        // `_gather_memory_frame_outputs` falls back to `unselected_conditioning_outputs` for
        // the recent window. Without it, a conditioning frame that lost the closest-N contest
        // would vanish from the bank entirely even though it is adjacent.
        var parameters = VideoSegmentationParameters.default
        parameters.numMaskmem = 3  // offsets 2, 1
        parameters.maxCondFrameNum = 1

        let entries = MemoryBankPacker.gatherMemoryFrames(
            history: history(conditioning: [0, 9], nonConditioning: []),
            frameIndex: 10, reverse: false, parameters: parameters)
        #expect(entries[0].offset == 0)
        #expect(entries.count == 3)
    }

    @Test("Object pointers cover eligible conditioning frames and a contiguous look-back")
    func objectPointers() {
        var parameters = VideoSegmentationParameters.default
        parameters.maxObjectPointers = 4

        let (offsets, _, maxPointers) = MemoryBankPacker.objectPointers(
            history: history(conditioning: [2], nonConditioning: [7, 8, 9]),
            frameIndex: 10, totalFrames: 50, reverse: false, parameters: parameters)
        // Conditioning frame 2 is 8 back; then offsets 1, 2, 3 for frames 9, 8, 7.
        #expect(offsets == [8, 1, 2, 3])
        #expect(maxPointers == 4)
    }

    @Test("A future conditioning frame is ineligible when tracking forwards")
    func futureConditioningExcluded() {
        let (offsets, _, _) = MemoryBankPacker.objectPointers(
            history: history(conditioning: [20], nonConditioning: []),
            frameIndex: 10, totalFrames: 50, reverse: false, parameters: .default)
        #expect(offsets.isEmpty)
    }

    @Test("The look-back stops at the start of the video rather than skipping past it")
    func lookBackStopsAtBoundary() {
        // Upstream `break`s on an out-of-range index. A `continue` would keep scanning and
        // pick up nothing, which happens to agree here but not when frames are sparse.
        var parameters = VideoSegmentationParameters.default
        parameters.maxObjectPointers = 8

        let (offsets, _, maxPointers) = MemoryBankPacker.objectPointers(
            history: history(conditioning: [0], nonConditioning: [1]),
            frameIndex: 2, totalFrames: 3, reverse: false, parameters: parameters)
        #expect(offsets == [2, 1])
        #expect(maxPointers == 3, "capped by the video length, not the config")
    }
}

@Suite("MemoryBankPacker configuration")
@VideoSegmentationActor
struct MemoryBankConfigurationTests {
    private func shapes(spatialSlots: Int = 64) -> VideoSegmentationEngine.Shapes {
        VideoSegmentationEngine.Shapes(
            imageSize: 1008, textSequenceLength: 32, lowResMaskSize: 252, memoryMaskSize: 252,
            highResMaskSize: 1008, memoryTokenCount: 4, memoryDim: 4, hiddenDim: 4,
            spatialSlots: spatialSlots, ptrSlots: 24)
    }

    private func packed() -> PackedMemory {
        func stub() -> NDArray { NDArray(shape: [1, 1, 4], scalarType: .float16) }
        return PackedMemory(
            spatialMemory: stub(), spatialMemoryPosition: stub(), spatialTemporalIndex: stub(),
            spatialSlotOccupancy: stub(), objectPointers: stub(), pointerTemporalPosition: stub(),
            pointerSlotOccupancy: stub())
    }

    private func makePacker(_ mutate: (inout VideoSegmentationParameters) -> Void) throws
        -> MemoryBankPacker
    {
        var parameters = VideoSegmentationParameters.default
        mutate(&parameters)
        return try MemoryBankPacker(
            shapes: shapes(), parameters: parameters, packed: packed())
    }

    @Test("The default configuration is accepted")
    func defaultsAccepted() throws {
        _ = try makePacker { _ in }
    }

    @Test("num_maskmem of zero is refused, since the temporal index divides by it")
    func rejectsZeroMaskmem() {
        #expect(throws: VideoSegmentationError.self) {
            try makePacker { $0.numMaskmem = 0 }
        }
    }

    @Test("max_cond_frame_num below two is refused")
    func rejectsSmallCondFrameCap() {
        // `_select_closest_cond_frames` takes the nearest before and the nearest at-or-after
        // before honouring the cap, so a cap of 1 yields two entries and the slot budget
        // computed from it would be short.
        #expect(throws: VideoSegmentationError.self) {
            try makePacker { $0.maxCondFrameNum = 1 }
        }
    }

    @Test("An unbounded max_cond_frame_num is refused by a fixed-slot bank")
    func rejectsUnboundedCondFrameCap() {
        #expect(throws: VideoSegmentationError.self) {
            try makePacker { $0.maxCondFrameNum = -1 }
        }
    }
}

@Suite("ObjectOutputHistory pruning")
@VideoSegmentationActor
struct HistoryPruningTests {
    private static let ptrSlots = 24

    private func parameters() -> VideoSegmentationParameters {
        var parameters = VideoSegmentationParameters.default
        parameters.numMaskmem = 7
        parameters.maxCondFrameNum = 4
        parameters.maxObjectPointers = 16
        return parameters
    }

    private func capacity(_ parameters: VideoSegmentationParameters) -> Int {
        max(Self.ptrSlots, memoryCapacity(parameters))
    }

    private func memoryCapacity(_ parameters: VideoSegmentationParameters) -> Int {
        parameters.maxCondFrameNum + parameters.numMaskmem
    }

    private func payload() -> MemoryPayload {
        MemoryPayload(reading: NDArray(shape: [1, 1, 4], scalarType: .float16))
    }

    /// One frame in the real write order: store, promote if reconditioning, attach memory.
    /// `objectScoreLogit` carries the frame so a packed slot can be traced to its source.
    private func advance(
        _ history: inout ObjectOutputHistory, to frame: Int, reconditioning: Bool = true
    ) {
        history.store(
            StoredFrameOutput(
                predictedMasks: [0],
                objectPointer: NDArray(shape: [1, 1, 4], scalarType: .float16),
                objectScoreLogit: Float(frame)),
            at: frame, conditioning: false)
        if reconditioning { history.promoteToConditioning(frame: frame) }
        history.attachMemory(features: payload(), positionEncoding: payload(), at: frame)
    }

    private func prune(
        _ history: inout ObjectOutputHistory, at frame: Int,
        _ parameters: VideoSegmentationParameters
    ) {
        history.prune(
            before: frame, memoryWindow: parameters.numMaskmem,
            pointerWindow: max(parameters.maxObjectPointers, parameters.numMaskmem),
            conditioningCapacity: capacity(parameters),
            conditioningMemoryCapacity: memoryCapacity(parameters))
    }

    /// What reaches the spatial bank: `packSpatial` skips entries missing either payload,
    /// so a slot is just its frame plus whether it still carries memory.
    private func slots(
        _ history: ObjectOutputHistory, frameIndex: Int, _ parameters: VideoSegmentationParameters
    ) -> [String] {
        MemoryBankPacker.gatherMemoryFrames(
            history: history, frameIndex: frameIndex, reverse: false, parameters: parameters
        ).map { entry in
            guard let output = entry.output, output.memoryFeatures != nil,
                output.memoryPositionEncoding != nil
            else { return "\(entry.offset):empty" }
            return "\(entry.offset):\(output.objectScoreLogit)"
        }
    }

    /// The offsets surviving `packPointers`, which keeps the `ptrSlots` closest. An offset
    /// identifies its frame uniquely, so equal offsets mean equal pointers.
    private func pointerSlots(
        _ history: ObjectOutputHistory, frameIndex: Int, _ parameters: VideoSegmentationParameters
    ) -> [Int] {
        let (offsets, _, _) = MemoryBankPacker.objectPointers(
            history: history, frameIndex: frameIndex, totalFrames: 10_000, reverse: false,
            parameters: parameters)
        guard offsets.count > Self.ptrSlots else { return offsets }
        return offsets.indices
            .sorted { abs(offsets[$0]) < abs(offsets[$1]) }
            .prefix(Self.ptrSlots)
            .sorted()
            .map { offsets[$0] }
    }

    @Test(
        "Pruning never changes what the packer would pack",
        arguments: [1, 16])
    func pruningIsBitExact(reconditionEvery: Int) {
        // The claim behind dropping conditioning history: a forward pass can never read
        // what prune removes. 16 is the shipped cadence, 1 the worst case for growth.
        let parameters = parameters()
        var pruned = ObjectOutputHistory()
        var retained = ObjectOutputHistory()

        for frame in 0..<600 {
            // Both banks are read at the top of a frame, before that frame's own entry lands.
            #expect(
                slots(pruned, frameIndex: frame, parameters)
                    == slots(retained, frameIndex: frame, parameters),
                "spatial bank diverged at frame \(frame), cadence \(reconditionEvery)")
            #expect(
                pointerSlots(pruned, frameIndex: frame, parameters)
                    == pointerSlots(retained, frameIndex: frame, parameters),
                "pointer bank diverged at frame \(frame), cadence \(reconditionEvery)")

            let reconditioning = frame % reconditionEvery == 0
            advance(&pruned, to: frame, reconditioning: reconditioning)
            advance(&retained, to: frame, reconditioning: reconditioning)
            prune(&pruned, at: frame, parameters)
        }

        #expect(retained.conditioningOrder.count == (600 + reconditionEvery - 1) / reconditionEvery)
        #expect(pruned.conditioningOrder.count <= capacity(parameters))
    }

    @Test("Conditioning history stays bounded when every frame reconditions")
    func conditioningStaysBounded() {
        let parameters = parameters()
        var history = ObjectOutputHistory()
        for frame in 0..<2_000 {
            advance(&history, to: frame)
            prune(&history, at: frame, parameters)
        }

        #expect(history.conditioningOrder.count <= capacity(parameters))
        #expect(history.conditioning.count <= capacity(parameters))
        #expect(history.framesTracked.count <= capacity(parameters))

        // The megabyte-scale half expires on the shorter memory horizon.
        let withMemory = history.conditioning.values.filter { $0.memoryFeatures != nil }
        #expect(withMemory.count <= memoryCapacity(parameters))
        let withMasks = history.conditioning.values.filter { $0.predictedMasks != nil }
        #expect(withMasks.count <= parameters.numMaskmem + 1)
    }

    @Test("A conditioning frame keeps its pointer after its memory expires")
    func pointerOutlivesMemory() {
        // Horizons differ on purpose: a pointer is kilobytes and stays reachable far
        // longer than the memory it came with.
        let parameters = parameters()
        var history = ObjectOutputHistory()
        for frame in 0...20 {
            advance(&history, to: frame)
            prune(&history, at: frame, parameters)
        }
        #expect(history.conditioning[5] != nil, "still inside the pointer budget")
        #expect(history.conditioning[5]?.memoryFeatures == nil, "outside the memory budget")
        #expect(history.conditioning[18]?.memoryFeatures != nil, "inside the memory budget")
    }

    @Test("Pruning backwards is refused rather than silently dropping reachable history")
    func rejectsBackwardPrune() throws {
        let session = VideoInferenceSession(videoWidth: 64, videoHeight: 64)
        try session.prune(
            currentFrame: 10, memoryWindow: 7, pointerWindow: 16, conditioningCapacity: 24,
            conditioningMemoryCapacity: 11)
        try session.prune(
            currentFrame: 10, memoryWindow: 7, pointerWindow: 16, conditioningCapacity: 24,
            conditioningMemoryCapacity: 11)
        #expect(throws: VideoSegmentationError.self) {
            try session.prune(
                currentFrame: 9, memoryWindow: 7, pointerWindow: 16, conditioningCapacity: 24,
                conditioningMemoryCapacity: 11)
        }
    }
}

@Suite("buildOutputs object pairing")
@VideoSegmentationActor
struct BuildOutputsPairingTests {
    /// Tiny geometry so a mask is one float. `buildOutputs` only reaches `lowResMaskSize` and
    /// the parameters, never the engine, so no asset has to load.
    private func shapes() -> VideoSegmentationEngine.Shapes {
        VideoSegmentationEngine.Shapes(
            imageSize: 8, textSequenceLength: 4, lowResMaskSize: 1, memoryMaskSize: 1,
            highResMaskSize: 1, memoryTokenCount: 4, memoryDim: 4, hiddenDim: 4,
            spatialSlots: 64, ptrSlots: 24)
    }

    private func makeProcessor(hotstartEnabled: Bool) throws -> FrameProcessor {
        var parameters = VideoSegmentationParameters.default
        parameters.hotstartDelay = hotstartEnabled ? 15 : 0  // `hotstartEnabled` is derived
        parameters.fillHoleArea = 0  // leave detection masks byte-identical
        let shapes = shapes()
        let engine = VideoSegmentationEngine(modelURL: URL(fileURLWithPath: "/nonexistent"))
        func stub() -> NDArray { NDArray(shape: [1, 1, 4], scalarType: .float16) }
        let packer = try MemoryBankPacker(
            shapes: shapes, parameters: parameters,
            packed: PackedMemory(
                spatialMemory: stub(), spatialMemoryPosition: stub(), spatialTemporalIndex: stub(),
                spatialSlotOccupancy: stub(), objectPointers: stub(),
                pointerTemporalPosition: stub(), pointerSlotOccupancy: stub()))
        return FrameProcessor(
            engine: engine, shapes: shapes, parameters: parameters,
            tracker: TrackerLoop(
                engine: engine, shapes: shapes, parameters: parameters, packer: packer))
    }

    /// A session tracking `ids`, in registration order.
    private func session(tracking ids: [Int]) -> VideoInferenceSession {
        let session = VideoInferenceSession(videoWidth: 8, videoHeight: 8)
        for id in ids { session.index(ofObject: id) }
        return session
    }

    /// Each object's mask and score logit encode its own id, so a mis-pairing is legible.
    private func maskLogits(for ids: [Int]) -> [[Float]] { ids.map { [Float($0)] } }
    private func scoreLogits(for ids: [Int]) -> [Float] { ids.map { Float($0) } }

    @Test("A mid-list removal leaves every survivor holding its own mask")
    func removalDoesNotShiftMasks() throws {
        // The bug this guards: `execute` removes object 2, `ObjectRegistry.remove` compacts the
        // registry, and a post-removal zip hands object 3 the mask computed for object 2.
        let processor = try makeProcessor(hotstartEnabled: true)
        let tracked = [1, 2, 3]
        let session = session(tracking: tracked)

        var plan = TrackerUpdatePlan()
        plan.newlyRemovedObjectIDs = [2]
        session.removeObject(2)  // what `execute` already did by the time outputs are built

        let output = processor.buildOutputs(
            session: session, frameIndex: 0, detections: MergedDetections(),
            trackedObjectIDs: tracked,
            trackerLogits: maskLogits(for: tracked), trackerScoreLogits: scoreLogits(for: tracked),
            plan: plan, newScores: [:])

        #expect(output.maskLogitsByObjectID[1] == [1])
        #expect(output.maskLogitsByObjectID[3] == [3], "object 3 must not inherit object 2's mask")
        #expect(output.maskLogitsByObjectID[2] == nil, "the removed object emits no mask")
    }

    @Test("A mid-list removal leaves every survivor holding its own tracker score")
    func removalDoesNotShiftScores() throws {
        // The second zip, which would otherwise be fixed only by accident.
        let processor = try makeProcessor(hotstartEnabled: true)
        let tracked = [1, 2, 3]
        let session = session(tracking: tracked)

        var plan = TrackerUpdatePlan()
        plan.newlyRemovedObjectIDs = [2]
        session.removeObject(2)

        let output = processor.buildOutputs(
            session: session, frameIndex: 0, detections: MergedDetections(),
            trackedObjectIDs: tracked,
            trackerLogits: maskLogits(for: tracked), trackerScoreLogits: scoreLogits(for: tracked),
            plan: plan, newScores: [:])

        #expect(output.trackerScoreByObjectID[1] == DetectionDecoder.sigmoid(1))
        #expect(
            output.trackerScoreByObjectID[3] == DetectionDecoder.sigmoid(3),
            "object 3 must not inherit object 2's score")
    }

    @Test("A removed object emits no mask even with hotstart disabled")
    func removedObjectHiddenWithoutHotstart() throws {
        // `MaskPostprocessor` only filters `hotstartRemovedObjectIDs`, which stays empty when
        // hotstart is off, so `buildOutputs` has to drop the removal itself.
        let processor = try makeProcessor(hotstartEnabled: false)
        let tracked = [1, 2]
        let session = session(tracking: tracked)

        var plan = TrackerUpdatePlan()
        plan.newlyRemovedObjectIDs = [2]
        session.removeObject(2)

        let output = processor.buildOutputs(
            session: session, frameIndex: 0, detections: MergedDetections(),
            trackedObjectIDs: tracked,
            trackerLogits: maskLogits(for: tracked), trackerScoreLogits: scoreLogits(for: tracked),
            plan: plan, newScores: [:])

        #expect(output.maskLogitsByObjectID[2] == nil)
        #expect(session.hotstartRemovedObjectIDs.isEmpty)
    }

    @Test("With no removal every object keeps its own mask, new objects included")
    func additionsArePairedUnchanged() throws {
        // New objects are appended after the tracker ran, so the zip stops short of them and
        // the detection-mask override fills them in.
        let processor = try makeProcessor(hotstartEnabled: true)
        let tracked = [1, 2]
        let session = session(tracking: tracked)
        session.index(ofObject: 3)

        var plan = TrackerUpdatePlan()
        plan.newObjectIDs = [3]
        plan.newDetectionIndices = [0]
        var detections = MergedDetections()
        detections.maskLogits = [[99]]

        let output = processor.buildOutputs(
            session: session, frameIndex: 0, detections: detections,
            trackedObjectIDs: tracked,
            trackerLogits: maskLogits(for: tracked), trackerScoreLogits: scoreLogits(for: tracked),
            plan: plan, newScores: [:])

        #expect(output.maskLogitsByObjectID[1] == [1])
        #expect(output.maskLogitsByObjectID[2] == [2])
        #expect(output.maskLogitsByObjectID[3] == [99], "seeded from its detection mask")
    }
}

/// The slot-writing half of the packer. `MemoryBankPacker selection` covers *which* memories
/// are chosen; this covers where they land, what temporal index they carry, and whether a
/// slot the previous object filled is cleared before the next one reads the bank.
@Suite("MemoryBankPacker packing")
@VideoSegmentationActor
struct MemoryBankPackingTests {
    private static let spatialSlots = 8
    private static let ptrSlots = 4
    /// `memoryTokenCount * memoryDim` and `hiddenDim` from ``shapes()``.
    private static let slotElements = 16
    private static let pointerElements = 4

    private func shapes() -> VideoSegmentationEngine.Shapes {
        VideoSegmentationEngine.Shapes(
            imageSize: 1008, textSequenceLength: 32, lowResMaskSize: 252, memoryMaskSize: 252,
            highResMaskSize: 1008, memoryTokenCount: 4, memoryDim: 4, hiddenDim: 4,
            spatialSlots: Self.spatialSlots, ptrSlots: Self.ptrSlots)
    }

    /// A bank at its real element counts. The configuration suite gets away with `[1, 1, 4]`
    /// stubs because `init` never writes; `pack` does, and would trip `copyIntoNDArray`'s
    /// capacity precondition on an undersized tensor.
    private func packed() -> PackedMemory {
        func half(_ count: Int) -> NDArray { NDArray(shape: [count], scalarType: .float16) }
        func float(_ count: Int) -> NDArray { NDArray(shape: [count], scalarType: .float32) }
        return PackedMemory(
            spatialMemory: half(Self.spatialSlots * Self.slotElements),
            spatialMemoryPosition: half(Self.spatialSlots * Self.slotElements),
            spatialTemporalIndex: NDArray(shape: [Self.spatialSlots], scalarType: .int32),
            spatialSlotOccupancy: float(Self.spatialSlots),
            objectPointers: half(Self.ptrSlots * Self.pointerElements),
            pointerTemporalPosition: float(Self.ptrSlots),
            pointerSlotOccupancy: float(Self.ptrSlots))
    }

    private func parameters(
        numMaskmem: Int = 3, maxCondFrameNum: Int = 2, maxObjectPointers: Int = 4
    ) -> VideoSegmentationParameters {
        var parameters = VideoSegmentationParameters.default
        parameters.numMaskmem = numMaskmem
        parameters.maxCondFrameNum = maxCondFrameNum
        parameters.maxObjectPointers = maxObjectPointers
        return parameters
    }

    private func makePacker(
        _ parameters: VideoSegmentationParameters
    ) throws -> MemoryBankPacker {
        try MemoryBankPacker(shapes: shapes(), parameters: parameters, packed: packed())
    }

    /// Every element of the payload carries `value`, so a packed slot can be traced back to
    /// the frame that produced it. `memory: false` models a frame stored but not yet encoded.
    private func stored(_ value: Float, memory: Bool = true) -> StoredFrameOutput {
        func filled(_ count: Int) -> NDArray {
            var array = NDArray(shape: [count], scalarType: .float16)
            fillFloatNDArray(&array, with: [Float](repeating: value, count: count))
            return array
        }
        var output = StoredFrameOutput(
            predictedMasks: nil, objectPointer: filled(Self.pointerElements),
            objectScoreLogit: value)
        if memory {
            let payload = MemoryPayload(reading: filled(Self.slotElements))
            output.memoryFeatures = payload
            output.memoryPositionEncoding = payload
        }
        return output
    }

    private func history(
        conditioning: [(frame: Int, value: Float)],
        nonConditioning: [(frame: Int, value: Float, memory: Bool)] = []
    ) -> ObjectOutputHistory {
        var history = ObjectOutputHistory()
        for entry in conditioning {
            history.store(stored(entry.value), at: entry.frame, conditioning: true)
        }
        for entry in nonConditioning {
            history.store(
                stored(entry.value, memory: entry.memory), at: entry.frame, conditioning: false)
        }
        return history
    }

    private func slot(_ index: Int, of array: NDArray, stride: Int) -> [Float] {
        Array(flattenAsFloat(array)[(index * stride)..<((index + 1) * stride)])
    }

    @Test("Memories fill slots from zero, and a frame without encoded memory is skipped")
    func packsEligibleMemoriesIntoLeadingSlots() throws {
        let packer = try makePacker(parameters())
        // numMaskmem 3 gives recent offsets 2 then 1, i.e. frames 8 then 9. Frame 9 was
        // stored but never encoded, so it must not consume a slot.
        let result = try packer.pack(
            history: history(
                conditioning: [(0, 100)],
                nonConditioning: [(8, 8, true), (9, 9, false)]),
            objectIndex: 0, frameIndex: 10, totalFrames: 50, reverse: false)

        #expect(flattenAsFloat(result.spatialSlotOccupancy) == [1, 1, 0, 0, 0, 0, 0, 0])
        #expect(slot(0, of: result.spatialMemory, stride: Self.slotElements).allSatisfy { $0 == 100 })
        #expect(slot(1, of: result.spatialMemory, stride: Self.slotElements).allSatisfy { $0 == 8 })
        // Position encodings are written in lockstep with the features.
        #expect(
            slot(1, of: result.spatialMemoryPosition, stride: Self.slotElements)
                .allSatisfy { $0 == 8 })
    }

    @Test("A conditioning frame takes the last temporal row, a recent frame takes offset - 1")
    func writesTemporalIndices() throws {
        let packer = try makePacker(parameters())
        let result = try packer.pack(
            history: history(conditioning: [(0, 100)], nonConditioning: [(8, 8, true)]),
            objectIndex: 0, frameIndex: 10, totalFrames: 50, reverse: false)

        let indices = readNDArray(result.spatialTemporalIndex, as: Int32.self, count: Self.spatialSlots)
        // Offset 0 is Python's `[-1]`, which wraps to numMaskmem - 1 = 2. Offset 2 gives 1.
        // Getting this wrong reads a neighbouring row of the positional encoding, which is a
        // plausible-looking result rather than a crash.
        #expect(indices[0] == 2)
        #expect(indices[1] == 1)
        #expect(indices.dropFirst(2).allSatisfy { $0 == 0 })
    }

    @Test("Slots the previous object filled are cleared before the next object is packed")
    func clearsStaleSlotsBetweenObjects() throws {
        // The bank is shared across objects within a frame. Without the clear, object B
        // inherits object A's bytes in the slots B does not reach. The key mask hides them
        // from the graph, so this only ever surfaces as a parity divergence.
        let packer = try makePacker(parameters())
        _ = try packer.pack(
            history: history(
                conditioning: [(0, 100)],
                nonConditioning: [(8, 8, true), (9, 9, true)]),
            objectIndex: 0, frameIndex: 10, totalFrames: 50, reverse: false)

        let result = try packer.pack(
            history: history(conditioning: [(0, 55)]),
            objectIndex: 1, frameIndex: 10, totalFrames: 50, reverse: false)

        #expect(flattenAsFloat(result.spatialSlotOccupancy) == [1, 0, 0, 0, 0, 0, 0, 0])
        #expect(slot(0, of: result.spatialMemory, stride: Self.slotElements).allSatisfy { $0 == 55 })
        for stale in 1...2 {
            #expect(
                slot(stale, of: result.spatialMemory, stride: Self.slotElements)
                    .allSatisfy { $0 == 0 },
                "slot \(stale) still holds the previous object's memory")
            #expect(
                slot(stale, of: result.spatialMemoryPosition, stride: Self.slotElements)
                    .allSatisfy { $0 == 0 })
        }
    }

    @Test("Pointer slots carry the offset normalised by the pointer budget")
    func writesPointerTemporalPositions() throws {
        let packer = try makePacker(parameters())
        let result = try packer.pack(
            history: history(
                conditioning: [(2, 2)],
                nonConditioning: [(7, 7, true), (8, 8, true), (9, 9, true)]),
            objectIndex: 0, frameIndex: 10, totalFrames: 50, reverse: false)

        // Conditioning frame 2 is 8 back, then the contiguous look-back gives 1, 2, 3.
        // maxObjectPointers 4 makes the divisor 3.
        let positions = flattenAsFloat(result.pointerTemporalPosition)
        #expect(flattenAsFloat(result.pointerSlotOccupancy) == [1, 1, 1, 1])
        #expect(zip(positions, [8.0 / 3, 1.0 / 3, 2.0 / 3, 1]).allSatisfy { abs($0 - $1) < 1e-6 })
    }

    @Test("More pointers than slots keeps the temporally closest and drops the furthest")
    func trimsPointerOverflow() throws {
        let packer = try makePacker(parameters())
        // Five eligible pointers for four slots: conditioning frames 0 and 1 at offsets 10
        // and 9, then the look-back at 1, 2, 3. Offset 10 is furthest and loses.
        let result = try packer.pack(
            history: history(
                conditioning: [(0, 0), (1, 1)],
                nonConditioning: [(7, 7, true), (8, 8, true), (9, 9, true)]),
            objectIndex: 0, frameIndex: 10, totalFrames: 50, reverse: false)

        let positions = flattenAsFloat(result.pointerTemporalPosition)
        #expect(flattenAsFloat(result.pointerSlotOccupancy) == [1, 1, 1, 1])
        // Survivors stay in offset order rather than closeness order, so 9 leads.
        #expect(zip(positions, [3, 1.0 / 3, 2.0 / 3, 1]).allSatisfy { abs($0 - $1) < 1e-6 })
    }

    @Test("Pointer slots left over from a longer history are cleared")
    func clearsStalePointerSlots() throws {
        let packer = try makePacker(parameters())
        _ = try packer.pack(
            history: history(
                conditioning: [(2, 2)],
                nonConditioning: [(7, 7, true), (8, 8, true), (9, 9, true)]),
            objectIndex: 0, frameIndex: 10, totalFrames: 50, reverse: false)

        let result = try packer.pack(
            history: history(conditioning: [(2, 44)]),
            objectIndex: 1, frameIndex: 10, totalFrames: 50, reverse: false)

        #expect(flattenAsFloat(result.pointerSlotOccupancy) == [1, 0, 0, 0])
        #expect(slot(0, of: result.objectPointers, stride: Self.pointerElements).allSatisfy { $0 == 44 })
        for stale in 1...3 {
            #expect(
                slot(stale, of: result.objectPointers, stride: Self.pointerElements)
                    .allSatisfy { $0 == 0 },
                "pointer slot \(stale) still holds the previous object's pointer")
        }
    }
}
