// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
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
