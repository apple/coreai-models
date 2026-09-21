// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAIShared
@testable import CoreAIVideoSegmenter

/// The opt-in low-resolution mask channel. The contract worth pinning is that
/// `lowResolutionMasks` is index-aligned with `objects`, because the two are appended in
/// separate places and every `continue` in the loop has to skip both.
@Suite("MaskPostprocessor low-resolution masks")
@VideoSegmentationActor
struct LowResolutionMaskTests {
    private static let lowResolutionSide = 4
    private static let videoSide = 8

    /// A constant logit field: bilinear upsampling reproduces `value` everywhere, so a
    /// positive value gives a full mask and a negative one gives an empty mask.
    private func logits(_ value: Float) -> [Float] {
        [Float](repeating: value, count: Self.lowResolutionSide * Self.lowResolutionSide)
    }

    private func makeSession(_ objects: [(id: Int, prompt: String)]) -> VideoInferenceSession {
        let session = VideoInferenceSession(
            videoWidth: Self.videoSide, videoHeight: Self.videoSide)
        for object in objects {
            session.index(ofObject: object.id)
            session.promptIDByObjectID[object.id] = session.addPrompt(object.prompt)
        }
        return session
    }

    private func raw(_ masks: [Int: [Float]], suppressed: Set<Int> = []) -> RawFrameOutput {
        RawFrameOutput(
            frameIndex: 0,
            maskLogitsByObjectID: masks,
            scoreByObjectID: masks.mapValues { _ in 0.9 },
            trackerScoreByObjectID: masks.mapValues { _ in 0.8 },
            suppressedObjectIDs: suppressed)
    }

    private func postprocessor(emit: Bool) -> MaskPostprocessor {
        MaskPostprocessor(
            lowResolutionSize: Self.lowResolutionSide,
            videoWidth: Self.videoSide, videoHeight: Self.videoSide,
            emitLowResolutionMasks: emit)
    }

    @Test("Off by default, so nothing is emitted even when objects survive")
    func offByDefault() {
        let session = makeSession([(10, "cat")])
        let result = MaskPostprocessor(
            lowResolutionSize: Self.lowResolutionSide,
            videoWidth: Self.videoSide, videoHeight: Self.videoSide
        ).postprocess(raw([10: logits(1)]), session: session)

        #expect(result.objects.map(\.id) == [10])
        #expect(result.lowResolutionMasks.isEmpty)
    }

    @Test("On, one entry per surviving object in the order `objects` reports")
    func oneEntryPerSurvivingObject() {
        let session = makeSession([(10, "cat"), (40, "dog")])
        let result = postprocessor(emit: true)
            .postprocess(raw([10: logits(10), 40: logits(40)]), session: session)

        #expect(result.objects.map(\.id) == [10, 40])
        #expect(result.lowResolutionMasks.count == result.objects.count)
        #expect(result.lowResolutionMasks[0].allSatisfy { $0 == 10 })
        #expect(result.lowResolutionMasks[1].allSatisfy { $0 == 40 })
    }

    @Test("What comes back is the pre-upsample logit field, not the upsampled mask")
    func emitsPreUpsampleLogits() {
        let session = makeSession([(10, "cat")])
        let result = postprocessor(emit: true)
            .postprocess(raw([10: logits(3)]), session: session)

        #expect(result.lowResolutionMasks.count == 1)
        // 16 low-resolution logits in, an 8x8 mask out. Emitting the upsampled field
        // instead would give 64 values and defeat the point of the flag.
        #expect(result.lowResolutionMasks[0] == logits(3))
        #expect(result.objects[0].mask.width == Self.videoSide)
    }

    @Test("A dropped object leaves no hole, so the two arrays stay index-aligned")
    func droppedObjectsKeepArraysAligned() {
        let session = makeSession([
            (10, "cat"), (20, "dog"), (30, "bird"), (40, "fish"), (50, "hat"),
        ])
        session.hotstartRemovedObjectIDs = [50]
        let result = postprocessor(emit: true).postprocess(
            raw(
                [
                    10: logits(10),
                    20: logits(20),  // suppressed this frame
                    30: logits(-1),  // upsamples to an empty mask
                    40: logits(40),
                    50: logits(50),  // removed by hotstart
                ],
                suppressed: [20]),
            session: session)

        #expect(result.objects.map(\.id) == [10, 40])
        #expect(result.lowResolutionMasks.count == 2)
        // Each surviving object's logits must sit at its own index, not at the index it
        // would have had before the three drops.
        for (index, object) in result.objects.enumerated() {
            #expect(result.lowResolutionMasks[index].allSatisfy { $0 == Float(object.id) })
        }
    }

    @Test("Nothing survives, so both arrays come back empty")
    func nothingSurvives() {
        let session = makeSession([(10, "cat")])
        let result = postprocessor(emit: true)
            .postprocess(raw([10: logits(-1)]), session: session)

        #expect(result.objects.isEmpty)
        #expect(result.lowResolutionMasks.isEmpty)
    }
}

/// Boxes are taken from the masks *before* `applyObjectWiseNonOverlap` eats the loser's
/// contested pixels, so a box can be wider than the mask it labels. The two steps are
/// adjacent lines in `postprocess` and swapping them compiles, runs, and changes the
/// rendered output — hence a test rather than only the comment that is already there.
@Suite("MaskPostprocessor box and overlap ordering")
@VideoSegmentationActor
struct BoxBeforeNonOverlapTests {
    private static let lowResolutionSide = 4
    private static let videoSide = 8

    /// One value per low-resolution column, repeated down every row, so the upsampled mask
    /// is a band of full-height columns and the overlap is easy to reason about.
    private func columns(_ values: [Float]) -> [Float] {
        var field: [Float] = []
        for _ in 0..<Self.lowResolutionSide { field.append(contentsOf: values) }
        return field
    }

    /// Both objects share a prompt, which is what puts them in the same contending group.
    private func makeSession(_ ids: [Int]) -> VideoInferenceSession {
        let session = VideoInferenceSession(
            videoWidth: Self.videoSide, videoHeight: Self.videoSide)
        for id in ids {
            session.index(ofObject: id)
            session.promptIDByObjectID[id] = session.addPrompt("cat")
        }
        return session
    }

    private func postprocess(
        _ masks: [Int: [Float]], trackerScores: [Int: Float]
    ) -> MaskPostprocessor.Postprocessed {
        let session = makeSession(masks.keys.sorted())
        let raw = RawFrameOutput(
            frameIndex: 0,
            maskLogitsByObjectID: masks,
            scoreByObjectID: masks.mapValues { _ in 0.9 },
            trackerScoreByObjectID: trackerScores,
            suppressedObjectIDs: [])
        return MaskPostprocessor(
            lowResolutionSize: Self.lowResolutionSide,
            videoWidth: Self.videoSide, videoHeight: Self.videoSide
        ).postprocess(raw, session: session)
    }

    @Test("The loser of an overlap keeps the box it had before its mask was eaten")
    func boxPredatesOverlapResolution() throws {
        // Two overlapping bands in one prompt group. 20 has the lower tracker score, so the
        // pixel-level argmax hands the contested columns to 10.
        let left = columns([4, 4, 4, -4])
        let right = columns([-4, 4, 4, 4])

        let contested = postprocess(
            [10: left, 20: right], trackerScores: [10: 0.9, 20: 0.1])
        // The same object with no one to lose to, as the reference for "unshrunk".
        let alone = postprocess([20: right], trackerScores: [20: 0.1])

        #expect(contested.objects.count == 2)
        let loser = try #require(contested.objects.first { $0.id == 20 })
        let reference = try #require(alone.objects.first { $0.id == 20 })

        // The mask really did shrink, otherwise the box assertion below proves nothing.
        #expect(loser.mask.area < reference.mask.area)
        // ...but the box is byte-for-byte the one it would have had alone. Computing boxes
        // after the suppression would tighten this onto the surviving columns.
        #expect(loser.box == reference.box)
        #expect(loser.box.width > loser.mask.boundingBox.width)
    }

    @Test("The winner of an overlap is unaffected in both mask and box")
    func winnerIsUntouched() throws {
        let left = columns([4, 4, 4, -4])
        let right = columns([-4, 4, 4, 4])

        let contested = postprocess(
            [10: left, 20: right], trackerScores: [10: 0.9, 20: 0.1])
        let alone = postprocess([10: left], trackerScores: [10: 0.9])

        let winner = try #require(contested.objects.first { $0.id == 10 })
        let reference = try #require(alone.objects.first { $0.id == 10 })
        #expect(winner.mask.area == reference.mask.area)
        #expect(winner.box == reference.box)
    }
}
