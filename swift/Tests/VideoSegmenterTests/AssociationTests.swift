// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAIShared
@testable import CoreAIVideoSegmenter

@Suite("Associator")
struct AssociatorTests {
    /// A 10x10 mask covering `columns` of every row, so IoU between two of these is a
    /// straightforward column overlap.
    private func stripe(_ columns: Range<Int>) -> MaskBitset {
        var mask = MaskBitset(width: 10, height: 10)
        for y in 0..<10 {
            for x in columns { mask[x, y] = true }
        }
        return mask
    }

    private func detections(
        _ entries: [(mask: MaskBitset, score: Float, prompt: Int)]
    ) -> MergedDetections {
        var merged = MergedDetections()
        for entry in entries {
            merged.masks.append(entry.mask)
            merged.maskLogits.append([])
            merged.scores.append(entry.score)
            merged.promptIDs.append(entry.prompt)
        }
        return merged
    }

    @Test("With no tracks, every detection is new regardless of score")
    func firstFrame() {
        // The empty-tracks branch skips the `newDetThresh` gate entirely, which is how the
        // first frame seeds tracks from detections that clear the detection threshold but
        // not the stricter new-object one.
        var parameters = VideoSegmentationParameters.default
        parameters.newDetThresh = 0.9
        let result = Associator.associate(
            detections: detections([(stripe(0..<5), 0.55, 0), (stripe(5..<10), 0.60, 0)]),
            trackMasks: [], trackIDs: [], trackPromptIDs: [], parameters: parameters)
        #expect(result.newDetectionIndices == [0, 1])
    }

    @Test("With no detections, non-empty tracks are unmatched and empty ones are not")
    func noDetections() {
        // The distinction drives keep-alive: an occluded track should not be penalised the
        // way a track the detector stopped confirming is.
        let result = Associator.associate(
            detections: MergedDetections(),
            trackMasks: [stripe(0..<5), MaskBitset(width: 10, height: 10)],
            trackIDs: [1, 2], trackPromptIDs: [0, 0],
            parameters: .default)
        #expect(result.unmatchedTrackIDs == [1])
        #expect(result.emptyTrackIDs == [2])
    }

    @Test("A detection covered by a track is not new, but may recondition it")
    func coveredDetection() {
        let result = Associator.associate(
            detections: detections([(stripe(0..<5), 0.95, 0)]),
            trackMasks: [stripe(0..<5)], trackIDs: [7], trackPromptIDs: [0],
            parameters: .default)
        #expect(result.newDetectionIndices.isEmpty)
        #expect(result.detectionToMatchedTrackIDs[0] == [7])
        #expect(result.unmatchedTrackIDs.isEmpty)
        // Confident and well-overlapping, so it is also a reconditioning candidate.
        #expect(result.trackIDToHighConfidenceDetection[7] == 0)
    }

    @Test("The loose and strict thresholds disagree independently")
    func twoThresholds() {
        // 2 of 10 columns shared: IoU is 2/8 = 0.25. Above the loose 0.1 association
        // threshold, so the detection is not new; below the strict 0.5 track threshold, so
        // the track still counts as unmatched. Both at once is the intended behaviour.
        let result = Associator.associate(
            detections: detections([(stripe(0..<4), 0.95, 0)]),
            trackMasks: [stripe(2..<6)], trackIDs: [1], trackPromptIDs: [0],
            parameters: .default)
        #expect(result.newDetectionIndices.isEmpty)
        #expect(result.unmatchedTrackIDs == [1])
    }

    @Test("Detections never associate across prompt groups")
    func crossPromptZeroed() {
        // Identical masks, different prompts. Without the IoU zeroing the "dog" detection
        // would be swallowed by the "person" track and never become its own object.
        let result = Associator.associate(
            detections: detections([(stripe(0..<5), 0.95, 1)]),
            trackMasks: [stripe(0..<5)], trackIDs: [1], trackPromptIDs: [0],
            parameters: .default)
        #expect(result.newDetectionIndices == [0])
        #expect(result.detectionToMatchedTrackIDs[0] == [])
        #expect(result.unmatchedTrackIDs == [1])
    }

    @Test("A new detection is never also a reconditioning candidate")
    func newDetectionsExcluded() {
        // `det_is_high_conf` is explicitly `& ~is_new_det` upstream: a detection cannot
        // both create a track and correct a different one.
        let result = Associator.associate(
            detections: detections([(stripe(0..<5), 0.99, 0)]),
            trackMasks: [stripe(8..<10)], trackIDs: [7], trackPromptIDs: [0],
            parameters: .default)
        #expect(result.newDetectionIndices == [0])
        #expect(result.trackIDToHighConfidenceDetection.isEmpty)
    }

    @Test("One detection can match several tracks, which is how duplicates are spotted")
    func manyToOne() {
        let result = Associator.associate(
            detections: detections([(stripe(0..<10), 0.95, 0)]),
            trackMasks: [stripe(0..<5), stripe(5..<10)], trackIDs: [1, 2],
            trackPromptIDs: [0, 0], parameters: .default)
        #expect(result.detectionToMatchedTrackIDs[0] == [1, 2])
    }
}

@Suite("OcclusionSuppressor")
struct OcclusionSuppressorTests {
    @Test("An object that loses most of its area to the argmax is blanked")
    func areaShrinkage() {
        // Object 1 is entirely inside object 0 and loses everywhere, so its retained
        // fraction is 0, below the 0.3 threshold. Object 0 keeps all four pixels.
        var logits: [[Float]] = [
            [5, 5, 5, 5],
            [1, 1, 1, 1],
        ]
        OcclusionSuppressor.suppressAreaShrinkage(logits: &logits, promptIDs: [0, 0])
        #expect(logits[0] == [5, 5, 5, 5])
        #expect(logits[1].allSatisfy { $0 <= OcclusionSuppressor.noObjectLogit })
    }

    @Test("An object that keeps enough area survives intact")
    func areaShrinkageSpares() {
        // Object 1 wins three of its four pixels, a 0.75 retention.
        var logits: [[Float]] = [
            [5, 1, 1, 1],
            [1, 5, 5, 5],
        ]
        OcclusionSuppressor.suppressAreaShrinkage(logits: &logits, promptIDs: [0, 0])
        #expect(logits[1] == [1, 5, 5, 5])
    }

    @Test("The rule is applied inside a prompt group, never across")
    func areaShrinkagePerPrompt() {
        var logits: [[Float]] = [
            [5, 5, 5, 5],
            [1, 1, 1, 1],
        ]
        OcclusionSuppressor.suppressAreaShrinkage(logits: &logits, promptIDs: [0, 1])
        #expect(logits[1] == [1, 1, 1, 1])
    }

    @Test("Contested pixels go to the highest-scoring object in the group")
    func nonOverlap() {
        var masks = [MaskBitset(width: 4, height: 1), MaskBitset(width: 4, height: 1)]
        for x in 0..<3 { masks[0][x, 0] = true }
        for x in 1..<4 { masks[1][x, 0] = true }
        OcclusionSuppressor.applyObjectWiseNonOverlap(
            masks: &masks, scores: [0.9, 0.4], promptIDs: [0, 0])
        #expect(masks[0].area == 3)
        #expect(masks[1].area == 1)
        #expect(masks[1][3, 0] == true)
    }

    @Test("Two prompts may claim the same pixels")
    func nonOverlapPerPrompt() {
        var masks = [MaskBitset(width: 4, height: 1), MaskBitset(width: 4, height: 1)]
        for x in 0..<4 {
            masks[0][x, 0] = true
            masks[1][x, 0] = true
        }
        OcclusionSuppressor.applyObjectWiseNonOverlap(
            masks: &masks, scores: [0.9, 0.4], promptIDs: [0, 1])
        #expect(masks[0].area == 4)
        #expect(masks[1].area == 4)
    }

    @Test("A zero-scored object loses even its uncontested pixels")
    func zeroScoreLosesEverything() {
        // Upstream tests `pixel_nonoverlap > 0` on a field built from the object's own
        // score, so a score of exactly 0 fails the test everywhere. Surprising, but a track
        // only reaches 0 when it has no recorded score at all.
        var masks = [MaskBitset(width: 4, height: 1), MaskBitset(width: 4, height: 1)]
        masks[0][0, 0] = true
        masks[1][3, 0] = true
        OcclusionSuppressor.applyObjectWiseNonOverlap(
            masks: &masks, scores: [0.9, 0.0], promptIDs: [0, 0])
        #expect(masks[0].area == 1)
        #expect(masks[1].area == 0)
    }
}
