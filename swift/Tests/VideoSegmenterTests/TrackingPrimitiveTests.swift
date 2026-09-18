// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Testing

@testable import CoreAIShared
@testable import CoreAIVideoSegmenter

@Suite("ObjectRegistry")
struct ObjectRegistryTests {
    @Test("Indices are assigned in first-sighting order and reused")
    func assignment() {
        var registry = ObjectRegistry()
        #expect(registry.index(of: 7).index == 0)
        #expect(registry.index(of: 7).isNew == false)
        #expect(registry.index(of: 3).index == 1)
        #expect(registry.ids == [7, 3])
    }

    @Test("Removing an object renumbers every index above it")
    func removalRenumbers() {
        // This is `remove_object`'s `_map_keys`. Leaving a hole instead would misattribute
        // every later object's memory bank, which is silent and catastrophic.
        var registry = ObjectRegistry()
        for id in [10, 11, 12, 13] { registry.index(of: id) }

        let survivors = registry.remove(11)
        #expect(survivors == [0, 2, 3])
        #expect(registry.ids == [10, 12, 13])
        #expect(registry.existingIndex(of: 12) == 1)
        #expect(registry.existingIndex(of: 13) == 2)
        #expect(registry.existingIndex(of: 11) == nil)
        // The survivor list is exactly what compacts a parallel array.
        #expect(survivors?.map { ["a", "b", "c", "d"][$0] } == ["a", "c", "d"])
    }

    @Test("Removing an unknown id is a no-op, not an error")
    func removeUnknown() {
        // `remove_object(strict=False)` returns quietly, and the frame loop relies on it:
        // an object can be removed by two heuristics on the same frame.
        var registry = ObjectRegistry()
        registry.index(of: 1)
        #expect(registry.remove(99) == nil)
        #expect(registry.ids == [1])
    }
}

@Suite("ConnectedComponents")
struct ConnectedComponentsTests {
    private func boolMask(_ rows: [String]) -> [Bool] {
        rows.flatMap { $0.map { $0 == "#" } }
    }

    @Test("Diagonal neighbours join, because cc_2d is 8-connected")
    func eightConnectivity() {
        // Under 4-connectivity these two pixels would be separate components of area 1 each,
        // and sprinkle removal would delete both.
        let mask = boolMask([
            "#..",
            ".#.",
            "...",
        ])
        let areas = ConnectedComponents.areas(of: mask, width: 3, height: 3)
        #expect(areas[0] == 2)
        #expect(areas[4] == 2)
        #expect(areas[1] == 0)
    }

    @Test("Separate blobs get their own areas")
    func separateComponents() {
        let mask = boolMask([
            "##...",
            "##...",
            ".....",
            "....#",
        ])
        let areas = ConnectedComponents.areas(of: mask, width: 5, height: 4)
        #expect(areas[0] == 4)
        #expect(areas[6] == 4)
        #expect(areas[19] == 1)
    }

    @Test("A background hole under the area limit is filled to a weak positive")
    func fillsHoles() {
        var logits = [Float](repeating: 1, count: 25)
        logits[12] = -1  // one background pixel enclosed by foreground
        ConnectedComponents.fillHoles(&logits, width: 5, height: 5, maxArea: 4)
        // Upstream writes 0.1, not a saturated value: the mask stays a logit field.
        #expect(logits[12] == 0.1)
    }

    @Test("A background region above the area limit is left alone")
    func leavesLargeBackground() {
        var logits = [Float](repeating: -1, count: 25)
        logits[0] = 1
        ConnectedComponents.fillHoles(&logits, width: 5, height: 5, maxArea: 4)
        #expect(logits[12] == -1)
    }

    @Test("A foreground speck is removed, and the object it sits beside is not")
    func removesSprinkles() {
        var logits = [Float](repeating: -1, count: 100)
        // A 6x6 block, plus one isolated pixel far away.
        for y in 0..<6 {
            for x in 0..<6 { logits[y * 10 + x] = 1 }
        }
        logits[99] = 1
        ConnectedComponents.fillHoles(&logits, width: 10, height: 10, maxArea: 4)
        #expect(logits[99] == -0.1)
        #expect(logits[0] == 1)
    }

    @Test("A tiny object does not delete itself")
    func halfAreaGuard() {
        // The foreground threshold is `min(maxArea, totalForeground / 2)`. Here the whole
        // mask is 3 pixels, so the threshold is 1 rather than the configured 16.
        var logits = [Float](repeating: -1, count: 100)
        logits[11] = 1
        logits[12] = 1
        logits[13] = 1
        ConnectedComponents.fillHoles(&logits, width: 10, height: 10, maxArea: 16)
        #expect(logits[11] == 1)
        #expect(logits[12] == 1)
    }

    @Test("A zero area limit disables both passes")
    func disabled() {
        var logits = [Float](repeating: -1, count: 25)
        logits[12] = 1
        let before = logits
        ConnectedComponents.fillHoles(&logits, width: 5, height: 5, maxArea: 0)
        #expect(logits == before)
    }
}

@Suite("DetectionDecoder")
struct DetectionDecoderTests {
    private func mask(_ rows: [String]) -> MaskBitset {
        var bitset = MaskBitset(width: rows[0].count, height: rows.count)
        for (y, row) in rows.enumerated() {
            for (x, character) in row.enumerated() where character == "#" {
                bitset[x, y] = true
            }
        }
        return bitset
    }

    @Test("sigmoid does not overflow on a large positive logit")
    func sigmoidStability() {
        // `1 / (1 + exp(-x))` is fine for large x but `exp(x) / (1 + exp(x))` overflows to
        // inf/inf = NaN. The detector's logits are unbounded, so the branch matters.
        #expect(DetectionDecoder.sigmoid(0) == 0.5)
        #expect(DetectionDecoder.sigmoid(100) == 1)
        // Not exactly zero: the negative branch keeps the denormal rather than underflowing.
        #expect(DetectionDecoder.sigmoid(-100) < 1e-40)
        #expect(DetectionDecoder.sigmoid(-100) >= 0)
        #expect(DetectionDecoder.sigmoid(-100).isNaN == false)
        #expect(abs(DetectionDecoder.sigmoid(2) - 0.880797) < 1e-5)
    }

    @Test("NMS keeps the highest-scoring member of an overlapping group")
    func nmsSuppresses() {
        let masks = [mask(["####"]), mask(["###."]), mask(["...#"])]
        let kept = DetectionDecoder.nonMaximumSuppression(
            masks: masks, scores: [0.9, 0.8, 0.7], iouThreshold: 0.5)
        // Index 1 overlaps index 0 at IoU 0.75 and is dropped; index 2 overlaps at 0.25
        // and survives.
        #expect(kept == [0, 2])
    }

    @Test("NMS returns survivors in input order, not score order")
    func nmsPreservesOrder() {
        // Detections are indexed positionally downstream, and new objects are numbered by
        // that position, so returning score order would renumber tracks.
        let masks = [mask(["#..."]), mask([".#.."]), mask(["..#."])]
        let kept = DetectionDecoder.nonMaximumSuppression(
            masks: masks, scores: [0.1, 0.9, 0.5], iouThreshold: 0.5)
        #expect(kept == [0, 1, 2])
    }

    @Test("Merging concatenates prompts in order and tags each detection")
    func merge() {
        var first = MergedDetections()
        first.scores = [0.9]
        first.promptIDs = [0]
        first.masks = [mask(["#."])]
        first.maskLogits = [[1, -1]]

        var second = MergedDetections()
        second.scores = [0.8, 0.7]
        second.promptIDs = [1, 1]
        second.masks = [mask([".#"]), mask(["##"])]
        second.maskLogits = [[-1, 1], [1, 1]]

        let merged = DetectionDecoder.merge([first, second])
        #expect(merged.count == 3)
        #expect(merged.promptIDs == [0, 1, 1])
        #expect(merged.scores == [0.9, 0.8, 0.7])
    }

    // MARK: - decode

    /// `decode` reads its tensors through `flattenAsFloat`/`floatElements` and never touches
    /// the engine, so the whole detector core runs on hand-built arrays.
    private func outputs(queries: [Float], presence: Float, masks: [[Float]]) -> DetectOutputs {
        var predictedLogits = NDArray(shape: [1, queries.count], scalarType: .float32)
        fillFloatNDArray(&predictedLogits, with: queries)
        var presenceLogits = NDArray(shape: [1], scalarType: .float32)
        fillFloatNDArray(&presenceLogits, with: [presence])
        var predictedMasks = NDArray(shape: [1, masks.count, 2, 2], scalarType: .float32)
        fillFloatNDArray(&predictedMasks, with: masks.flatMap { $0 })
        // `decode` ignores boxes; the tracker takes its boxes from the masks instead.
        return DetectOutputs(
            predictedMasks: predictedMasks,
            predictedBoxes: NDArray(shape: [1, masks.count, 4], scalarType: .float32),
            predictedLogits: predictedLogits,
            presenceLogits: presenceLogits)
    }

    private func parameters(threshold: Float, nms: Float) -> VideoSegmentationParameters {
        var parameters = VideoSegmentationParameters.default
        parameters.scoreThresholdDetection = threshold
        parameters.detNmsThresh = nms
        return parameters
    }

    @Test("A detection score is the query logit gated by the prompt's presence logit")
    func decodeScoresArePresenceGated() {
        // `pred_probs * presence.sigmoid()`. A plain `sigmoid(logit)` would put this at
        // 0.881 and silently admit detections upstream rejects.
        let decoded = DetectionDecoder.decode(
            outputs(queries: [2], presence: 1, masks: [[1, 1, -1, -1]]),
            promptID: 0, maskSize: 2, parameters: parameters(threshold: 0.1, nms: 0))
        #expect(decoded.count == 1)
        let expected = DetectionDecoder.sigmoid(2) * DetectionDecoder.sigmoid(1)
        #expect(abs(decoded.scores[0] - expected) < 1e-6)
        #expect(abs(expected - 0.643914) < 1e-5)
    }

    @Test("Only queries above the score threshold survive")
    func decodeFiltersByThreshold() {
        // Presence is high enough to be a near no-op, isolating the per-query filter.
        let decoded = DetectionDecoder.decode(
            outputs(
                queries: [3, -3, 1], presence: 8,
                masks: [[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1]]),
            promptID: 7, maskSize: 2, parameters: parameters(threshold: 0.5, nms: 0))
        // sigmoid(3) = 0.953 and sigmoid(1) = 0.731 clear 0.5; sigmoid(-3) = 0.047 does not.
        #expect(decoded.count == 2)
        #expect(decoded.promptIDs == [7, 7])
        #expect(decoded.maskLogits == [[1, -1, -1, -1], [-1, -1, 1, -1]])
    }

    @Test("The presence gate can reject a query its own logit would have passed")
    func decodePresenceGateCanReject() {
        // One logit, two prompts: the detector says "this looks like the object" while the
        // presence head says "the object is not in this frame". Upstream lets presence win.
        let masks = [[Float]]([[1, 1, -1, -1]])
        let confident = DetectionDecoder.decode(
            outputs(queries: [2], presence: 5, masks: masks),
            promptID: 0, maskSize: 2, parameters: parameters(threshold: 0.5, nms: 0))
        let absent = DetectionDecoder.decode(
            outputs(queries: [2], presence: -2, masks: masks),
            promptID: 0, maskSize: 2, parameters: parameters(threshold: 0.5, nms: 0))
        #expect(confident.count == 1)
        #expect(absent.count == 0)
    }

    @Test("An NMS drop re-indexes masks, logits, scores and prompt IDs together")
    func decodeKeepsArraysParallelAcrossNMS() {
        // The four arrays are re-indexed by hand after NMS. If one is missed, a surviving
        // detection keeps its own score while rendering the dropped query's mask.
        let decoded = DetectionDecoder.decode(
            outputs(
                queries: [3, 2, 1], presence: 8,
                // Queries 0 and 1 are the same mask (IoU 1.0); query 2 is disjoint.
                masks: [[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, -1, 1]]),
            promptID: 3, maskSize: 2, parameters: parameters(threshold: 0.1, nms: 0.5))

        #expect(decoded.count == 2)
        #expect(decoded.maskLogits.count == 2)
        #expect(decoded.masks.count == 2)
        #expect(decoded.promptIDs == [3, 3])
        // Query 1 loses to the higher-scoring query 0. The survivor at position 1 must be
        // query 2 in every array, not query 1 in some of them.
        #expect(decoded.maskLogits[1] == [-1, -1, -1, 1])
        #expect(decoded.masks[1].area == 1)
        let expected = [
            DetectionDecoder.sigmoid(3) * DetectionDecoder.sigmoid(8),
            DetectionDecoder.sigmoid(1) * DetectionDecoder.sigmoid(8),
        ]
        #expect(zip(decoded.scores, expected).allSatisfy { abs($0 - $1) < 1e-6 })
    }
}
