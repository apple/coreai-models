// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Turns `detect`'s raw tensors into the detection set the tracker consumes.
///
/// Port of the scoring half of `Sam3VideoModel.run_detection` plus
/// `_merge_detections_from_prompts` and `nms_masks`.
enum DetectionDecoder {
    /// Decode one prompt's detections.
    ///
    /// The logits are read first and thresholded before any mask is touched. `pred_masks` is
    /// `[1, 200, 288, 288]`, 66 MB as `Float`, and only a handful of the 200 queries usually
    /// survive, so converting just their slices saves most of that per prompt per frame.
    static func decode(
        _ outputs: DetectOutputs,
        promptID: Int,
        maskSize: Int,
        parameters: VideoSegmentationParameters
    ) -> MergedDetections {
        let logits = flattenAsFloat(outputs.predictedLogits)
        let presence = flattenAsFloat(outputs.presenceLogits)
        // A single presence logit gates the whole prompt: `pred_probs * presence.sigmoid()`.
        let presenceScore = sigmoid(presence.first ?? 0)

        var probabilities = [Float](repeating: 0, count: logits.count)
        for index in logits.indices {
            probabilities[index] = sigmoid(logits[index]) * presenceScore
        }

        var candidates = probabilities.indices.filter {
            probabilities[$0] > parameters.scoreThresholdDetection
        }
        guard !candidates.isEmpty else { return MergedDetections() }

        let pixels = maskSize * maskSize
        var maskLogits = candidates.map { query in
            floatElements(outputs.predictedMasks, in: (query * pixels)..<((query + 1) * pixels))
        }
        var masks = maskLogits.map {
            MaskBitset(thresholding: $0, width: maskSize, height: maskSize)
        }

        if parameters.detNmsThresh > 0 {
            let keep = nonMaximumSuppression(
                masks: masks,
                scores: candidates.map { probabilities[$0] },
                iouThreshold: parameters.detNmsThresh)
            // Upstream zeroes the suppressed probabilities and re-thresholds, which is the
            // same as dropping them: `pred_probs[0][~keep] = 0.0` then
            // `pred_probs > score_threshold_detection`.
            candidates = keep.map { candidates[$0] }
            maskLogits = keep.map { maskLogits[$0] }
            masks = keep.map { masks[$0] }
        }

        return MergedDetections(
            maskLogits: maskLogits,
            masks: masks,
            scores: candidates.map { probabilities[$0] },
            promptIDs: [Int](repeating: promptID, count: candidates.count))
    }

    /// Concatenate per-prompt detections in prompt-id order.
    ///
    /// Port of `_merge_detections_from_prompts`. Order matters downstream: new objects are
    /// numbered by their position here, so a reordering would renumber every track.
    static func merge(_ perPrompt: [MergedDetections]) -> MergedDetections {
        var merged = MergedDetections()
        for detections in perPrompt {
            merged.maskLogits.append(contentsOf: detections.maskLogits)
            merged.masks.append(contentsOf: detections.masks)
            merged.scores.append(contentsOf: detections.scores)
            merged.promptIDs.append(contentsOf: detections.promptIDs)
        }
        return merged
    }

    /// Greedy mask-IoU non-maximum suppression, returning the surviving indices in input
    /// order.
    ///
    /// Port of `nms_masks`, which prefilters by score and then defers to
    /// `cv_utils_kernel.generic_nms`. The caller has already applied the score prefilter, so
    /// only the greedy pass remains.
    static func nonMaximumSuppression(
        masks: [MaskBitset], scores: [Float], iouThreshold: Float
    ) -> [Int] {
        precondition(masks.count == scores.count, "NMS: masks and scores must be parallel")
        // Ties broken by index so the result does not depend on sort stability.
        let order = scores.indices.sorted {
            scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
        }
        var suppressed = [Bool](repeating: false, count: masks.count)
        var kept: [Int] = []
        for candidate in order where !suppressed[candidate] {
            kept.append(candidate)
            for other in order where other != candidate && !suppressed[other] {
                if masks[candidate].iou(masks[other]) >= iouThreshold {
                    suppressed[other] = true
                }
            }
        }
        // Restore input order: downstream indexes detections positionally and expects the
        // prompt's own ordering, not the score ordering.
        return kept.sorted()
    }

    @inline(__always)
    static func sigmoid(_ x: Float) -> Float {
        // Branch on the sign so the exponent argument is never positive; `exp` of a large
        // positive logit overflows to infinity and turns the result into NaN.
        if x >= 0 {
            return 1 / (1 + Foundation.exp(-x))
        }
        let e = Foundation.exp(x)
        return e / (1 + e)
    }
}
