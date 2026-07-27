// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Testing

@testable import CoreAIImageSegmenter

@Suite("CoreAISegmentationEngine")
struct CoreAISegmentationEngineTests {
    // MARK: - Input name discovery

    @Test("findImageInputName: matches 'pixel' and 'image' variants")
    func findImageInputName() {
        #expect(CoreAISegmentationEngine.findImageInputName(in: ["pixel_values", "input_ids"]) == "pixel_values")
        #expect(CoreAISegmentationEngine.findImageInputName(in: ["text_tokens", "image_input"]) == "image_input")
        #expect(CoreAISegmentationEngine.findImageInputName(in: ["text_tokens", "input_ids"]) == nil)
    }

    @Test("findTextInputName: matches 'input_id', 'token', and 'text' variants")
    func findTextInputName() {
        #expect(CoreAISegmentationEngine.findTextInputName(in: ["pixel_values", "input_ids"]) == "input_ids")
        #expect(CoreAISegmentationEngine.findTextInputName(in: ["image", "text_tokens"]) == "text_tokens")
        #expect(CoreAISegmentationEngine.findTextInputName(in: ["pixel_values", "embed_input"]) == nil)
    }

    @Test("findTextInputName: ignores text_features (multi-function detect sibling)")
    func findTextInputNameSkipsFeatures() {
        // input_ids is the token input.
        #expect(
            CoreAISegmentationEngine.findTextInputName(in: ["input_ids"]) == "input_ids"
        )
        // text_features is a `detect` input, not a text input — must not be picked.
        #expect(
            CoreAISegmentationEngine.findTextInputName(in: ["backbone_features", "text_features"]) == nil
        )
    }

    @Test("findBackboneFeaturesName: matches outputs/inputs containing 'backbone'")
    func findBackboneFeaturesName() {
        #expect(
            CoreAISegmentationEngine.findBackboneFeaturesName(in: ["backbone_features"])
                == "backbone_features"
        )
        #expect(
            CoreAISegmentationEngine.findBackboneFeaturesName(in: ["text_features", "backbone_features"])
                == "backbone_features"
        )
        #expect(CoreAISegmentationEngine.findBackboneFeaturesName(in: ["pred_masks"]) == nil)
    }

    @Test("findTextFeaturesName: matches 'text_features' / 'text_feat' but not unrelated 'text' inputs")
    func findTextFeaturesName() {
        #expect(
            CoreAISegmentationEngine.findTextFeaturesName(in: ["backbone_features", "text_features"])
                == "text_features"
        )
        #expect(CoreAISegmentationEngine.findTextFeaturesName(in: ["text_feat"]) == "text_feat")
        // 'text_tokens' is the text input, not the text-features intermediate — must not match.
        #expect(CoreAISegmentationEngine.findTextFeaturesName(in: ["text_tokens", "input_ids"]) == nil)
    }

    @Test("findPointsInputName: matches 'point' but excludes 'point_label'")
    func findPointsInputName() {
        let inputs = ["batched_images", "batched_points", "batched_point_labels"]
        #expect(CoreAISegmentationEngine.findPointsInputName(in: inputs) == "batched_points")
        #expect(CoreAISegmentationEngine.findPointsInputName(in: ["pixel_values"]) == nil)
    }

    @Test("findPointLabelsInputName: matches names with both 'point' and 'label'")
    func findPointLabelsInputName() {
        let inputs = ["batched_images", "batched_points", "batched_point_labels"]
        #expect(CoreAISegmentationEngine.findPointLabelsInputName(in: inputs) == "batched_point_labels")
        // Without 'point', should not match.
        #expect(CoreAISegmentationEngine.findPointLabelsInputName(in: ["labels", "input_ids"]) == nil)
    }

    // MARK: - Output name discovery

    @Test("findLogitsOutputName: skips presence_logits, picks pred_logits")
    func findLogitsOutputNameSkipsPresence() {
        let outputs = ["pred_masks", "pred_boxes", "pred_logits", "presence_logits", "semantic_seg"]
        #expect(CoreAISegmentationEngine.findLogitsOutputName(in: outputs) == "pred_logits")
        // Order shouldn't matter
        let reversed = outputs.reversed() as [String]
        #expect(CoreAISegmentationEngine.findLogitsOutputName(in: reversed) == "pred_logits")
    }

    @Test("findPresenceOutputName: picks presence_logits and not pred_logits")
    func findPresenceOutputName() {
        let outputs = ["pred_logits", "presence_logits", "pred_masks"]
        #expect(CoreAISegmentationEngine.findPresenceOutputName(in: outputs) == "presence_logits")
        #expect(CoreAISegmentationEngine.findPresenceOutputName(in: ["pred_logits"]) == nil)
    }

    @Test("findIouScoresOutputName: matches 'iou' or 'score' (skipping 'logit')")
    func findIouScoresOutputName() {
        #expect(CoreAISegmentationEngine.findIouScoresOutputName(in: ["pred_masks", "iou_scores"]) == "iou_scores")
        #expect(CoreAISegmentationEngine.findIouScoresOutputName(in: ["mask_scores"]) == "mask_scores")
        // 'pred_logits' contains "score"-less but ends in "logit" — should not be picked.
        #expect(CoreAISegmentationEngine.findIouScoresOutputName(in: ["pred_logits", "presence_logits"]) == nil)
    }

    // MARK: - extractBoxesFromPointQuery

    @Test("extractBoxesFromPointQuery: empty queries → empty output")
    func extractBoxesEmpty() {
        let pointQuery = PointQuery()
        let boxes = CoreAISegmentationEngine.extractBoxesFromPointQuery(
            pointQuery, imageSize: CGSize(width: 100, height: 100)
        )
        #expect(boxes.isEmpty)
    }

    @Test("extractBoxesFromPointQuery: single box query produces normalized [x0,y0,x1,y1]")
    func extractBoxesSingleQuery() {
        let pointQuery = PointQuery(points: [
            .init(x: 25, y: 50, label: .boxTopLeft),
            .init(x: 75, y: 100, label: .boxBottomRight),
        ])
        let boxes = CoreAISegmentationEngine.extractBoxesFromPointQuery(
            pointQuery, imageSize: CGSize(width: 100, height: 200)
        )
        #expect(boxes.count == 4)
        #expect(abs(boxes[0] - 0.25) < 1e-6)
        #expect(abs(boxes[1] - 0.25) < 1e-6)
        #expect(abs(boxes[2] - 0.75) < 1e-6)
        #expect(abs(boxes[3] - 0.50) < 1e-6)
    }

    @Test("extractBoxesFromPointQuery: query without TL/BR pair zeros that slot")
    func extractBoxesPartialPair() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 0, y: 0, label: .boxTopLeft), .init(x: 50, y: 50, label: .boxBottomRight)],
            [.init(x: 30, y: 30, label: .foreground)],  // no box pair
        ])
        let boxes = CoreAISegmentationEngine.extractBoxesFromPointQuery(
            pointQuery, imageSize: CGSize(width: 100, height: 100)
        )
        #expect(boxes.count == 8)
        // Query 0 has the box.
        #expect(boxes[2] == 0.5 && boxes[3] == 0.5)
        // Query 1 should be zeroed.
        #expect(boxes[4] == 0 && boxes[5] == 0 && boxes[6] == 0 && boxes[7] == 0)
    }

    @Test("extractBoxesFromPointQuery: zero imageSize is treated as empty (no NaN/inf)")
    func extractBoxesZeroSize() {
        let pointQuery = PointQuery(points: [
            .init(x: 0, y: 0, label: .boxTopLeft),
            .init(x: 10, y: 10, label: .boxBottomRight),
        ])
        let boxes = CoreAISegmentationEngine.extractBoxesFromPointQuery(pointQuery, imageSize: .zero)
        #expect(boxes.isEmpty)
    }

    // MARK: - sliceUserQueries

    @Test("sliceUserQueries: trims phantom slots from masks + scores")
    func sliceUserQueriesUnderfill() {
        // [B=1, Q=3, H=2, W=2] — 12 values. Per-query: query 0 = 1.0s, query 1 = 2.0s, query 2 = 3.0s.
        let masks: [Float] = [
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
        ]
        let scores: [Float] = [0.7, 0.8, 0.9]
        let (outMasks, outShape, outScores) = CoreAISegmentationEngine.sliceUserQueries(
            flatMasks: masks, flatScores: scores, shape: [1, 3, 2, 2], userQueryCount: 1
        )
        #expect(outShape == [1, 1, 2, 2])
        #expect(outMasks == [1, 1, 1, 1])
        #expect(outScores == [0.7])
    }

    @Test("sliceUserQueries: no-op when userQueryCount == Q")
    func sliceUserQueriesNoOp() {
        let masks: [Float] = [1, 2, 3, 4]
        let scores: [Float] = [0.5, 0.6]
        let (outMasks, outShape, outScores) = CoreAISegmentationEngine.sliceUserQueries(
            flatMasks: masks, flatScores: scores, shape: [1, 2, 1, 2], userQueryCount: 2
        )
        #expect(outShape == [1, 2, 1, 2])
        #expect(outMasks == masks)
        #expect(outScores == scores)
    }

    // MARK: - Token value indexing

    @Test("tokenValue: reads tokens in row-major order, pads out-of-range with EOT")
    func tokenValueIndexing() {
        let eot: Int32 = 49407
        let batch: [[Int32]] = [[10, 20, 30], [40, 50, 60]]
        let sequenceLength = 5

        // In-range reads
        #expect(
            CoreAISegmentationEngine.tokenValue(at: 0, sequenceLength: sequenceLength, batch: batch, eotTokenId: eot)
                == 10)
        #expect(
            CoreAISegmentationEngine.tokenValue(at: 2, sequenceLength: sequenceLength, batch: batch, eotTokenId: eot)
                == 30)
        #expect(
            CoreAISegmentationEngine.tokenValue(at: 5, sequenceLength: sequenceLength, batch: batch, eotTokenId: eot)
                == 40)

        // Token index past the end of a sequence → EOT padding
        #expect(
            CoreAISegmentationEngine.tokenValue(at: 3, sequenceLength: sequenceLength, batch: batch, eotTokenId: eot)
                == eot)

        // Batch index out of range → EOT padding
        #expect(
            CoreAISegmentationEngine.tokenValue(at: 10, sequenceLength: sequenceLength, batch: batch, eotTokenId: eot)
                == eot)
    }

    // MARK: - Embedding value indexing

    @Test("embeddingValue: reads [batch, seq, hidden] flat in C-order, zero-pads out-of-range")
    func embeddingValueIndexing() {
        // batch=1, sequenceLength=2, hiddenSize=3: flat layout [b][s][h]
        let batch: [[[Float]]] = [[[1, 2, 3], [4, 5, 6]]]
        let sequenceLength = 2
        let hiddenSize = 3

        // b=0, s=0, h=0..2
        #expect(
            CoreAISegmentationEngine.embeddingValue(
                at: 0, sequenceLength: sequenceLength, hiddenSize: hiddenSize, batch: batch) == 1)
        #expect(
            CoreAISegmentationEngine.embeddingValue(
                at: 2, sequenceLength: sequenceLength, hiddenSize: hiddenSize, batch: batch) == 3)
        // b=0, s=1, h=0
        #expect(
            CoreAISegmentationEngine.embeddingValue(
                at: 3, sequenceLength: sequenceLength, hiddenSize: hiddenSize, batch: batch) == 4)

        // Sequence index out of range → 0
        let batchShort: [[[Float]]] = [[[1, 2, 3]]]  // only 1 seq token, sequenceLength=2
        #expect(
            CoreAISegmentationEngine.embeddingValue(
                at: 3, sequenceLength: sequenceLength, hiddenSize: hiddenSize, batch: batchShort)
                == 0)

        // Batch index out of range → 0
        #expect(
            CoreAISegmentationEngine.embeddingValue(
                at: 6, sequenceLength: sequenceLength, hiddenSize: hiddenSize, batch: batch) == 0)
    }

    // MARK: - reduceBestOfK

    @Test("reduceBestOfK: picks the highest-scoring K per (B, Q) and copies its mask slab")
    func reduceBestOfKHappyPath() {
        // [B=1, Q=2, K=3, H=2, W=2]: 12 mask floats per (B, Q), 3 candidates each.
        // Distinguish candidates by filling each with a constant: 0.10 for K=0, 0.20 for K=1, 0.30 for K=2.
        let pixelCount = 4
        var flatMasks = [Float]()
        for _ in 0..<2 {  // 2 queries
            flatMasks.append(contentsOf: Array(repeating: Float(0.10), count: pixelCount))
            flatMasks.append(contentsOf: Array(repeating: Float(0.20), count: pixelCount))
            flatMasks.append(contentsOf: Array(repeating: Float(0.30), count: pixelCount))
        }
        // Q0: scores [0.1, 0.9, 0.2] → argmax = K=1 (mask filled with 0.20)
        // Q1: scores [0.5, 0.4, 0.6] → argmax = K=2 (mask filled with 0.30)
        let flatScores: [Float] = [0.1, 0.9, 0.2, 0.5, 0.4, 0.6]

        let result = CoreAISegmentationEngine.reduceBestOfK(
            flatMasks: flatMasks, flatScores: flatScores, shape: [1, 2, 3, 2, 2]
        )

        #expect(result.shape == [1, 2, 2, 2])
        #expect(result.masks.count == 1 * 2 * pixelCount)
        // Q0 slab is the K=1 fill (0.20).
        #expect(result.masks[0..<pixelCount].allSatisfy { abs($0 - 0.20) < 1e-6 })
        // Q1 slab is the K=2 fill (0.30).
        #expect(result.masks[pixelCount..<(2 * pixelCount)].allSatisfy { abs($0 - 0.30) < 1e-6 })
        #expect(result.scores.count == 2)
        #expect(abs(result.scores[0] - 0.9) < 1e-6)
        #expect(abs(result.scores[1] - 0.6) < 1e-6)
    }

    @Test("reduceBestOfK: ties go to the first candidate (stable argmax)")
    func reduceBestOfKTieBreaking() {
        // K=3 with all-equal scores → bestCandidate stays 0.
        let flatMasks: [Float] = [
            7, 7,  // K=0
            8, 8,  // K=1
            9, 9,  // K=2
        ]
        let flatScores: [Float] = [0.5, 0.5, 0.5]
        let result = CoreAISegmentationEngine.reduceBestOfK(
            flatMasks: flatMasks, flatScores: flatScores, shape: [1, 1, 3, 1, 2]
        )
        #expect(result.masks == [7, 7])
        #expect(result.scores == [0.5])
    }

    @Test("reduceBestOfK: B=2 keeps batches independent")
    func reduceBestOfKMultipleBatches() {
        // [B=2, Q=1, K=2, H=1, W=1] → 1 float per slab.
        let flatMasks: [Float] = [
            10, 20,  // batch 0: K=0=10, K=1=20
            30, 40,  // batch 1: K=0=30, K=1=40
        ]
        // batch 0: K=1 wins; batch 1: K=0 wins.
        let flatScores: [Float] = [0.1, 0.9, 0.8, 0.2]
        let result = CoreAISegmentationEngine.reduceBestOfK(
            flatMasks: flatMasks, flatScores: flatScores, shape: [2, 1, 2, 1, 1]
        )
        #expect(result.shape == [2, 1, 1, 1])
        #expect(result.masks == [20, 30])
        #expect(result.scores == [0.9, 0.8])
    }

    // MARK: - resolveQueries

    @Test("resolveQueries: empty queries fan out to a gridSide×gridSide foreground grid")
    func resolveQueriesSegmentEverything() throws {
        let imageWidth: Float = 100
        let imageHeight: Float = 200
        let resolved = try CoreAISegmentationEngine.resolveQueries(
            PointQuery(),
            queryCount: 4,  // gridSide = 2
            pointsPerQuery: 1,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )

        #expect(resolved.count == 4)
        // Each query is a single foreground point at the cell center: (col+0.5)/2 * W, (row+0.5)/2 * H.
        let expected: [(Float, Float)] = [
            (0.25 * imageWidth, 0.25 * imageHeight),  // row=0, col=0
            (0.75 * imageWidth, 0.25 * imageHeight),  // row=0, col=1
            (0.25 * imageWidth, 0.75 * imageHeight),  // row=1, col=0
            (0.75 * imageWidth, 0.75 * imageHeight),  // row=1, col=1
        ]
        for (idx, q) in resolved.enumerated() {
            #expect(q.count == 1)
            #expect(q[0].label == .foreground)
            #expect(abs(q[0].x - expected[idx].0) < 1e-4)
            #expect(abs(q[0].y - expected[idx].1) < 1e-4)
        }
    }

    @Test("resolveQueries: non-square queryCount in segment-everything throws")
    func resolveQueriesNonSquareThrows() {
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                PointQuery(),
                queryCount: 5,
                pointsPerQuery: 1,
                imageWidth: 100,
                imageHeight: 100
            )
        }
    }

    @Test("resolveQueries: too many queries vs model capacity throws")
    func resolveQueriesTooManyQueriesThrows() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 1, y: 1)],
            [.init(x: 2, y: 2)],
            [.init(x: 3, y: 3)],
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery,
                queryCount: 2,  // model only supports 2
                pointsPerQuery: 4,
                imageWidth: 100,
                imageHeight: 100
            )
        }
    }

    @Test("resolveQueries: too many points in a query throws")
    func resolveQueriesTooManyPointsThrows() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 1, y: 1), .init(x: 2, y: 2), .init(x: 3, y: 3)]
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery,
                queryCount: 4,
                pointsPerQuery: 2,  // model only supports 2 points/query
                imageWidth: 100,
                imageHeight: 100
            )
        }
    }

    @Test("resolveQueries: well-formed user queries pass through unchanged")
    func resolveQueriesPassThrough() throws {
        let pointQuery = PointQuery(queries: [
            [.init(x: 10, y: 20, label: .foreground)],
            [.init(x: 30, y: 40, label: .background)],
        ])
        let resolved = try CoreAISegmentationEngine.resolveQueries(
            pointQuery,
            queryCount: 4,
            pointsPerQuery: 4,
            imageWidth: 100,
            imageHeight: 100
        )
        #expect(resolved.count == 2)
        #expect(resolved[0][0].x == 10 && resolved[0][0].y == 20)
        #expect(resolved[1][0].label == .background)
    }

    @Test("resolveQueries: empty inner query throws")
    func resolveQueriesEmptyQueryThrows() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 1, y: 1, label: .foreground)],
            [],  // empty
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        }
    }

    @Test("resolveQueries: point outside image bounds throws")
    func resolveQueriesOutOfBoundsThrows() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 150, y: 50, label: .foreground)]  // x past width
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        }
    }

    @Test("resolveQueries: negative coordinate throws")
    func resolveQueriesNegativeCoordThrows() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 50, y: -1, label: .foreground)]
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        }
    }

    @Test("resolveQueries: NaN coordinate throws")
    func resolveQueriesNaNCoordThrows() {
        let pointQuery = PointQuery(queries: [
            [.init(x: .nan, y: 50, label: .foreground)]
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        }
    }

    @Test("resolveQueries: points exactly on image edges are allowed")
    func resolveQueriesEdgeCoordsAllowed() throws {
        let pointQuery = PointQuery(queries: [
            [.init(x: 0, y: 0, label: .foreground)],
            [.init(x: 100, y: 100, label: .foreground)],
        ])
        let resolved = try CoreAISegmentationEngine.resolveQueries(
            pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        #expect(resolved.count == 2)
    }

    @Test("resolveQueries: lone box-top-left without box-bottom-right throws")
    func resolveQueriesLoneTopLeftThrows() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 10, y: 10, label: .boxTopLeft)]
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        }
    }

    @Test("resolveQueries: lone box-bottom-right without box-top-left throws")
    func resolveQueriesLoneBottomRightThrows() {
        let pointQuery = PointQuery(queries: [
            [.init(x: 90, y: 90, label: .boxBottomRight)]
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        }
    }

    @Test("resolveQueries: multiple box-top-left in one query throws")
    func resolveQueriesMultipleTopLeftThrows() {
        let pointQuery = PointQuery(queries: [
            [
                .init(x: 10, y: 10, label: .boxTopLeft),
                .init(x: 20, y: 20, label: .boxTopLeft),
                .init(x: 90, y: 90, label: .boxBottomRight),
            ]
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        }
    }

    @Test("resolveQueries: multiple box-bottom-right in one query throws")
    func resolveQueriesMultipleBottomRightThrows() {
        let pointQuery = PointQuery(queries: [
            [
                .init(x: 10, y: 10, label: .boxTopLeft),
                .init(x: 80, y: 80, label: .boxBottomRight),
                .init(x: 90, y: 90, label: .boxBottomRight),
            ]
        ])
        #expect(throws: SegmentationRuntimeError.self) {
            _ = try CoreAISegmentationEngine.resolveQueries(
                pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        }
    }

    @Test("resolveQueries: box pair plus refinement clicks is allowed")
    func resolveQueriesBoxWithRefinementAllowed() throws {
        let pointQuery = PointQuery(queries: [
            [
                .init(x: 10, y: 10, label: .boxTopLeft),
                .init(x: 90, y: 90, label: .boxBottomRight),
                .init(x: 50, y: 50, label: .foreground),
                .init(x: 80, y: 20, label: .background),
            ]
        ])
        let resolved = try CoreAISegmentationEngine.resolveQueries(
            pointQuery, queryCount: 4, pointsPerQuery: 4, imageWidth: 100, imageHeight: 100)
        #expect(resolved.count == 1)
        #expect(resolved[0].count == 4)
    }
}
