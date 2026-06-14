// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Testing

@testable import CoreAIObjectDetector

@Suite("ObjectDetector")
struct ObjectDetectorTests {
    // MARK: - DetectionPostprocessor guards

    @Test("decode returns empty for non-3D logitsShape")
    func decodeNon3DShape() {
        let output = DetectionOutput(logits: [1, 2, 3], logitsShape: [3], predictedBoxes: [])
        #expect(DetectionPostprocessor.decode(output: output, inputSize: .init(width: 100, height: 100)).isEmpty)
    }

    @Test("decode returns empty when queryCount is zero")
    func decodeZeroQueries() {
        let output = DetectionOutput(logits: [], logitsShape: [1, 0, 3], predictedBoxes: [])
        #expect(DetectionPostprocessor.decode(output: output, inputSize: .init(width: 100, height: 100)).isEmpty)
    }

    @Test("decode returns empty when classCount is one (no object classes)")
    func decodeSingleClass() {
        let output = DetectionOutput(logits: [1], logitsShape: [1, 1, 1], predictedBoxes: [0.5, 0.5, 0.2, 0.2])
        #expect(DetectionPostprocessor.decode(output: output, inputSize: .init(width: 100, height: 100)).isEmpty)
    }

    @Test("decode returns empty when logits count mismatches shape")
    func decodeLogitsSizeMismatch() {
        let output = DetectionOutput(
            logits: [1, 2, 3, 4],
            logitsShape: [1, 2, 3],  // expects 6
            predictedBoxes: Array(repeating: 0.5, count: 8)
        )
        #expect(DetectionPostprocessor.decode(output: output, inputSize: .init(width: 100, height: 100)).isEmpty)
    }

    @Test("decode returns empty when predictedBoxes count mismatches queries")
    func decodepredictedBoxesSizeMismatch() {
        let output = DetectionOutput(
            logits: Array(repeating: 1, count: 6),
            logitsShape: [1, 2, 3],  // 2 queries → 8 box values needed
            predictedBoxes: Array(repeating: 0.5, count: 4)
        )
        #expect(DetectionPostprocessor.decode(output: output, inputSize: .init(width: 100, height: 100)).isEmpty)
    }

    // MARK: - Threshold filtering

    @Test("decode filters detections below threshold")
    func decodeThresholdFiltering() {
        // query 0: class 0 wins with high confidence (~0.9999)
        // query 1: uniform → ~0.33 per class
        let logits: [Float] = [
            10.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]
        let predictedBoxes: [Float] = Array(repeating: 0.5, count: 8)
        let output = DetectionOutput(logits: logits, logitsShape: [1, 2, 3], predictedBoxes: predictedBoxes)
        let result = DetectionPostprocessor.decode(
            output: output,
            inputSize: .init(width: 100, height: 100),
            parameters: DetectionParameters(threshold: 0.5)
        )
        #expect(result.count == 1)
        #expect(result[0].labelIndex == 0)
    }

    @Test("decode returns empty when all scores are below threshold")
    func decodeAllBelowThreshold() {
        let logits: [Float] = [0.0, 0.0, 0.0]  // uniform softmax ~0.33 each
        let output = DetectionOutput(logits: logits, logitsShape: [1, 1, 3], predictedBoxes: [0.5, 0.5, 0.2, 0.2])
        let result = DetectionPostprocessor.decode(
            output: output,
            inputSize: .init(width: 100, height: 100),
            parameters: DetectionParameters(threshold: 0.9)
        )
        #expect(result.isEmpty)
    }

    // MARK: - Sorting and maxDetections

    @Test("decode results are sorted by score descending")
    func decodeSortedByScore() {
        let logits: [Float] = [
            5.0, 0.0, 0.0,  // query 0: moderate score
            10.0, 0.0, 0.0,  // query 1: higher score
        ]
        let predictedBoxes: [Float] = Array(repeating: 0.5, count: 8)
        let output = DetectionOutput(logits: logits, logitsShape: [1, 2, 3], predictedBoxes: predictedBoxes)
        let result = DetectionPostprocessor.decode(
            output: output,
            inputSize: .init(width: 100, height: 100),
            parameters: DetectionParameters(threshold: 0.0)
        )
        #expect(result.count == 2)
        #expect(result[0].confidence >= result[1].confidence)
    }

    @Test("decode respects maxDetections cap")
    func decodeMaxDetectionsCap() {
        let logits: [Float] = Array(repeating: 10.0, count: 9) + Array(repeating: 0.0, count: 6)
        let predictedBoxes: [Float] = Array(repeating: 0.5, count: 12)
        let output = DetectionOutput(logits: logits, logitsShape: [1, 3, 5], predictedBoxes: predictedBoxes)
        let result = DetectionPostprocessor.decode(
            output: output,
            inputSize: .init(width: 100, height: 100),
            parameters: DetectionParameters(threshold: 0.0, maxDetections: 2)
        )
        #expect(result.count == 2)
    }

    // MARK: - Box coordinate decoding

    @Test("decode converts normalized cx/cy/w/h to pixel top-left CGRect")
    func decodeBoxCoordinates() {
        // Centered at (0.5, 0.5), size 0.4×0.2 on a 100×200 canvas
        let logits: [Float] = [10.0, 0.0, 0.0]
        let predictedBoxes: [Float] = [0.5, 0.5, 0.4, 0.2]
        let output = DetectionOutput(logits: logits, logitsShape: [1, 1, 3], predictedBoxes: predictedBoxes)
        let result = DetectionPostprocessor.decode(
            output: output,
            inputSize: .init(width: 100, height: 200),
            parameters: DetectionParameters(threshold: 0.0)
        )
        let box = try! #require(result.first).boundingBox
        // x = (0.5 - 0.2) * 100 = 30
        #expect(abs(box.origin.x - 30) < 0.001)
        // y = (0.5 - 0.1) * 200 = 80
        #expect(abs(box.origin.y - 80) < 0.001)
        // w = 0.4 * 100 = 40
        #expect(abs(box.width - 40) < 0.001)
        // h = 0.2 * 200 = 40
        #expect(abs(box.height - 40) < 0.001)
    }

    // MARK: - Label lookup

    @Test("decode uses classLabels for known indices")
    func decodeKnownLabel() {
        let logits: [Float] = [10.0, 0.0, 0.0]
        let output = DetectionOutput(logits: logits, logitsShape: [1, 1, 3], predictedBoxes: [0.5, 0.5, 0.2, 0.2])
        let result = DetectionPostprocessor.decode(
            output: output,
            inputSize: .init(width: 100, height: 100),
            parameters: DetectionParameters(threshold: 0.0, classLabels: [0: "cat"])
        )
        #expect(result.first?.label == "cat")
    }

    @Test("decode falls back to class_N for unknown label index")
    func decodeUnknownLabelFallback() {
        let logits: [Float] = [10.0, 0.0, 0.0]
        let output = DetectionOutput(logits: logits, logitsShape: [1, 1, 3], predictedBoxes: [0.5, 0.5, 0.2, 0.2])
        let result = DetectionPostprocessor.decode(
            output: output,
            inputSize: .init(width: 100, height: 100),
            parameters: DetectionParameters(threshold: 0.0, classLabels: [:])
        )
        #expect(result.first?.label == "class_0")
    }

    @Test("decode excludes last (no-object) class from scoring")
    func decodeNoObjectClassExcluded() {
        // 2 classes: class 0 (object) + class 1 (no-object).
        // High logit on the no-object class → object score ≈ 0.
        let logits: [Float] = [0.0, 100.0]
        let output = DetectionOutput(logits: logits, logitsShape: [1, 1, 2], predictedBoxes: [0.5, 0.5, 0.2, 0.2])
        let result = DetectionPostprocessor.decode(
            output: output,
            inputSize: .init(width: 100, height: 100),
            parameters: DetectionParameters(threshold: 0.5)
        )
        #expect(result.isEmpty)
    }

    // MARK: - Softmax

    @Test("softmax output sums to 1")
    func softmaxSumsToOne() {
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let result = DetectionPostprocessor.softmax(input, offset: 0, count: 4)
        #expect(abs(result.reduce(0, +) - 1.0) < 1e-5)
    }

    @Test("softmax(0,0,0) produces uniform distribution")
    func softmaxUniform() {
        let input: [Float] = [0.0, 0.0, 0.0]
        let result = DetectionPostprocessor.softmax(input, offset: 0, count: 3)
        for p in result { #expect(abs(p - (1.0 / 3.0)) < 1e-5) }
    }

    @Test("softmax respects offset into the array")
    func softmaxOffset() {
        let input: [Float] = [99.0, 99.0, 0.0, 0.0]  // offset 2: uniform over [0,0]
        let result = DetectionPostprocessor.softmax(input, offset: 2, count: 2)
        #expect(result.count == 2)
        #expect(abs(result[0] - 0.5) < 1e-5)
        #expect(abs(result[1] - 0.5) < 1e-5)
    }

    @Test("softmax is numerically stable for large inputs")
    func softmaxNumericalStability() {
        let input: [Float] = [1000.0, 1000.0]
        let result = DetectionPostprocessor.softmax(input, offset: 0, count: 2)
        #expect(result.allSatisfy { $0.isFinite })
        #expect(abs(result[0] - 0.5) < 1e-5)
    }

    // MARK: - Batch planning

    @Test("planBatch: single image, dynamic dims, no overrides → parameter defaults")
    func planBatchSingleDefault() throws {
        let p = DetectionParameters()
        let plan = try ObjectDetector.planBatch(
            expectedShape: [-1, 3, -1, -1],
            imageCount: 1,
            parameters: .default
        )
        #expect(plan == ObjectDetector.BatchPlan(batch: 1, height: p.inputHeight, width: p.inputWidth))
    }

    @Test("planBatch: multi-image, dynamic dims, no overrides → parameter defaults")
    func planBatchMultiDefault() throws {
        let p = DetectionParameters()
        let plan = try ObjectDetector.planBatch(
            expectedShape: [-1, 3, -1, -1],
            imageCount: 3,
            parameters: .default
        )
        #expect(plan == ObjectDetector.BatchPlan(batch: 3, height: p.inputHeight, width: p.inputWidth))
    }

    @Test("planBatch: multi-image, dynamic dims, explicit overrides win")
    func planBatchMultiOverride() throws {
        var params = DetectionParameters.default
        params.inputHeight = 512
        params.inputWidth = 512
        let plan = try ObjectDetector.planBatch(
            expectedShape: [-1, 3, -1, -1],
            imageCount: 2,
            parameters: params
        )
        #expect(plan == ObjectDetector.BatchPlan(batch: 2, height: 512, width: 512))
    }

    @Test("planBatch: static spatial dims override parameter values silently")
    func planBatchStaticSpatialIgnoresParams() throws {
        // Static [1, 3, 800, 800] with mismatching params → planBatch uses
        // the static dims; parameter values are silently ignored for fixed axes.
        var params = DetectionParameters.default
        params.inputHeight = 512
        params.inputWidth = 512
        let plan = try ObjectDetector.planBatch(
            expectedShape: [1, 3, 800, 800],
            imageCount: 1,
            parameters: params
        )
        #expect(plan == ObjectDetector.BatchPlan(batch: 1, height: 800, width: 800))
    }

    @Test("planBatch: static batch mismatch throws (multi-image into batch=1 model)")
    func planBatchStaticBatchMismatchThrows() {
        #expect(throws: DetectionRuntimeError.self) {
            try ObjectDetector.planBatch(
                expectedShape: [1, 3, -1, -1],
                imageCount: 2,
                parameters: .default
            )
        }
    }
}
