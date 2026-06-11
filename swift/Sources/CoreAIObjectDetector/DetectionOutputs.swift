// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

// MARK: - Raw Engine Output

/// Raw outputs from a detection engine — engine-agnostic Float arrays.
///
/// Uses `[Float]` rather than backend-specific tensor types so that
/// `DetectionPostprocessor` remains independent of any particular inference framework.
///
/// All tensors are flattened in C (row-major) order. This type is an intermediate
/// representation produced by `DetectionEngine` implementations and consumed by
/// `DetectionPostprocessor.decode(output:inputSize:parameters:)`.
struct DetectionOutput: Sendable {
    /// Flat class logits, shape `[batch, queryCount, classCount]`.
    /// For YOLOS/DETR: classCount includes the "no-object" class as the last index.
    let logits: [Float]

    /// Shape of `logits`: `[batch, queryCount, classCount]`.
    let logitsShape: [Int]

    /// Flat bounding-box coordinates, shape `[batch, queryCount, 4]`.
    /// Format: `[cx, cy, w, h]` normalized to [0, 1].
    let predictedBoxes: [Float]
}

/// A single detected object: bounding box, class label, and confidence score.
public struct DetectedObject: Sendable {
    /// Bounding box in pixel coordinates (top-left origin).
    public let boundingBox: CGRect

    /// Class label index from the model's vocabulary.
    public let labelIndex: Int

    /// Human-readable class label (e.g. "dog", "car").
    public let label: String

    /// Confidence score in [0, 1].
    public let confidence: Float

    public init(boundingBox: CGRect, labelIndex: Int, label: String, confidence: Float) {
        self.boundingBox = boundingBox
        self.labelIndex = labelIndex
        self.label = label
        self.confidence = confidence
    }
}

// MARK: - Parameters

/// Runtime parameters that control preprocessing and output decoding.
public struct DetectionParameters: Sendable {
    /// Confidence threshold — detections below this are discarded.
    public var threshold: Float

    /// Maximum number of detections returned, sorted by score (highest first).
    public var maxDetections: Int

    /// Per-channel normalization means applied after scaling pixels to [0, 1].
    /// Default `(0.485, 0.456, 0.406)` matches ImageNet normalization.
    public var normalizationMeans: (CGFloat, CGFloat, CGFloat)

    /// Per-channel normalization standard deviations.
    /// Default `(0.229, 0.224, 0.225)` matches ImageNet normalization.
    public var normalizationStds: (CGFloat, CGFloat, CGFloat)

    /// Class label vocabulary. Maps class index → human-readable name.
    /// When empty, labels default to "class_N".
    public var classLabels: [Int: String]

    /// Model input height. Only consulted when the model declares a dynamic
    /// spatial dimension; ignored for static-shape models. Defaults to 800
    /// (matches the YOLOS export's reference input and the training-time
    /// canvas geometry).
    public var inputHeight: Int

    /// Model input width. Only consulted when the model declares a dynamic
    /// spatial dimension; ignored for static-shape models. Defaults to 800.
    public var inputWidth: Int

    public init(
        threshold: Float = 0.3,
        maxDetections: Int = 100,
        normalizationMeans: (CGFloat, CGFloat, CGFloat) = (0.485, 0.456, 0.406),
        normalizationStds: (CGFloat, CGFloat, CGFloat) = (0.229, 0.224, 0.225),
        classLabels: [Int: String] = ObjectDetectionLabels.coco,
        inputHeight: Int = 800,
        inputWidth: Int = 800
    ) {
        self.threshold = threshold
        self.maxDetections = maxDetections
        self.normalizationMeans = normalizationMeans
        self.normalizationStds = normalizationStds
        self.classLabels = classLabels
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
    }

    public static let `default` = DetectionParameters()
}

// MARK: - Labels

/// Built-in class label vocabularies for common model families.
public enum ObjectDetectionLabels {
    /// COCO 91-class label mapping for DETR/YOLOS models trained on COCO.
    public static let coco: [Int: String] = [
        0: "N/A", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
        5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
        10: "traffic light", 11: "fire hydrant", 12: "N/A", 13: "stop sign",
        14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
        19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
        24: "zebra", 25: "giraffe", 26: "N/A", 27: "backpack", 28: "umbrella",
        29: "N/A", 30: "N/A", 31: "handbag", 32: "tie", 33: "suitcase",
        34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball",
        38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard",
        42: "surfboard", 43: "tennis racket", 44: "bottle", 45: "N/A",
        46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
        51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
        56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
        61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
        66: "N/A", 67: "dining table", 68: "N/A", 69: "N/A", 70: "toilet",
        71: "N/A", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
        76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
        80: "toaster", 81: "sink", 82: "refrigerator", 83: "N/A", 84: "book",
        85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
        89: "hair drier", 90: "toothbrush",
    ]
}
