// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation

// MARK: - Model Structure Detection

/// Well-known graph function names used for structure detection.
public enum GraphNames {
    public static let main = "main"
    public static let loadEmbeddings = "load_embeddings"
    public static let extendPrefix = "extend"
    // Multi-function segmenter (lite SAM3 export for iOS).
    public static let imageEncode = "image_encode"
    public static let textEncode = "text_encode"
    public static let detect = "detect"
}

/// Represents the detected structure of a Core AI model.
///
/// Model structure determines which inference engine variant should be used:
/// - `chunkedStatic`: Uses static-shape `StaticShapeEngine`
/// - `dynamic`: Uses `CoreAISequentialEngine` or `CoreAIPipelinedEngine`
/// - `multiFunctionSegmenter`: Uses `CoreAISegmentationEngine` against an asset
///   with `image_encode` / `text_encode` / `detect` graphs (e.g. optimized SAM3).
public enum ModelStructure: Equatable, Sendable, CustomStringConvertible {
    /// Chunked static model with fixed batch size for static-shape execution.
    /// Identified by presence of `extend_*` and `load_embeddings` functions.
    case chunkedStatic(batchSize: Int)

    /// Dynamic model with single `main` function for GPU/CPU inference.
    case dynamic

    /// Three-function segmenter targeting iOS.
    /// Identified by presence of `image_encode`, `text_encode`, and `detect` graphs.
    case multiFunctionSegmenter

    public var description: String {
        switch self {
        case .chunkedStatic(let batchSize):
            return "chunkedStatic(batchSize: \(batchSize))"
        case .dynamic:
            return "dynamic"
        case .multiFunctionSegmenter:
            return "multiFunctionSegmenter"
        }
    }

    /// The preferred device for this model structure.
    ///
    /// - `chunkedStatic` → NeuralEngine
    /// - `dynamic` → GPU
    /// - `multiFunctionSegmenter` → NeuralEngine
    public var preferredDevice: String {
        switch self {
        case .chunkedStatic, .multiFunctionSegmenter:
            return "NeuralEngine"
        case .dynamic:
            return "GPU"
        }
    }

    /// Returns `SpecializationOptions` derived from the model structure.
    ///
    /// - `chunkedStatic` → prefer `.neuralEngine`
    /// - `dynamic` → prefer `.gpu` + `expectFrequentReshapes`
    /// - `multiFunctionSegmenter` → prefer `.neuralEngine`
    public var specializationOptions: SpecializationOptions {
        switch self {
        case .chunkedStatic, .multiFunctionSegmenter:
            return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .dynamic:
            var opts = SpecializationOptions(preferredComputeUnitKind: .gpu)
            opts.expectFrequentReshapes = true
            return opts
        }
    }
}

// MARK: - Prepared Asset Container

/// Container holding a pre-loaded model asset with detected structure.
///
/// This struct is passed to engine initializers to avoid double-loading the model.
/// The `AIModel` is already loaded and JIT-compiled, so engines can
/// directly use it without repeating the expensive loading process.
///
/// ## Usage
/// ```swift
/// let asset = try await PreparedModelAsset.prepare(at: modelURL)
/// // Pass asset.model directly to engine
/// ```
///
/// ## Thread Safety
/// `PreparedModelAsset` is `Sendable` and can be safely shared across actor boundaries.
/// The underlying `AIModel` is thread-safe for read access.
public struct PreparedModel: Sendable {
    /// The pre-loaded and JIT-compiled model.
    public let model: AIModel

    /// Detected model structure (chunked/static vs dynamic).
    public let structure: ModelStructure

    // MARK: - Core AI Model URL Resolution

    /// If `url` is already `.aimodel`, returns it unchanged. Otherwise looks for
    /// a sibling `.aimodel` directory with the same base name.
    public static func resolveCoreAIModelURL(from url: URL) -> URL {
        let ext = url.pathExtension

        // Already a Core AI format
        if ext == "aimodel" {
            return url
        }

        // Check for sibling Core AI model directory
        let parentDir = url.deletingLastPathComponent()
        let baseName = url.deletingPathExtension().lastPathComponent

        let candidate = parentDir.appendingPathComponent("\(baseName).aimodel")
        if FileManager.default.fileExists(atPath: candidate.path) {
            CLILogger.log(
                "  - Resolved CoreAI model path: \(url.lastPathComponent) → \(candidate.lastPathComponent)")
            return candidate
        }

        // Fall through to original URL (AIModel may still handle it)
        return url
    }

    // MARK: - Asset Preparation

    /// Prepares a Core AI model asset by loading via `AIModel` and detecting its structure.
    ///
    /// Specialization options are derived automatically from the detected model structure:
    /// dynamic models prefer GPU with frequent reshapes; chunked-static models prefer Neural Engine.
    ///
    /// - Parameters:
    ///   - url: URL to the model asset (`.aimodel` bundle)
    /// - Returns: Prepared asset with compiled library and detected structure
    /// - Throws: Error from `AIModel` if loading or specialization fails
    public static func prepare(
        at url: URL
    ) async throws -> PreparedModel {
        CLILogger.log("PreparedModelAsset: Preparing \(url.lastPathComponent)")

        // Probe structure before specializing so we can pick the right compute-unit preference.
        let probedStructure = probeStructure(at: url)
        CLILogger.log("  - Probed structure: \(probedStructure.description)")

        let options = probedStructure.specializationOptions
        let model = try await AIModel(contentsOf: url, options: options)
        CLILogger.log("  - Loaded \(model.functionNames.count) graphs")

        // Re-detect from compiled library — source of truth, should match the probe.
        let structure = detectStructure(from: model.functionNames)

        return PreparedModel(
            model: model,
            structure: structure
        )
    }

    // MARK: - Structure Probing (pre-specialization)

    /// Probes model structure via `AIModelAsset.summary()` without triggering specialization.
    private static func probeStructure(at url: URL) -> ModelStructure {
        do {
            let asset = try AIModelAsset(contentsOf: url)
            if let summary = try asset.summary(includingStatistics: false) {
                let names = summary.functions.map(\.name)
                if !names.isEmpty {
                    CLILogger.log("  - Probe (summary): \(names.count) functions")
                    return detectStructure(from: names)
                }
            }
            CLILogger.log("  - Probe (summary) returned empty; defaulting to .dynamic")
        } catch {
            CLILogger.log("  - Probe (summary) failed: \(error); defaulting to .dynamic")
        }
        return .dynamic
    }

    /// Detects model structure from graph names.
    ///
    /// Shared implementation used by both fast paths (AIModel) and source path (ModelAsset).
    private static func detectStructure(from graphNames: [String]) -> ModelStructure {
        let graphSet = Set(graphNames)

        // Static-shape model (chunked/static)
        let extendFunctions = graphNames.filter { $0.hasPrefix(GraphNames.extendPrefix) }
        if !extendFunctions.isEmpty && graphSet.contains(GraphNames.loadEmbeddings) {
            let batchSize = extractBatchSize(from: extendFunctions.first!) ?? 1
            return .chunkedStatic(batchSize: batchSize)
        }

        // Multi-function segmenter (e.g. optimized SAM3 — image_encode / text_encode / detect).
        // Targets neuralEngine; checked before the `main` fallback because some asset variants ship
        // a thin `main` graph alongside the trio.
        if graphSet.contains(GraphNames.imageEncode)
            && graphSet.contains(GraphNames.textEncode)
            && graphSet.contains(GraphNames.detect)
        {
            return .multiFunctionSegmenter
        }

        // GPU model (dynamic)
        if graphSet.contains(GraphNames.main) {
            return .dynamic
        }

        // Unknown - default to GPU dynamic
        CLILogger.log("  - Warning: Unknown model structure, defaulting to GPU dynamic")
        return .dynamic
    }

    // MARK: - Private Helpers

    /// Extracts batch size from extend function name (e.g., `extend_512_8` → 8).
    private static func extractBatchSize(from functionName: String) -> Int? {
        let parts = functionName.split(separator: "_")
        guard parts.count >= 3 else { return nil }
        // Index 2 is the batch size: extend_<context>_<batch>
        return Int(parts[2])
    }
}
