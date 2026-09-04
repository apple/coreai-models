// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI

/// Shared descriptor analysis, built once at engine init.
public struct InputLayout: Sendable {
    public let inputIdsName: String
    public let positionIdsName: String
    public let logitsName: String

    public let queryPolicy: QueryPolicy
    public let logitsPolicy: LogitsPolicy
    public let positionPolicy: PositionPolicy
    public let prefillPolicy: PrefillPolicy

    public enum QueryPolicy: Sendable, Equatable {
        case dynamic
        case fixed(Int)
    }

    public enum LogitsPolicy: Sendable, Equatable {
        case dynamic
        case fixed(seqLen: Int)
    }

    public enum PositionPolicy: Sendable, Equatable {
        case full
        case compact
    }

    public enum PrefillPolicy: Sendable, Equatable {
        case prefillGraph
        case chunk(threshold: Int, chunkSize: Int)
    }
}

extension InputLayout {
    public static func analyze(
        model: AIModel,
        functionName: String,
        config: ModelConfig,
        useCompactPositionIds: Bool
    ) throws -> InputLayout {
        guard let descriptor = model.functionDescriptor(for: functionName) else {
            throw InferenceRuntimeError.genericError(
                "Cannot find function '\(functionName)' in model")
        }
        guard !descriptor.outputNames.isEmpty else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 1 output, got 0")
        }

        let inputIdsName = try resolveRequired(
            from: descriptor.inputNames,
            candidates: knownInputIdNames,
            label: "input_ids")
        let logitsName = descriptor.outputNames[0]

        let inputIdsSeqLen: Int?
        if case .ndArray(let desc) = descriptor.inputDescriptor(of: inputIdsName),
            desc.shape.count >= 2, desc.shape[1] > 0
        {
            inputIdsSeqLen = desc.shape[1]
        } else {
            inputIdsSeqLen = nil
        }

        let logitsSeqLen: Int?
        if case .ndArray(let desc) = descriptor.outputDescriptor(of: logitsName),
            desc.shape.count >= 2, desc.shape[1] > 0
        {
            logitsSeqLen = desc.shape[1]
        } else {
            logitsSeqLen = nil
        }

        let hasPrefillGraph = model.functionDescriptor(for: prefillGraphFunctionName) != nil

        return try analyze(
            inputNames: descriptor.inputNames,
            outputNames: descriptor.outputNames,
            inputIdsSeqLen: inputIdsSeqLen,
            logitsSeqLen: logitsSeqLen,
            chunkThreshold: config.chunkThreshold,
            chunkSize: config.prefillChunkSize,
            hasPrefillGraph: hasPrefillGraph,
            useCompactPositionIds: useCompactPositionIds)
    }

    static func analyze(
        inputNames: [String],
        outputNames: [String],
        inputIdsSeqLen: Int?,
        logitsSeqLen: Int?,
        chunkThreshold: Int,
        chunkSize: Int,
        hasPrefillGraph: Bool,
        useCompactPositionIds: Bool
    ) throws -> InputLayout {
        guard !outputNames.isEmpty else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected at least 1 output, got 0")
        }

        let inputIdsName = try resolveRequired(
            from: inputNames, candidates: knownInputIdNames, label: "input_ids")
        let positionIdsName = try resolveRequired(
            from: inputNames, candidates: knownPositionIdNames, label: "position_ids")
        let logitsName = outputNames[0]

        let queryPolicy: QueryPolicy =
            if let s = inputIdsSeqLen, s > 0 { .fixed(s) } else { .dynamic }

        let logitsPolicy: LogitsPolicy =
            if let s = logitsSeqLen, s > 0 { .fixed(seqLen: s) } else { .dynamic }

        let positionPolicy: PositionPolicy = useCompactPositionIds ? .compact : .full

        let prefillPolicy: PrefillPolicy
        if hasPrefillGraph {
            prefillPolicy = .prefillGraph
        } else if case .fixed(let q) = queryPolicy {
            prefillPolicy = .chunk(threshold: q, chunkSize: q)
        } else {
            prefillPolicy = .chunk(threshold: chunkThreshold, chunkSize: chunkSize)
        }

        return InputLayout(
            inputIdsName: inputIdsName,
            positionIdsName: positionIdsName,
            logitsName: logitsName,
            queryPolicy: queryPolicy,
            logitsPolicy: logitsPolicy,
            positionPolicy: positionPolicy,
            prefillPolicy: prefillPolicy)
    }

    static let knownInputIdNames = ["input_ids", "in_new_token_ids"]
    static let knownPositionIdNames = ["position_ids", "pos_ids"]

    static func resolveRequired(
        from names: [String], candidates: [String], label: String
    ) throws -> String {
        for c in candidates where names.contains(c) { return c }
        throw InferenceRuntimeError.invalidState(
            "No \(label) input found. Expected one of \(candidates), got \(names)")
    }
}
