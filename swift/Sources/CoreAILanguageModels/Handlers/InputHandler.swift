// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Context for each inference step, used by both dynamic (GPU) and static (ANE) engines.
public struct InputContext: Sendable {
    /// Tokens to process in this step.
    public let tokens: ArraySlice<Int32>
    /// Number of tokens already processed before this step.
    public let processedTokenCount: Int
    /// Batch-aligned start position.
    public let alignedStep: Int
    /// Total batch slot count (may exceed tokens.count for static-shape padding).
    public let batchSize: Int
    /// Sliding window size (nil for models without sliding attention).
    public let slidingWindow: Int?

    /// For dynamic-shape engines (Sequential, Pipelined).
    /// alignedStep = processedTokenCount, batchSize = tokens.count.
    public static func dynamic(
        tokens: ArraySlice<Int32>,
        processedTokenCount: Int
    ) -> InputContext {
        InputContext(
            tokens: tokens,
            processedTokenCount: processedTokenCount,
            alignedStep: processedTokenCount,
            batchSize: tokens.count,
            slidingWindow: nil)
    }

    /// For static-shape engines. Batch is fixed-size and aligned.
    public static func `static`(
        tokens: ArraySlice<Int32>,
        alignedStep: Int,
        batchSize: Int,
        slidingWindow: Int?
    ) -> InputContext {
        InputContext(
            tokens: tokens,
            processedTokenCount: alignedStep,
            alignedStep: alignedStep,
            batchSize: batchSize,
            slidingWindow: slidingWindow)
    }
}

/// Prepares model inputs for each inference step.
///
/// The engine calls `prepare(...)` each step and passes the result to `function.run()`.
/// Standard models use `TokenInputHandler`; models with extra inputs (RoPE,
/// sliding step, PLE) wrap it with `CompositeInputHandler`.
public protocol SyncInputHandler {
    /// Input names this handler produces.
    var inputNames: [String] { get }

    /// Prepare inputs for the current step.
    mutating func prepare(_ context: InputContext) async throws -> [String: NDArray]
}

// MARK: - Load-time Coverage Check

public enum InputCoverage {
    /// Verify that a set of handlers covers all required inputs declared by the model descriptor.
    /// Call at engine init to fail fast on missing handlers rather than producing NaN at runtime.
    ///
    /// - Parameters:
    ///   - handlers: All input handlers the engine will use.
    ///   - descriptor: The model function's descriptor declaring required inputs.
    ///   - ignoring: Input names to exclude from the check (e.g. "embedding_table" passed directly).
    /// - Throws: If any declared input is not produced by any handler.
    public static func verify(
        handlers: [any SyncInputHandler],
        descriptor: InferenceFunctionDescriptor,
        ignoring: Set<String> = []
    ) throws {
        let produced = handlers.reduce(into: Set<String>()) { $0.formUnion($1.inputNames) }
        let declared = Set(descriptor.inputNames).subtracting(ignoring)
        let uncovered = declared.subtracting(produced)
        guard uncovered.isEmpty else {
            throw InferenceRuntimeError.invalidState(
                "No input handler produces required input(s): \(uncovered.sorted()). "
                    + "Produced: \(produced.sorted())")
        }
    }
}
