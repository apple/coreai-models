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
    /// Context bucket size for the current graph (static engines only).
    /// Determines causal mask height. 0 for dynamic engines.
    public let contextBucket: Int

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
            slidingWindow: nil,
            contextBucket: 0)
    }

    /// For static-shape engines. Batch is fixed-size and aligned.
    public static func `static`(
        tokens: ArraySlice<Int32>,
        alignedStep: Int,
        batchSize: Int,
        slidingWindow: Int?,
        contextBucket: Int
    ) -> InputContext {
        InputContext(
            tokens: tokens,
            processedTokenCount: alignedStep,
            alignedStep: alignedStep,
            batchSize: batchSize,
            slidingWindow: slidingWindow,
            contextBucket: contextBucket)
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

// MARK: - New InputHandler Protocol (inout fill pattern)

/// Sentinel value for masked (non-attending) positions in the causal attention mask.
/// Large negative that becomes ~0 after softmax.
public let causalMaskSentinel = Float16(-40000)

/// Pre-allocated input buffer set. Owns NDArrays, reused across steps.
/// The engine creates this once at init and passes it to `StaticInputHandler.fill()` each step.
public struct InputBuffers {
    private var buffers: [String: NDArray] = [:]
    private var pool: [String: [[Int]: NDArray]] = [:]

    public init() {}

    /// Pre-allocate a buffer for a specific descriptor shape. Called at init to
    /// populate the pool with every shape the engine will use.
    /// Does NOT set the active buffer — the first ensureCapacity call will move
    /// the desired shape out of the pool with exclusive ownership.
    public mutating func preAllocate(name: String, descriptor: NDArrayDescriptor) {
        pool[name, default: [:]][descriptor.shape] = NDArray(descriptor: descriptor)
    }

    /// Swap the active buffer to a pre-allocated one matching the descriptor's shape.
    /// Moves the NDArray out of the pool (sole ownership) so fillNDArray avoids COW.
    /// The current buffer is parked back into the pool for later reuse.
    public mutating func ensureCapacity(name: String, descriptor: NDArrayDescriptor) {
        if buffers[name]?.shape == descriptor.shape { return }
        // Park current buffer back into pool
        if let current = buffers.removeValue(forKey: name) {
            pool[name, default: [:]][current.shape] = current
        }
        // Move desired buffer out of pool (removeValue drops the pool's reference)
        if let moved = pool[name]?.removeValue(forKey: descriptor.shape) {
            buffers[name] = moved
        } else {
            buffers[name] = NDArray(descriptor: descriptor)
        }
    }

    /// Access a buffer by name.
    ///
    /// - Warning: The `get` accessor returns a *copy* of the NDArray reference.
    ///   Holding the result (`let a = buffers[name]`) raises the refcount and
    ///   forces COW on the next in-place fill. For zero-copy mutation, use
    ///   `withMutableBuffer(_:body:)` or mutate through the subscript directly
    ///   (`fillNDArray(&buffers[name]!, ...)`), which routes through `_modify`.
    ///
    /// The `_modify` accessor yields the NDArray in-place from the dictionary
    /// without copying the struct out and back. This avoids COW reference counting
    /// overhead that would otherwise add ~30µs per access. Do NOT replace with a
    /// simple get/set — that reintroduces the COW cost (measured: 28µs → 1.1µs).
    public subscript(name: String) -> NDArray? {
        get { buffers[name] }
        set { buffers[name] = newValue }
        _modify { yield &buffers[name] }
    }

    /// Access a buffer's MutableRawView for zero-copy filling.
    /// Same pattern as StateHandler+NDArray.swift bind(into:).
    public subscript(view name: String) -> NDArray.MutableRawView? {
        mutating get {
            _overrideLifetime(buffers[name]?.mutableRawView(), borrowing: ())
        }
    }

    /// Get a mutable reference to a buffer for fillNDArray.
    /// Note: prefer `buffers[name]` with _modify for zero-overhead access.
    public mutating func withBuffer(
        _ name: String,
        body: (inout NDArray) -> Void
    ) {
        guard var array = buffers[name] else { return }
        body(&array)
        buffers[name] = array
    }

    /// Mutate a buffer in-place through the dictionary's _modify accessor.
    /// Avoids the extract/put-back COW overhead of withBuffer.
    public mutating func withMutableBuffer(
        _ name: String,
        body: (inout NDArray) throws -> Void
    ) throws {
        guard buffers[name] != nil else {
            throw InferenceRuntimeError.invalidState("No buffer registered for '\(name)'")
        }
        try body(&buffers[name]!)
    }

    /// Borrow the current buffer dictionary for a single inference call.
    ///
    /// Returns a shallow copy of the dictionary. Because NDArray is a
    /// reference-type wrapper, the returned values share backing storage
    /// with this InputBuffers instance. The caller must NOT retain the
    /// result past the current inference step — doing so lets `fill()`
    /// on the next step mutate arrays the caller still references.
    public func borrowedInputs() -> [String: NDArray] {
        buffers
    }
}

/// Fills pre-allocated input buffers in-place. No per-step allocation.
///
/// The engine owns `InputBuffers` and passes it to `fill()` each step.
/// Implementations write into the buffers via `fillNDArray` — no dict creation,
/// no COW risk from accidental copy-out.
///
/// Handlers are stateless — all per-step state lives in `InputContext` or `InputBuffers`.
/// This allows handlers to be stored as `let` and shared across engines.
public protocol StaticInputHandler: Sendable {
    /// Input names this filler produces.
    var inputNames: [String] { get }

    /// Register required buffers into the InputBuffers at engine init time.
    func registerBuffers(into buffers: inout InputBuffers)

    /// Fill input buffers for the given context.
    /// Non-mutating: handler holds no per-step state. All mutation goes into `buffers`.
    func fill(_ context: InputContext, into buffers: inout InputBuffers) throws
}
