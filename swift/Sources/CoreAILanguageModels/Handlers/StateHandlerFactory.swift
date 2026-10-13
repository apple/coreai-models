// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared

/// Classification of a model state's lifecycle behavior.
public enum StateKind: String, Codable, Sendable {
    /// KV cache — grows dynamically with context, supports truncation (causal mask).
    case kvCache = "kv_cache"
    /// Sliding window cache — fixed size, supports truncation (causal mask).
    case slidingCache = "sliding_cache"
    /// Fixed state (conv, recurrent) — fixed size, does NOT support truncation.
    case fixed
}

/// Result of state handler creation.
struct SyncStateHandlerSet {
    /// Growing states (KV caches with dynamic sequence dimension).
    var kvCache: any SyncStateHandler
    /// Fixed states (sliding caches + recurrent/conv). Nil for transformer-only models.
    var additionalStates: FixedNDArrayState?
    /// Whether any state is non-truncatable (triggers full-reset-only mode).
    var hasNonTruncatableStates: Bool
    /// True when ALL states are sliding caches (ring buffer model).
    /// The engine uses compact position_ids [offset, ..., offset+queryLen-1] instead of [0, ..., N-1].
    var isAllSlidingCache: Bool
    /// Capacity of the sliding (ring) caches, nil when there are none (hybrid or pure-ring).
    var slidingRingCapacity: Int?
}

/// Creates state handlers from a model's function descriptor.
///
/// Classification priority:
/// 1. Explicit metadata (`"states"` field in metadata.json) — preferred
/// 2. Shape-based heuristic — dynamic dim → kvCache, static + "sliding" in name → slidingCache,
///    static + "cache"/"kv" in name → kvCache (fixed capacity), else → fixed
enum StateHandlerFactory {
    /// Classify states using metadata or heuristic fallback.
    static func classifyStates(
        descriptor: InferenceFunctionDescriptor,
        stateKinds: [String: StateKind]? = nil,
        verbose: Bool = false
    ) -> [(name: String, kind: StateKind)] {
        let names = descriptor.stateNames

        if let kinds = stateKinds {
            // Explicit metadata — validate and use
            return names.map { name in
                let kind = kinds[name] ?? inferKind(name: name, descriptor: descriptor)
                return (name, kind)
            }
        }

        // Heuristic fallback: classify all states by shape + name
        let classified = names.map { name -> (String, StateKind) in
            (name, inferKind(name: name, descriptor: descriptor))
        }

        if verbose {
            CLILogger.log("State classification (heuristic):", component: "StateHandlerFactory")
            for (name, kind) in classified {
                guard case .ndArray(let desc) = descriptor.stateDescriptor(of: name) else { continue }
                let shapeStr = desc.shape.map { $0 < 0 ? "?" : "\($0)" }.joined(separator: "×")
                let growth = desc.shape.contains(where: { $0 < 0 }) ? "GROWING" : "FIXED"
                CLILogger.log(
                    "  \(name): \(growth) \(kind.rawValue) (\(shapeStr))",
                    component: "StateHandlerFactory")
            }
            CLILogger.log(
                "  Add \"states\" to metadata.json for explicit control.",
                component: "StateHandlerFactory")
        }
        if !verbose && names.count > 2 {
            CLILogger.log(
                "StateHandlerFactory: \(names.count) states classified by heuristic. "
                    + "Add \"states\" to metadata.json for explicit control.",
                component: "StateHandlerFactory")
        }

        return classified
    }

    /// Infer state kind from shape and name.
    private static func inferKind(name: String, descriptor: InferenceFunctionDescriptor) -> StateKind {
        guard case .ndArray(let desc) = descriptor.stateDescriptor(of: name) else {
            return .fixed
        }
        return inferKind(name: name, shape: desc.shape)
    }

    /// Infer state kind from a state's name and shape (negative entries are dynamic dims).
    ///
    /// A static-shape cache is only a ring buffer when its name says so: an export whose
    /// context fits the trace-time cache length pins `keyCache`/`valueCache` to the full
    /// context, and those are still regular KV caches.
    static func inferKind(name: String, shape: [Int]) -> StateKind {
        if shape.contains(where: { $0 < 0 }) {
            return .kvCache
        }
        let lower = name.lowercased()
        if lower.contains("sliding") {
            return .slidingCache
        }
        if lower.contains("cache") || lower.contains("kv") {
            return .kvCache
        }
        return .fixed
    }

    /// States given a fixed-size kind (`slidingCache` or `fixed`) despite a dynamic dimension.
    /// Those kinds are allocated at a fixed size, which a dynamic shape cannot have; only a
    /// `kvCache` grows.
    static func dynamicStatesWithFixedSizeKind(
        _ classified: [(name: String, kind: StateKind)], isDynamic: (String) -> Bool
    ) -> [String] {
        classified.filter { $0.kind != .kvCache && isDynamic($0.name) }.map(\.name)
    }

    /// Throws if a state with a dynamic dimension was given a fixed-size kind. The shape
    /// heuristic never does this, so only an explicit `language.states` entry can.
    static func validateKinds(
        _ classified: [(name: String, kind: StateKind)], descriptor: InferenceFunctionDescriptor
    ) throws {
        let invalid = dynamicStatesWithFixedSizeKind(classified) { name in
            guard case .ndArray(let desc) = descriptor.stateDescriptor(of: name) else { return false }
            return desc.shape.contains(where: { $0 < 0 })
        }
        guard !invalid.isEmpty else { return }
        let details = classified.filter { invalid.contains($0.name) }
            .map { "\($0.name)=\($0.kind.rawValue)" }.joined(separator: ", ")
        throw InferenceRuntimeError.invalidArgument(
            "language.states gives a fixed-size kind to a state with a dynamic shape "
                + "(\(details)); only kv_cache states may have dynamic dimensions")
    }

    /// Create sync state handlers from classified states.
    static func createSyncHandlers(
        descriptor: InferenceFunctionDescriptor,
        maxContextLength: Int,
        stateKinds: [String: StateKind]? = nil,
        options: EngineOptions = EngineOptions(),
        verbose: Bool = false
    ) throws -> SyncStateHandlerSet {
        guard !descriptor.stateNames.isEmpty else {
            throw InferenceRuntimeError.invalidOutputType(
                "Expected states but found none")
        }

        let classified = classifyStates(
            descriptor: descriptor, stateKinds: stateKinds, verbose: verbose)
        try validateKinds(classified, descriptor: descriptor)

        // Separate into growing (kvCache) and fixed (slidingCache + fixed)
        var growingPairs: [(name: String, descriptor: NDArrayDescriptor)] = []
        var fixedPairs: [(name: String, descriptor: NDArrayDescriptor)] = []
        var hasNonTruncatable = false

        for (name, kind) in classified {
            guard case .ndArray(let desc) = descriptor.stateDescriptor(of: name) else {
                throw InferenceRuntimeError.invalidOutputType(
                    "Cannot get state descriptor for '\(name)'")
            }

            switch kind {
            case .kvCache:
                growingPairs.append((name, desc))
            case .slidingCache:
                fixedPairs.append((name, desc))
            case .fixed:
                fixedPairs.append((name, desc))
                hasNonTruncatable = true
            }
        }

        // Build growing handler (KV caches)
        let kvCache: any SyncStateHandler
        if !growingPairs.isEmpty {
            if options.kvCacheStrategy == .fixedSize {
                let resolved = growingPairs.map { (name, desc) -> (name: String, descriptor: NDArrayDescriptor) in
                    let resolvedDesc = desc.resolvingDynamicDimensions(
                        desc.shape.map { $0 < 0 ? maxContextLength : $0 })
                    return (name, resolvedDesc)
                }
                kvCache = FixedNDArrayState(states: resolved)
            } else {
                let initial = min(256, maxContextLength)
                kvCache = GrowingNDArrayState(
                    states: growingPairs,
                    initialCapacity: initial,
                    maxCapacity: maxContextLength
                )
            }
        } else {
            // All states are fixed (e.g., all sliding caches at fixed-size mode)
            // Use the fixed pairs as KV cache too
            kvCache = FixedNDArrayState(states: fixedPairs)
            fixedPairs = []
        }

        // Build fixed handler (sliding caches + recurrent/conv)
        var additionalStates: FixedNDArrayState? = nil
        if !fixedPairs.isEmpty {
            additionalStates = FixedNDArrayState(states: fixedPairs)
        }

        return SyncStateHandlerSet(
            kvCache: kvCache,
            additionalStates: additionalStates,
            hasNonTruncatableStates: hasNonTruncatable,
            isAllSlidingCache: classified.allSatisfy { $0.kind == .slidingCache },
            slidingRingCapacity: slidingRingCapacity(classified: classified, descriptor: descriptor)
        )
    }

    // MARK: - Sliding Ring Write-First Guard

    /// Capacity of the sliding (ring) states, nil when there are none.
    static func slidingRingCapacity(
        classified: [(name: String, kind: StateKind)], descriptor: InferenceFunctionDescriptor
    ) -> Int? {
        let shapes = classified.filter { $0.kind == .slidingCache }.compactMap { state -> [Int]? in
            guard case .ndArray(let desc) = descriptor.stateDescriptor(of: state.name) else { return nil }
            return desc.shape
        }
        return ringCapacity(shapes: shapes)
    }

    /// Ring capacity from sliding-state shapes `[layers, batch, heads, seq, head_dim]`.
    static func ringCapacity(shapes: [[Int]]) -> Int? {
        shapes.compactMap { $0.count >= 2 && $0[$0.count - 2] > 0 ? $0[$0.count - 2] : nil }.min()
    }

    /// Whether a write-first `main` step evicts ring history its own earlier queries attend to.
    ///
    /// Write-first stores the step's keys before attention, so the ring then holds positions
    /// `[offset + queryLength - capacity, offset + queryLength)`. Query 0 needs
    /// `[offset - capacity + 1, offset]`, so it loses keys exactly when the step has more than
    /// one token and ends past the ring. Assumes the window equals the ring capacity.
    static func writeFirstStepEvicts(offset: Int, queryLength: Int, ringCapacity: Int) -> Bool {
        queryLength > 1 && offset + queryLength > ringCapacity
    }

    /// Throws when a write-first `main` step would evict in-window ring history.
    static func checkWriteFirstStep(offset: Int, queryLength: Int, ringCapacity: Int?) throws {
        guard let ringCapacity,
            writeFirstStepEvicts(offset: offset, queryLength: queryLength, ringCapacity: ringCapacity)
        else { return }
        throw InferenceRuntimeError.invalidState(
            "Sliding-window model: a \(queryLength)-token step at position \(offset) on the "
                + "write-first 'main' graph would evict ring history (capacity \(ringCapacity)). "
                + "Past the ring, only single-token steps are correct on 'main'.")
    }

    /// Whether rewinding from `processed` tokens to `target` leaves intact the ring slots the
    /// query at `target` reads.
    ///
    /// After `processed` writes the ring holds positions `[processed - capacity, processed)`,
    /// while the query at `target` reads `[target - capacity + 1, target)`. Slots for positions
    /// below `processed - capacity` now hold later tokens, so the rewind is unsafe once the ring
    /// has wrapped and the rewind spans two or more tokens.
    static func ringRewindIsSafe(from processed: Int, to target: Int, ringCapacity: Int?) -> Bool {
        guard let ringCapacity, target > 0 else { return true }
        return processed <= ringCapacity || processed - target <= 1
    }
}
