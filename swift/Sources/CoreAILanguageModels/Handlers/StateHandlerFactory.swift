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
}

/// Creates state handlers from a model's function descriptor.
///
/// Classification priority:
/// 1. Explicit metadata (`"states"` field in metadata.json) — preferred
/// 2. Shape-based heuristic — dynamic dim → kvCache, static + "cache" in name → slidingCache, else → fixed
/// 3. Legacy (2 states) — both kvCache, no warning
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

        // Heuristic fallback
        if names.count == 2 {
            // Legacy: 2 states = both KV cache (no warning)
            return names.map { ($0, .kvCache) }
        }

        // N states: classify by shape + name
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
        let hasDynamicDim = desc.shape.contains(where: { $0 < 0 })
        if hasDynamicDim {
            return .kvCache
        }
        let lower = name.lowercased()
        if lower.contains("cache") || lower.contains("kv") {
            return .slidingCache
        }
        return .fixed
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
            hasNonTruncatableStates: hasNonTruncatable
        )
    }
}
