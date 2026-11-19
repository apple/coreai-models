// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Synchronization

/// Counts which CPU sampler implementation sampled each token, process-wide.
///
/// A backend can hand individual rows to another implementation (``BNNSGraphSampler`` gives rows it
/// can't decide to ``ReferenceSampler``), so the selected backend doesn't say which implementation
/// produced a given token; these counters do. They cover tokens sampled through
/// ``SamplingConfiguration/sampleToken(from:step:)`` (and so `cpuSampler`, constrained decoding and
/// batched sampling). Each count is one relaxed atomic increment.
public enum SamplerUsage {
    /// Token counts per implementation.
    public struct Counts: Sendable, Equatable {
        /// Tokens sampled by ``ReferenceSampler``, including ``bnnsFallbacks``.
        public var reference = 0
        /// Tokens sampled by ``BNNSGraphSampler``'s graph.
        public var bnns = 0
        /// Tokens a ``BNNSGraphSampler`` request gave to ``ReferenceSampler``.
        public var bnnsFallbacks = 0

        public init(reference: Int = 0, bnns: Int = 0, bnnsFallbacks: Int = 0) {
            self.reference = reference
            self.bnns = bnns
            self.bnnsFallbacks = bnnsFallbacks
        }

        /// One line for a run summary, such as `Sampler: bnns 98.1% (9812), reference 1.9% (188; 188 BNNS fallbacks)`.
        public var summary: String {
            let total = reference + bnns
            guard total > 0 else { return "Sampler: no CPU-sampled tokens" }
            func share(_ n: Int) -> String { String(format: "%.1f%%", 100 * Double(n) / Double(total)) }
            var parts: [String] = []
            if bnns > 0 { parts.append("bnns \(share(bnns)) (\(bnns))") }
            if reference > 0 {
                let fallbacks = bnnsFallbacks > 0 ? "; \(bnnsFallbacks) BNNS fallbacks" : ""
                parts.append("reference \(share(reference)) (\(reference)\(fallbacks))")
            }
            return "Sampler: " + parts.joined(separator: ", ")
        }
    }

    private static let reference = Atomic<Int>(0)
    private static let bnns = Atomic<Int>(0)
    private static let bnnsFallbacks = Atomic<Int>(0)

    /// The counts since the process started or since the last ``reset()``.
    public static func snapshot() -> Counts {
        Counts(
            reference: reference.load(ordering: .relaxed), bnns: bnns.load(ordering: .relaxed),
            bnnsFallbacks: bnnsFallbacks.load(ordering: .relaxed))
    }

    /// Sets every count to zero, for example before a generation whose usage a tool reports.
    public static func reset() {
        reference.store(0, ordering: .relaxed)
        bnns.store(0, ordering: .relaxed)
        bnnsFallbacks.store(0, ordering: .relaxed)
    }

    static func recordReference() { reference.wrappingAdd(1, ordering: .relaxed) }
    static func recordBNNS() { bnns.wrappingAdd(1, ordering: .relaxed) }
    static func recordBNNSFallback() {
        bnnsFallbacks.wrappingAdd(1, ordering: .relaxed)
        reference.wrappingAdd(1, ordering: .relaxed)
    }
}
