// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared

/// A sampling configuration for controlling the randomness and diversity of token generation in language models.
///
/// Sampling parameters allow fine-tuning of the generation process to achieve different behaviors:
/// - **Temperature**: Controls overall randomness
/// - **TopK**: Limits vocabulary to K most likely tokens
/// - **TopP (nucleus)**: Limits vocabulary by cumulative probability threshold
/// - **MinP**: Limits vocabulary by minimum probability relative to the most likely token
///
/// ## Sampling Algorithm Order
/// When multiple parameters are set, they are applied in this order:
/// 1. Temperature scaling (logits / temperature)
/// 2. MinP filtering (relative probability threshold)
/// 3. TopP filtering (cumulative probability cutoff)
/// 4. TopK filtering (hard limit on vocabulary)
/// 5. Softmax and multinomial sampling
///
/// ## Usage Example
/// ```swift
/// // Creative writing with high temperature
/// let creative = SamplingConfiguration(temperature: 0.8)
///
/// // Deterministic output
/// let deterministic = SamplingConfiguration.greedy
///
/// // TopK sampling for focused but varied output
/// let topK = SamplingConfiguration(temperature: 0.8, topK: 50)
///
/// // Nucleus (TopP) sampling
/// let nucleus = SamplingConfiguration(temperature: 0.9, topP: 0.95)
///
/// // MinP sampling (cheaper alternative to TopP)
/// let minP = SamplingConfiguration(temperature: 0.9, minP: 0.05)
///
/// // Combined TopK + TopP (recommended for best quality)
/// let combined = SamplingConfiguration(temperature: 0.8, topK: 50, topP: 0.9)
/// ```
public struct SamplingConfiguration: Sendable, Equatable, Hashable {
    /// Controls the randomness of token generation.
    ///
    /// - **0.0**: Deterministic output (always picks the most likely token)
    /// - **0.1-0.3**: Focused, coherent output with minimal creativity
    /// - **0.4-0.7**: Balanced creativity and coherence
    /// - **0.8-1.0**: High creativity, potentially less coherent
    /// - **>1.0**: Very random, often incoherent output
    ///
    /// Higher values make the output more random by flattening the probability distribution,
    /// while lower values make it more deterministic by sharpening the distribution.
    public let temperature: Double

    /// Top-K sampling: only consider the K most likely tokens.
    ///
    /// - **nil**: Consider all tokens (full vocabulary)
    /// - **1**: Effectively greedy (use temperature=0 instead for efficiency)
    /// - **40-50**: Common default for creative generation
    /// - **100+**: Minimal filtering, allows rare tokens
    ///
    /// TopK provides a hard limit on vocabulary size, preventing
    /// sampling of very unlikely tokens that could be nonsense.
    public let topK: Int?

    /// Top-P (nucleus) sampling: only consider tokens comprising the top P probability mass.
    ///
    /// - **nil**: No nucleus filtering
    /// - **0.9**: Common default (consider tokens in top 90% probability)
    /// - **0.95**: More permissive (top 95%)
    /// - **1.0**: Consider all tokens (equivalent to nil, no filtering)
    ///
    /// TopP provides adaptive vocabulary size based on the distribution.
    /// When the model is confident (one token dominates), vocabulary shrinks.
    /// When uncertain (flat distribution), vocabulary expands.
    public let topP: Double?

    /// Min-P sampling: only consider tokens whose probability is at least minP times
    /// the most likely token's probability.
    ///
    /// - **nil**: No min-P filtering
    /// - **0.05**: Common default (keep tokens with >= 5% of top token's probability)
    /// - **0.1**: More aggressive filtering
    /// - **0.01**: Very permissive
    ///
    /// MinP is a simpler, cheaper alternative to TopP that adapts to the distribution shape.
    /// When the model is confident, fewer tokens pass. When uncertain, more tokens pass.
    /// Unlike TopP, it does not require sorting — it operates as a simple threshold in logit space.
    public let minP: Double?

    /// A boolean flag that requests the sampling operation be combined
    /// with logit inference.
    ///
    /// This is true by default, as the performance will be impacted by
    /// the additional synchronization point.
    ///
    /// Disabling the combined operation will allow more fine-grained
    /// instrumentation of discrete steps.
    public let combined: Bool

    /// Creates a new sampling configuration with validation.
    ///
    /// - Parameters:
    ///   - temperature: The randomness factor for token generation. Must be >= 0.0.
    ///   - topK: Optional top-K limit. Must be > 0 if set.
    ///   - topP: Optional top-P threshold. Must be in (0, 1] if set.
    ///   - minP: Optional min-P threshold. Must be in (0, 1] if set.
    ///   - combined: Whether to combine sampling with logit inference. Defaults to true.
    ///
    /// - Note: Call `validate()` to check for potentially suboptimal configurations.
    public init(temperature: Double, topK: Int? = nil, topP: Double? = nil, minP: Double? = nil, combined: Bool = true)
    {
        precondition(temperature >= 0, "Temperature must be non-negative.")
        precondition(topK == nil || topK! > 0, "TopK must be positive if set.")
        precondition(topP == nil || (topP! > 0 && topP! <= 1), "TopP must be in (0, 1] if set.")
        precondition(minP == nil || (minP! > 0 && minP! <= 1), "MinP must be in (0, 1] if set.")

        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.combined = combined
    }

    /// A predefined configuration for deterministic, greedy token generation.
    ///
    /// This configuration always selects the most likely token at each step,
    /// resulting in deterministic and reproducible output.
    ///
    /// Equivalent to `SamplingConfiguration(temperature: 0)`.
    public static let greedy = SamplingConfiguration(temperature: 0)

    /// Creates a temperature-only sampling configuration.
    ///
    /// - Parameter t: Temperature value (0 = greedy, higher = more random)
    /// - Returns: A sampling configuration with only temperature set
    public static func temperature(_ t: Double) -> SamplingConfiguration {
        SamplingConfiguration(temperature: t)
    }

    /// Whether this configuration uses greedy (argmax) sampling.
    ///
    /// True when temperature is 0, regardless of topK/topP settings.
    public var isGreedy: Bool {
        temperature == 0
    }

    /// Whether this configuration requires composite sampling (topK, topP, and/or minP).
    ///
    /// True when temperature > 0 and any of topK, topP, or minP is set.
    public var isComposite: Bool {
        temperature > 0 && (topK != nil || topP != nil || minP != nil)
    }

    /// Validates the configuration and returns warnings for potentially suboptimal settings.
    ///
    /// This method checks for:
    /// - topK=1 with temperature>0 (should use greedy instead)
    /// - topP=1.0 (effectively disabled, same as nil)
    /// - minP=1.0 (effectively greedy, should use temperature=0)
    /// - topK/topP/minP set with temperature=0 (ignored for greedy)
    ///
    /// - Returns: Array of warning messages, empty if configuration is optimal.
    private func validate() -> [String] {
        var warnings: [String] = []

        // Check for topK=1 with temperature > 0 (wasteful, should use greedy)
        if let k = topK, k == 1 && temperature > 0 {
            warnings.append(
                "topK=1 with temperature>0 is equivalent to greedy sampling but less efficient. "
                    + "Use temperature=0 for deterministic output."
            )
        }

        // Check for topP=1.0 (effectively disabled)
        if let p = topP, p == 1.0 {
            warnings.append(
                "topP=1.0 has no effect (considers all tokens). "
                    + "Remove topP or use a value < 1.0 for nucleus sampling."
            )
        }

        // Check for minP=1.0 (effectively greedy)
        if let m = minP, m == 1.0 {
            warnings.append(
                "minP=1.0 keeps only the single most-probable token. "
                    + "Use temperature=0 for deterministic output, or a smaller minP value."
            )
        }

        // Check for minP + topP together (unusual, may indicate confusion)
        if minP != nil && topP != nil {
            warnings.append(
                "Both minP and topP are set. They serve similar purposes (adaptive filtering). "
                    + "Both will apply (minP first, then topP), but typically only one is needed."
            )
        }

        // Check for topK/topP/minP with temperature=0 (ignored)
        if temperature == 0 && (topK != nil || topP != nil || minP != nil) {
            warnings.append(
                "topK/topP/minP are ignored when temperature=0 (greedy sampling). "
                    + "Set temperature>0 to enable filtering, or remove topK/topP/minP."
            )
        }

        return warnings
    }

    /// Validates the configuration and logs warnings to the console.
    ///
    /// Call this at configuration time to catch potentially suboptimal settings.
    public func validateAndWarn() {
        let warnings = validate()
        for warning in warnings {
            CLILogger.log("⚠️ SamplingConfiguration: \(warning)", component: "Sampling")
        }
    }

    /// Returns a normalized configuration with redundant settings removed.
    ///
    /// - topP=1.0 is replaced with nil (no effect)
    /// - topK/topP/minP are removed if temperature=0 (greedy ignores them)
    ///
    /// - Returns: A new configuration with redundant settings removed.
    public func normalized() -> SamplingConfiguration {
        let effectiveTopK: Int?
        let effectiveTopP: Double?
        let effectiveMinP: Double?

        if temperature == 0 {
            // Greedy ignores topK/topP/minP
            effectiveTopK = nil
            effectiveTopP = nil
            effectiveMinP = nil
        } else {
            effectiveTopK = topK
            // topP=1.0 is equivalent to nil
            effectiveTopP = (topP == 1.0) ? nil : topP
            effectiveMinP = minP
        }

        return SamplingConfiguration(
            temperature: temperature,
            topK: effectiveTopK,
            topP: effectiveTopP,
            minP: effectiveMinP,
            combined: combined
        )
    }
}

extension SamplingConfiguration {
    /// Samples the next token using CPU fallback samplers.
    ///
    /// Routes to the appropriate sampler based on configuration:
    /// - Temperature == 0: Greedy (argmax)
    /// - Otherwise: Composite sampler with Float32 internal math
    ///
    /// - Parameter logits: Mutable array of Float16 logits. May be modified during sampling.
    /// - Returns: The sampled token ID.
    public func fallbackSampler(from logits: inout [LogitsScalarType]) -> Int32 {
        return CompositeSampler.sample(from: &logits, config: self)
    }
}
