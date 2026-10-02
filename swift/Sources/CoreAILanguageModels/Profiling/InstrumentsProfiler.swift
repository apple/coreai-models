// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Synchronization
import os.signpost

// MARK: - Unified Profiling Span System

/// Category groups for organizing signpost metrics in output tables
///
/// Groups are sorted in this order: main → decoder → engine
public enum CategoryGroup: String, Comparable {
    case main = "main"  // Top-level lifecycle: model load, tokenizer load, warmup
    case decoder = "decoder"  // Decoding strategy layer: prompt, extend, decode, tokenization
    case engine = "engine"  // Inference engine layer: logits inference, cache, sampling

    /// Sort order for table output (main first, then decoder, then engine)
    var sortOrder: Int {
        switch self {
        case .main: return 0
        case .decoder: return 1
        case .engine: return 2
        }
    }

    public static func < (lhs: CategoryGroup, rhs: CategoryGroup) -> Bool {
        lhs.sortOrder < rhs.sortOrder
    }
}

/// Standard signpost categories for consistent Instruments visualization
public enum SignpostCategory: String, Sendable {
    case prompt = "Prompt"
    case extend = "Extend"  // Strategy layer: inter-token timing (N-1 spans)
    case decode = "Decode"  // Tokenizer: token ID → text string (N spans)
    case logitsInference = "LogitsInference"  // Engine layer: model forward pass (N spans)
    case warmup = "Warmup"
    case sample = "Sample"
    case sampleEncoding = "SampleEncoding"  // GPU sampler command buffer encoding
    case modelLoad = "ModelLoad"
    case tokenizerLoad = "TokenizerLoad"
    case tokenization = "Tokenization"
    case prepareStep = "PrepareStep"  // Engine layer: tensor reshape, rebind, input writes
    case cacheManagement = "CacheManagement"  // Engine layer: KV cache growth/reallocation
    case reset = "Reset"
    case cleanup = "Cleanup"
    // Engine layer sub-spans of PrepareStep / the logits readout — fine-grained
    // attribution of per-token runner-side (non-graph) work.
    case gatherEmbeddings = "GatherEmbeddings"  // separate embedding-gather graph dispatch
    case maskBuild = "MaskBuild"  // sliding-window mask fill
    case ropeBuild = "RopeBuild"  // runner-side RoPE cos/sin precompute (Gemma4)
    case pleGather = "PLEGather"  // INT8 per-layer-embedding gather
    case logitsCopy = "LogitsCopy"  // full-vocab logits copy out of the graph output

    /// Which group this category belongs to for table organization
    var group: CategoryGroup {
        switch self {
        case .modelLoad, .tokenizerLoad, .warmup, .reset, .cleanup:
            return .main
        case .prompt, .extend, .decode, .tokenization:
            return .decoder
        case .logitsInference, .prepareStep, .cacheManagement, .sample, .sampleEncoding,
            .gatherEmbeddings, .maskBuild, .ropeBuild, .pleGather, .logitsCopy:
            return .engine
        }
    }

    var staticString: StaticString {
        switch self {
        case .prompt: return "Prompt"
        case .extend: return "Extend"
        case .decode: return "Decode"
        case .logitsInference: return "LogitsInference"
        case .warmup: return "Warmup"
        case .sample: return "Sample"
        case .sampleEncoding: return "SampleEncoding"
        case .modelLoad: return "ModelLoad"
        case .tokenizerLoad: return "TokenizerLoad"
        case .tokenization: return "Tokenization"
        case .prepareStep: return "PrepareStep"
        case .cacheManagement: return "CacheManagement"
        case .reset: return "Reset"
        case .cleanup: return "Cleanup"
        case .gatherEmbeddings: return "GatherEmbeddings"
        case .maskBuild: return "MaskBuild"
        case .ropeBuild: return "RopeBuild"
        case .pleGather: return "PLEGather"
        case .logitsCopy: return "LogitsCopy"
        }
    }
}

/// Move-only profiling span that unifies signposts and wall clock timing
///
/// Example usage:
/// ```swift
/// let span = Instrumentation.beginPrompt(tokens: 10)
/// // ... do work ...
/// span.end()  // Consuming - cannot use span after this
/// ```
public struct ProfileSpan: ~Copyable {
    private let category: SignpostCategory
    private let signpostID: OSSignpostID
    private let log: OSLog
    private let startTime: UInt64
    private let metadata: [String: String]
    private var wasEnded: Bool = false

    /// The duration in nanoseconds.
    var duration: UInt64 {
        let endTime = mach_absolute_time()
        let duration = endTime - startTime

        // Convert to nanoseconds for stats
        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        return duration * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)
    }

    internal init(category: SignpostCategory, log: OSLog, metadata: [String: String]) {
        self.category = category
        self.log = log
        self.signpostID = OSSignpostID(log: log)
        self.startTime = mach_absolute_time()
        self.metadata = metadata

        // Begin signpost with metadata
        let metadataStr = metadata.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
        os_signpost(
            .begin, log: log, name: category.staticString, signpostID: signpostID,
            "%{public}s", metadataStr)

        if CLILogger.isEnabled(at: 3) {
            CLILogger.log("BEGIN \(metadataStr)", component: category.rawValue)
        }
    }

    deinit {
        // deinit is called after consuming operations complete
        // Only warn if end() was never explicitly called
        if !wasEnded {
            print("⚠️ PROFILING ERROR: ProfileSpan[\(category.rawValue)] was not explicitly ended!")
            print("   This indicates a programming error. Always call span.end()")
            print("   Metadata: \(metadata)")

            // End signpost to avoid leaving it open, but DO NOT record stats
            // (stats would be inaccurate since we don't know when work actually ended)
            endSignpost()
        }
    }

    /// End the profiling span
    public consuming func end() {
        // Capture duration up-front
        let duration = self.duration

        wasEnded = true
        endSignpost()
        record(duration: duration)
    }

    func endSignpost() {
        os_signpost(.end, log: log, name: category.staticString, signpostID: signpostID)
    }

    consuming func record(duration: UInt64) {
        let category = self.category
        let metadata = self.metadata

        Task {
            await StatsStorage.shared.recordSpan(
                category: category,
                durationNanoseconds: duration,
                metadata: metadata
            )
        }

        if CLILogger.isEnabled(at: 3) {
            CLILogger.log("END \(metadata) - \(duration / 1_000_000)ms", component: category.rawValue)
        }
    }

    /// End the profiling span with an explicit storage.
    ///
    /// Used for testing and for when the span can't be consumed.
    @MainActor
    public mutating func end(storingInto storage: StatsStorage) {
        // Capture duration up-front
        let duration = self.duration

        endSignpost()
        recordSync(duration: duration, storage)
        wasEnded = true
    }

    @MainActor
    func recordSync(duration: UInt64, _ storage: StatsStorage) {
        storage.recordSpan(
            category: category,
            durationNanoseconds: duration,
            metadata: metadata
        )

        if CLILogger.isEnabled(at: 3) {
            CLILogger.log("END \(metadata) - \(duration / 1_000_000)ms", component: category.rawValue)
        }
    }
}

/// Thread-safe storage for profiling statistics
@MainActor
public final class StatsStorage {
    public static let shared = StatsStorage()

    private var stats = [SignpostCategory: CategoryStats]()

    private init() {}

    /// Internal initializer for testing (creates isolated instance not connected to ProfileSpan)
    /// Use this in tests to avoid shared state conflicts between parallel test suites
    internal init(forTesting: Void) {}

    internal func recordSpan(
        category: SignpostCategory,
        durationNanoseconds: UInt64,
        metadata: [String: String]
    ) {
        stats[category, default: CategoryStats()].record(
            durationNanoseconds: durationNanoseconds
        )
    }

    /// Aggregate statistics for a category
    public struct AggregateStats: Sendable {
        public let count: Int
        public let totalSeconds: TimeInterval
        public let minSeconds: TimeInterval
        public let maxSeconds: TimeInterval
        public let avgSeconds: TimeInterval
    }

    public func stats(for category: SignpostCategory) -> AggregateStats? {
        guard let catStats = stats[category], catStats.count > 0 else { return nil }

        let nanosecondsToSeconds = { (nanoseconds: UInt64) -> TimeInterval in
            Double(nanoseconds) / 1_000_000_000.0
        }

        return AggregateStats(
            count: catStats.count,
            totalSeconds: nanosecondsToSeconds(catStats.totalNanoseconds),
            minSeconds: nanosecondsToSeconds(catStats.minNanoseconds),
            maxSeconds: nanosecondsToSeconds(catStats.maxNanoseconds),
            avgSeconds: nanosecondsToSeconds(catStats.totalNanoseconds) / Double(catStats.count)
        )
    }

    /// Total duration for a specific category
    public func totalDuration(for category: SignpostCategory) -> TimeInterval {
        return stats(for: category)?.totalSeconds ?? 0.0
    }

    /// Count of spans for a specific category
    public func count(for category: SignpostCategory) -> Int {
        return stats(for: category)?.count ?? 0
    }

    /// Reset all stored statistics
    public func reset() {
        stats.removeAll()
    }

    /// All categories that have recorded data
    public var allCategories: [SignpostCategory] {
        stats.keys.sorted(by: { $0.rawValue < $1.rawValue })
    }
}

extension StatsStorage {
    private struct CategoryStats {
        var count: Int = 0
        var totalNanoseconds: UInt64 = 0
        var minNanoseconds: UInt64 = .max
        var maxNanoseconds: UInt64 = 0

        mutating func record(durationNanoseconds: UInt64) {
            count += 1
            totalNanoseconds += durationNanoseconds
            minNanoseconds = min(minNanoseconds, durationNanoseconds)
            maxNanoseconds = max(maxNanoseconds, durationNanoseconds)
        }
    }
}

// MARK: - Stats Reporter (Presentation Layer)

/// Formats and presents profiling statistics from StatsStorage
///
/// Responsible for presentation logic only:
/// - `printVerboseTable()` - detailed ASCII table with all metrics
///
/// This separation follows Single Responsibility Principle:
/// - StatsStorage: thread-safe data storage and aggregation
/// - StatsReporter: presentation and formatting
///
/// NOTE: Summary printing is handled by `PerformanceMetrics.printSummary()` which
/// combines token counts with timing from StatsStorage. Consider consolidating in future.
@MainActor
public struct StatsReporter {
    private let storage: StatsStorage

    /// Create a reporter for a specific storage instance
    /// - Parameter storage: Storage to read stats from (typically `.shared`)
    public init(storage: StatsStorage) {
        self.storage = storage
    }

    // MARK: - Table Layout Configuration

    /// Column widths for verbose table formatting
    private struct TableLayout {
        let groupWidth: Int
        let metricWidth: Int
        let numWidth: Int
        let countWidth: Int

        static func calculate(rows: [StatsRow]) -> TableLayout {
            TableLayout(
                groupWidth: 7,
                metricWidth: max(16, rows.map { $0.category.rawValue.count }.max() ?? 16),
                numWidth: 9,
                countWidth: 5
            )
        }
    }

    /// A row of stats data for table formatting
    private typealias StatsRow = (group: CategoryGroup, category: SignpostCategory, stats: StatsStorage.AggregateStats)

    // MARK: - Formatting Helpers (Static - Pure Functions)

    /// Format a value in milliseconds with consistent 2 decimal places
    private static func formatMilliseconds(_ seconds: TimeInterval) -> String {
        let milliseconds = seconds * 1000.0
        return String(format: "%9.2f", milliseconds)
    }

    /// Build the table border string based on layout
    private static func buildBorder(layout: TableLayout) -> String {
        "+" + String(repeating: "-", count: layout.groupWidth + 2)
            + "+" + String(repeating: "-", count: layout.metricWidth + 2)
            + "+" + String(repeating: "-", count: layout.numWidth + 2)
            + "+" + String(repeating: "-", count: layout.numWidth + 2)
            + "+" + String(repeating: "-", count: layout.numWidth + 2)
            + "+" + String(repeating: "-", count: layout.countWidth + 2)
            + "+" + String(repeating: "-", count: layout.numWidth + 2) + "+"
    }

    /// Build the header row string based on layout
    private static func buildHeader(layout: TableLayout) -> String {
        let groupPad = String(repeating: " ", count: layout.groupWidth - 5)
        let metricPad = String(repeating: " ", count: layout.metricWidth - 6)
        return "| Group\(groupPad) | Metric\(metricPad) |    Min ms |    Avg ms |    Max ms | Count |  Total ms |"
    }

    /// Format a single data row based on layout
    private static func formatRow(_ row: StatsRow, layout: TableLayout) -> String {
        let groupStr = row.group.rawValue
        let groupRowPad = String(repeating: " ", count: layout.groupWidth - groupStr.count)
        let namePad = String(repeating: " ", count: layout.metricWidth - row.category.rawValue.count)

        // When count == 1, min/max are redundant (same as total), show "-"
        let minMillis: String
        let maxMillis: String
        if row.stats.count == 1 {
            minMillis = String(repeating: " ", count: 8) + "-"
            maxMillis = String(repeating: " ", count: 8) + "-"
        } else {
            minMillis = formatMilliseconds(row.stats.minSeconds)
            maxMillis = formatMilliseconds(row.stats.maxSeconds)
        }

        let avgMillis = formatMilliseconds(row.stats.avgSeconds)
        let count = String(format: "%5d", row.stats.count)
        let totalMillis = formatMilliseconds(row.stats.totalSeconds)

        return
            "| \(groupStr)\(groupRowPad) | \(row.category.rawValue)\(namePad) | \(minMillis) | \(avgMillis) | \(maxMillis) | \(count) | \(totalMillis) |"
    }

    /// Fetch and sort rows from storage
    private func fetchAndSortRows() -> [StatsRow] {
        let categories = storage.allCategories

        var rows: [StatsRow] = []
        for category in categories {
            if let s = storage.stats(for: category) {
                rows.append((category.group, category, s))
            }
        }

        // Sort by group first, then by category name within group
        rows.sort { (lhs, rhs) in
            if lhs.group != rhs.group {
                return lhs.group < rhs.group
            }
            return lhs.category.rawValue < rhs.category.rawValue
        }

        return rows
    }

    // MARK: - Public API

    /// Print all recorded stats as a parsable aligned table (ASCII format)
    ///
    /// Output format (grouped by category):
    /// ```
    /// +---------+------------------+----------+----------+----------+-------+----------+
    /// | Group   | Metric           |   Min ms |   Avg ms |   Max ms | Count | Total ms |
    /// +---------+------------------+----------+----------+----------+-------+----------+
    /// | main    | ModelLoad        |  1540.60 |  1540.60 |  1540.60 |     1 |  1540.60 |
    /// | main    | TokenizerLoad    |  1309.80 |  1309.80 |  1309.80 |     1 |  1309.80 |
    /// | main    | Warmup           |   437.28 |   437.28 |   437.28 |     1 |   437.28 |
    /// | decoder | Decode           |     0.04 |     0.08 |     0.12 |   100 |     7.46 |
    /// | decoder | Extend           |     8.25 |    10.50 |    42.15 |    99 |  1039.50 |
    /// | engine  | LogitsInference  |     6.79 |     8.02 |    36.75 |   100 |   802.15 |
    /// +---------+------------------+----------+----------+----------+-------+----------+
    /// ```
    ///
    public func printVerboseTable() {
        // Step 1: Fetch and sort data
        let rows = fetchAndSortRows()

        guard !rows.isEmpty else {
            print("No profiling data recorded.")
            return
        }

        // Step 2: Calculate layout
        let layout = TableLayout.calculate(rows: rows)

        // Step 3: Build border and header
        let border = Self.buildBorder(layout: layout)
        let header = Self.buildHeader(layout: layout)

        // Step 4: Print table
        print("")
        print("==> Detailed Profiling Statistics (Overlapping, See docs/Runtime_Signposts.md)")
        print(border)
        print(header)
        print(border)

        // Step 5: Print data rows
        for row in rows {
            print(Self.formatRow(row, layout: layout))
        }

        print(border)
        print("")
    }
}

/// Enhanced profiling system that integrates with Instruments
public struct InstrumentsProfiler {
    private static let log = OSLog(subsystem: "com.apple.coreai-models.performance", category: "performance")

    // MARK: - Unified Profiling Span API (New)

    /// Begin profiling prompt processing (first token, processes all input tokens at once)
    public static func beginPrompt(tokens: Int, engine: String? = nil) -> ProfileSpan {
        var metadata: [String: String] = ["tokens": "\(tokens)"]
        if let engine = engine {
            metadata["engine"] = engine
        }
        return ProfileSpan(category: .prompt, log: Self.log, metadata: metadata)
    }

    /// Begin profiling decode step (token→text conversion)
    public static func beginDecode(step: Int) -> ProfileSpan {
        let metadata: [String: String] = ["step": "\(step)"]
        return ProfileSpan(category: .decode, log: Self.log, metadata: metadata)
    }

    /// Begin profiling warmup step (kernel compilation)
    public static func beginWarmup(step: Int? = nil) -> ProfileSpan {
        var metadata: [String: String] = [:]
        if let step = step {
            metadata["step"] = "\(step)"
        }
        return ProfileSpan(category: .warmup, log: Self.log, metadata: metadata)
    }

    /// Begin profiling sampling operation
    public static func beginSample(strategy: String? = nil, temperature: Double? = nil) -> ProfileSpan {
        var metadata: [String: String] = [:]
        if let strategy = strategy {
            metadata["strategy"] = strategy
        }
        if let temperature = temperature {
            metadata["temperature"] = String(format: "%.3f", temperature)
        }
        return ProfileSpan(category: .sample, log: Self.log, metadata: metadata)
    }

    /// Begin profiling GPU sample encoding (command buffer submission)
    ///
    /// Used by pipelined engines that encode GPU sampling commands asynchronously.
    /// Measures CPU time to encode the sampler command buffer, not GPU execution time.
    ///
    /// - Parameters:
    ///   - step: Generation step number (optional, enables timeline tracking when provided)
    ///   - strategy: Sampling strategy name (e.g., "greedy", "temperature")
    ///   - temperature: Temperature value for sampling
    public static func beginSampleEncoding(step: Int? = nil, strategy: String, temperature: Double) -> ProfileSpan {
        var metadata: [String: String] = [
            "strategy": strategy,
            "temperature": String(format: "%.3f", temperature),
        ]
        if let step = step {
            metadata["step"] = "\(step)"
        }
        return ProfileSpan(category: .sampleEncoding, log: Self.log, metadata: metadata)
    }

    /// Begin profiling GPU sampling operation (tracks time from submission to GPU completion)
    ///
    /// Used by pipelined engines to measure actual GPU sampling duration.
    /// Create before runAsync(), end in completion handler.
    ///
    /// - Parameters:
    ///   - step: Generation step number (required for timeline tracking)
    ///   - strategy: Sampling strategy name (e.g., "argmax", "temperature")
    ///   - temperature: Temperature value for sampling
    public static func beginSample(step: Int, strategy: String, temperature: Double) -> ProfileSpan {
        let metadata: [String: String] = [
            "step": "\(step)",
            "strategy": strategy,
            "temperature": String(format: "%.3f", temperature),
        ]
        return ProfileSpan(category: .sample, log: Self.log, metadata: metadata)
    }

    /// Begin profiling model loading
    public static func beginModelLoad(name: String) -> ProfileSpan {
        let metadata: [String: String] = [
            "model": name
        ]
        return ProfileSpan(category: .modelLoad, log: Self.log, metadata: metadata)
    }

    /// Begin profiling tokenizer loading
    public static func beginTokenizerLoad(id: String) -> ProfileSpan {
        let metadata: [String: String] = ["tokenizer": id]
        return ProfileSpan(category: .tokenizerLoad, log: Self.log, metadata: metadata)
    }

    /// Begin profiling tokenization
    public static func beginTokenization(inputLength: Int) -> ProfileSpan {
        let metadata: [String: String] = ["inputLength": "\(inputLength)"]
        return ProfileSpan(category: .tokenization, log: Self.log, metadata: metadata)
    }

    /// Begin profiling a logits inference step (engine layer: model forward pass)
    ///
    /// Measures the time for the model to process input tokens and produce logits.
    ///
    /// - Parameters:
    ///   - step: Generation step number (optional, enables timeline tracking when provided)
    ///   - tokens: Number of tokens being processed
    ///   - processedCount: Total tokens processed so far
    ///   - engine: Engine identifier
    public static func beginLogitsInference(step: Int? = nil, tokens: Int, processedCount: Int, engine: String? = nil)
        -> ProfileSpan
    {
        var metadata: [String: String] = [
            "tokens": "\(tokens)",
            "processed": "\(processedCount)",
        ]
        if let step = step {
            metadata["step"] = "\(step)"
        }
        if let engine = engine {
            metadata["engine"] = engine
        }
        return ProfileSpan(category: .logitsInference, log: Self.log, metadata: metadata)
    }

    /// Begin profiling a logits inference step (for engines with GPU callbacks)
    ///
    /// Measures the time for the model to process input tokens and produce logits.
    /// Used by pipelined inference engines with completion handlers.
    ///
    /// - Parameters:
    ///   - step: Generation step number (required for timeline tracking)
    ///   - tokens: Number of tokens being processed
    ///   - engine: Engine identifier
    public static func beginLogitsInference(step: Int, tokens: Int? = nil, engine: String? = nil) -> ProfileSpan {
        var metadata: [String: String] = ["step": "\(step)"]
        if let tokens = tokens {
            metadata["tokens"] = "\(tokens)"
        }
        if let engine = engine {
            metadata["engine"] = engine
        }
        return ProfileSpan(category: .logitsInference, log: Self.log, metadata: metadata)
    }

    /// Begin profiling an extend step (strategy layer: inter-token timing)
    ///
    /// Measures the wall-clock time between receiving one token and the next.
    /// Used by VanillaDecodingStrategy to track autoregressive generation timing.
    /// Includes: engine inference + sampling + any overhead.
    /// NOTE: Use beginPrompt() for step 0 (prefill phase).
    public static func beginExtend(step: Int, tokens: Int? = nil) -> ProfileSpan {
        var metadata: [String: String] = ["step": "\(step)"]
        if let tokens = tokens {
            metadata["tokens"] = "\(tokens)"
        }
        return ProfileSpan(category: .extend, log: Self.log, metadata: metadata)
    }

    /// Begin profiling step preparation (tensor reshape, rebind, input writes)
    ///
    /// Measures CPU time to prepare inputs for the next inference step.
    /// Includes: tensor layout reshape, executable rebind, and writing tokens/cache positions.
    ///
    /// - Parameters:
    ///   - step: Generation step number (optional, enables timeline tracking when provided)
    ///   - operation: Description of operation (e.g., "reshape+write", "rebind")
    ///   - engine: Engine identifier (e.g., "Core AI-Pipelined")
    public static func beginPrepareStep(step: Int? = nil, operation: String, engine: String? = nil) -> ProfileSpan {
        var metadata: [String: String] = ["operation": operation]
        if let step = step {
            metadata["step"] = "\(step)"
        }
        if let engine = engine {
            metadata["engine"] = engine
        }
        return ProfileSpan(category: .prepareStep, log: Self.log, metadata: metadata)
    }

    /// Begin profiling cache management operations (KV cache growth, reallocation)
    ///
    /// Measures time for actual KV cache operations like growth and data copying.
    /// Used when cache capacity is insufficient and needs to be expanded.
    ///
    /// - Parameters:
    ///   - step: Generation step number (optional, enables timeline tracking when provided)
    ///   - operation: Description of cache operation (e.g., "grow", "rebind", "init")
    ///   - engine: Engine identifier
    public static func beginCacheManagement(step: Int? = nil, operation: String, engine: String? = nil) -> ProfileSpan {
        var metadata: [String: String] = ["operation": operation]
        if let step = step {
            metadata["step"] = "\(step)"
        }
        if let engine = engine {
            metadata["engine"] = engine
        }
        return ProfileSpan(category: .cacheManagement, log: Self.log, metadata: metadata)
    }

    // MARK: - PrepareStep sub-spans (fine-grained per-token runner attribution)

    /// Embedding-gather graph dispatch (a separate forward per decode step).
    public static func beginGatherEmbeddings() -> ProfileSpan {
        ProfileSpan(category: .gatherEmbeddings, log: Self.log, metadata: [:])
    }

    /// Sliding-window mask construction for the step.
    public static func beginMaskBuild() -> ProfileSpan {
        ProfileSpan(category: .maskBuild, log: Self.log, metadata: [:])
    }

    /// Runner-side RoPE cos/sin precompute for the step's tokens (Gemma4).
    public static func beginRopeBuild() -> ProfileSpan {
        ProfileSpan(category: .ropeBuild, log: Self.log, metadata: [:])
    }

    /// INT8 per-layer-embedding (PLE) gather for the step's tokens.
    public static func beginPLEGather() -> ProfileSpan {
        ProfileSpan(category: .pleGather, log: Self.log, metadata: [:])
    }

    /// Full-vocab logits copy out of the graph output buffer.
    public static func beginLogitsCopy() -> ProfileSpan {
        ProfileSpan(category: .logitsCopy, log: Self.log, metadata: [:])
    }

    /// Begin profiling engine reset operations
    public static func beginReset(engine: String? = nil) -> ProfileSpan {
        var metadata: [String: String] = [:]
        if let engine = engine {
            metadata["engine"] = engine
        }
        return ProfileSpan(category: .reset, log: Self.log, metadata: metadata)
    }

    /// Begin profiling engine cleanup operations
    public static func beginCleanup(engine: String? = nil) -> ProfileSpan {
        var metadata: [String: String] = [:]
        if let engine = engine {
            metadata["engine"] = engine
        }
        return ProfileSpan(category: .cleanup, log: Self.log, metadata: metadata)
    }

    // MARK: - Legacy Bracket Signposts (for Instruments visualization)
    //
    // These are intentional visual markers in Instruments, not for timing aggregation.
    // They bracket the entire inference/decoding operation for easy visual identification.

    private static let tokenizationName = StaticString("Tokenization")
    private static let inferenceName = StaticString("Inference")
    private static let decodingName = StaticString("DecodingStrategy")
    private static let tokenGenerationName = StaticString("TokenGeneration")
    private static let memoryUsageName = StaticString("MemoryUsage")
    private static let customIntervalName = StaticString("CustomInterval")

    private init() {}

    /// Log tokenizer completion event - marks end of all tokenizer work
    public static func logTokenizerComplete(tokenCount: Int) {
        os_signpost(
            .event,
            log: Self.log,
            name: Self.tokenizationName,
            "Tokenizer complete: %{public}d prompt tokens ready",
            tokenCount
        )
    }

    // MARK: - Inference Bracket (visual marker in Instruments)

    public static func beginInference(promptTokens: Int, maxTokens: Int) -> OSSignpostID {
        let inferenceSignpost = OSSignpostID(log: Self.log)
        os_signpost(
            .begin,
            log: Self.log,
            name: Self.inferenceName,
            signpostID: inferenceSignpost,
            "Starting inference: %{public}d prompt tokens, max %{public}d generation tokens",
            promptTokens,
            maxTokens
        )
        return inferenceSignpost
    }

    public static func endInference(generatedTokens: Int, signpostID: OSSignpostID) {
        os_signpost(
            .end,
            log: Self.log,
            name: Self.inferenceName,
            signpostID: signpostID,
            "Generated %{public}d tokens",
            generatedTokens
        )
    }

    // MARK: - Decoding Bracket (visual marker in Instruments)

    public static func beginDecoding(strategy: String) -> OSSignpostID {
        let decodingSignpost = OSSignpostID(log: Self.log)
        os_signpost(
            .begin,
            log: Self.log,
            name: Self.decodingName,
            signpostID: decodingSignpost,
            "Using %{public}s decoding",
            strategy
        )
        return decodingSignpost
    }

    public static func endDecoding(signpostID: OSSignpostID) {
        os_signpost(.end, log: Self.log, name: Self.decodingName, signpostID: signpostID)
    }

    // MARK: - Event Logging

    public static func logTokenGeneration(tokenIndex: Int, token: String) {
        os_signpost(
            .event,
            log: Self.log,
            name: Self.tokenGenerationName,
            "Token %{public}d: %{public}s",
            tokenIndex,
            token
        )
    }

    public static func logMemoryUsage(phase: String) {
        var memoryInfo = MachTaskBasicInfo()
        var count = mach_msg_type_number_t(MemoryLayout<MachTaskBasicInfo>.size) / 4

        let result = withUnsafeMutablePointer(to: &memoryInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(machTaskBasicInfo), $0, &count)
            }
        }

        if result == KERN_SUCCESS {
            let memoryMB = Double(memoryInfo.residentSize) / (1024 * 1024)
            os_signpost(
                .event, log: Self.log, name: Self.memoryUsageName,
                "%{public}s: %.1f MB", phase, memoryMB)
        }
    }

    // MARK: - Custom Intervals (for Core AI pipelined engine async spans)

    public static func beginCustomInterval(name: String, details: String = "") -> OSSignpostID {
        let signpostID = OSSignpostID(log: Self.log)
        if details.isEmpty {
            os_signpost(
                .begin,
                log: Self.log,
                name: Self.customIntervalName,
                signpostID: signpostID,
                "%{public}s",
                name
            )
        } else {
            os_signpost(
                .begin,
                log: Self.log,
                name: Self.customIntervalName,
                signpostID: signpostID,
                "%{public}s: %{public}s",
                name,
                details
            )
        }
        return signpostID
    }

    public static func endCustomInterval(name: String, signpostID: OSSignpostID, details: String = "") {
        if details.isEmpty {
            os_signpost(
                .end,
                log: Self.log,
                name: Self.customIntervalName,
                signpostID: signpostID,
                "%{public}s",
                name
            )
        } else {
            os_signpost(
                .end,
                log: Self.log,
                name: Self.customIntervalName,
                signpostID: signpostID,
                "%{public}s: %{public}s",
                name,
                details
            )
        }
    }
}

// MARK: - Mach Task Info Structure

// swiftformat:disable all
private struct MachTaskBasicInfo {
    var virtualSize: mach_vm_size_t = 0
    var residentSize: mach_vm_size_t = 0
    var residentSizeMax: mach_vm_size_t = 0
    var userTime: time_value_t = time_value_t()
    var systemTime: time_value_t = time_value_t()
    var policy: policy_t = 0
    var suspendCount: integer_t = 0
}

private let machTaskBasicInfo: Int32 = 20
