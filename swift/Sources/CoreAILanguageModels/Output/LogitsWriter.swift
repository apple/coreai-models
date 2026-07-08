// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Tokenizers

// MARK: - Logits Configuration

/// Maximum allowed top-K value (prevents excessive memory/output)
private let kMaxTopK = 20
/// Default top-K for console display
private let kDefaultConsoleTopK = 5

/// Specifies how many logits to save per generation step
public enum LogitsLength: Sendable {
    case full
    case count(Int)

    public init?(argument: String) {
        if argument.lowercased() == "full" {
            self = .full
            return
        }

        if let intValue = Int(argument) {
            guard (0...kMaxTopK).contains(intValue) else {
                return nil
            }
            self = .count(intValue)
            return
        }
        return nil
    }

    public var defaultValueDescription: String {
        switch self {
        case .full:
            return "full"
        case .count(let val):
            return "\(val)"
        }
    }

    /// Whether this represents full logits mode
    public var isFull: Bool {
        if case .full = self { return true }
        return false
    }

    /// Returns the top-K count for file output (capped at kMaxTopK)
    /// For full mode, returns the full vocab (uncapped)
    public var topKForFile: Int? {
        switch self {
        case .full:
            return nil  // Full mode saves all logits
        case .count(let k):
            return min(k, kMaxTopK)
        }
    }

    /// Returns the top-K count for console display
    /// Full mode uses the default console top-K for display
    public var topKForConsole: Int {
        switch self {
        case .full:
            return kDefaultConsoleTopK
        case .count(let k):
            return min(k, kMaxTopK)
        }
    }
}

// MARK: - Logits Data Structures

/// Represents logits information for a single generated token
public struct TokenLogits: Sendable {
    public let tokenId: Int32
    public let tokenText: String
    public let topLogits: [TopLogitEntry]

    public init(tokenId: Int32, tokenText: String, topLogits: [TopLogitEntry]) {
        self.tokenId = tokenId
        self.tokenText = tokenText
        self.topLogits = topLogits
    }
}

/// Represents a single entry in top-K logits
public struct TopLogitEntry: Codable, Sendable {
    public let tokenId: Int32
    public let tokenText: String
    public let logit: Float

    enum CodingKeys: String, CodingKey {
        case tokenId = "token_id"
        case tokenText = "incremental_text"
        case logit
    }
}

/// Top-level JSON structure for top-K logits output
struct LogitsOutput: Codable {
    let tokens: [TokenLogitsJSON]
}

/// JSON representation of token with top-K logits
struct TokenLogitsJSON: Codable {
    let tokenId: Int32
    let tokenText: String
    let topLogits: [TopLogitEntry]

    enum CodingKeys: String, CodingKey {
        case tokenId = "token_id"
        case tokenText = "incremental_text"
        case topLogits = "top_logits"
    }
}

/// Top-level JSON structure for full logits output
struct FullLogitsOutput: Codable {
    let tokens: [FullTokenLogitsJSON]
}

/// JSON representation of token with full logits (base64 encoded)
struct FullTokenLogitsJSON: Codable {
    let tokenId: Int32
    let tokenText: String
    let logitsBase64: String

    enum CodingKeys: String, CodingKey {
        case tokenId = "token_id"
        case tokenText = "incremental_text"
        case logitsBase64 = "logits_base64"
    }
}

// MARK: - Logits Writer

/// Utility for saving logits in various formats
public struct LogitsWriter {
    /// Escape special characters for display
    private static func escapeForDisplay(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\t", with: "\\t")
    }

    /// Save top-K logits to JSON file
    /// - Parameters:
    ///   - tokenLogits: Array of token logits information
    ///   - path: Output file path
    public static func saveTopKJSON(tokenLogits: [TokenLogits], path: String) throws {
        let jsonTokens = tokenLogits.map { token in
            TokenLogitsJSON(
                tokenId: token.tokenId,
                tokenText: token.tokenText,
                topLogits: token.topLogits
            )
        }

        let output = LogitsOutput(tokens: jsonTokens)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let data = try encoder.encode(output)
        let url = URL(fileURLWithPath: path)
        try data.write(to: url)

        CLILogger.log("Saved top-K logits to \(path)", component: "LogitsWriter")
    }

    /// Encode Float16 array to base64 string
    /// - Parameter logits: Array of Float16 logits
    /// - Returns: Base64-encoded string representation
    private static func encodeLogitsToBase64<T>(_ logits: [T]) -> String {
        let data = logits.withUnsafeBytes { Data($0) }
        return data.base64EncodedString()
    }

    /// Save full logits to JSON file with base64-encoded Float16 arrays
    /// - Parameters:
    ///   - logits: 2D array of logits [num_tokens, vocab_size]
    ///   - generatedTokens: Array of generated token IDs
    ///   - tokenizer: Tokenizer for decoding token IDs to text
    ///   - path: Output file path
    public static func saveFullJSON<T: BinaryFloatingPoint>(
        logits: [[T]],
        generatedTokens: [Int],
        tokenizer: any Tokenizer,
        path: String
    ) throws {
        guard !logits.isEmpty else {
            throw LogitsWriterError.emptyLogits
        }

        // Token count must match logits count - each generation step produces one token and one logits vector
        guard generatedTokens.count == logits.count else {
            throw LogitsWriterError.tokenCountMismatch(
                tokenCount: generatedTokens.count,
                logitsCount: logits.count
            )
        }

        let numTokens = logits.count
        let vocabSize = logits[0].count

        // Validate all rows have same size
        for (index, row) in logits.enumerated() {
            guard row.count == vocabSize else {
                throw LogitsWriterError.inconsistentVocabSize(
                    "Row \(index) has \(row.count) elements, expected \(vocabSize)"
                )
            }
        }

        // Build JSON tokens array with base64-encoded logits
        let jsonTokens = logits.enumerated().map { index, logitVector in
            let tokenId = Int32(generatedTokens[index])
            let tokenText = tokenizer.decode(tokens: [generatedTokens[index]])

            return FullTokenLogitsJSON(
                tokenId: tokenId,
                tokenText: tokenText,
                logitsBase64: encodeLogitsToBase64(logitVector)
            )
        }

        let output = FullLogitsOutput(tokens: jsonTokens)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let data = try encoder.encode(output)
        let url = URL(fileURLWithPath: path)
        try data.write(to: url)

        let fileSize = data.count / 1024 / 1024  // Convert to MB
        CLILogger.log(
            "Saved full logits to \(path) (tokens: \(numTokens), vocab: \(vocabSize), size: \(fileSize)MB)",
            component: "LogitsWriter"
        )
    }

    /// Extract top-K logits from a full logits array
    /// - Parameters:
    ///   - logits: Full logits array
    ///   - tokenizer: Tokenizer for decoding token IDs
    ///   - k: Number of top tokens to extract
    /// - Returns: Array of top-K logit entries
    public static func extractTopK<T: BinaryFloatingPoint>(
        from logits: [T],
        tokenizer: any Tokenizer,
        k: Int
    ) -> [TopLogitEntry] {
        extractTopK(from: logits.map({ Float($0) }), tokenizer: tokenizer, k: k)
    }

    public static func extractTopK(
        from logits: [Float],
        tokenizer: any Tokenizer,
        k: Int
    ) -> [TopLogitEntry] {
        // Create array of (index, logit) pairs
        let indexedLogits = logits.enumerated().map { ($0.offset, $0.element) }

        // Sort by logit value (descending) and take top K
        let topK =
            indexedLogits
            .sorted { $0.1 > $1.1 }
            .prefix(k)

        // Convert to TopLogitEntry
        return topK.map { index, logit in
            let tokenId = Int32(index)
            let tokenText = tokenizer.decode(tokens: [Int(tokenId)])
            return TopLogitEntry(
                tokenId: tokenId,
                tokenText: tokenText,
                logit: logit
            )
        }
    }

    /// Print top-K logits to console
    /// - Parameters:
    ///   - tokenId: ID of the chosen token
    ///   - tokenText: Text of the chosen token
    ///   - topLogits: Top-K logit entries
    ///   - stepNumber: Generation step number
    public static func printTopK(
        tokenId: Int32,
        tokenText: String,
        topLogits: [TopLogitEntry],
        stepNumber: Int
    ) {
        print("Token \(stepNumber): \"\(tokenText)\" (id: \(tokenId))")
        print("  Top candidates:")
        for (index, entry) in topLogits.enumerated().prefix(kDefaultConsoleTopK) {
            let escapedText = escapeForDisplay(entry.tokenText)
            print(
                String(
                    format: "    %d. \"%@\" (%d): %.2f",
                    index + 1, escapedText, entry.tokenId, entry.logit))
        }
    }

    /// Handle all logits output based on provided options
    /// - Parameters:
    ///   - logits: Array of logits for each generated token
    ///   - generatedText: The generated text
    ///   - tokenizer: Tokenizer for decoding token IDs
    ///   - saveLogitsLength: Specifies whether to save top-K or full logits
    ///   - saveJsonPath: Optional path to save logits as JSON
    ///   - printToConsole: Whether to print top-5 to console
    public static func handleOutput<T: BinaryFloatingPoint>(
        logits: [[T]],
        generatedText: String,
        tokenizer: any Tokenizer,
        saveLogitsLength: LogitsLength,
        saveJsonPath: String?,
        printToConsole: Bool
    ) throws {
        // Decode generated text to get token IDs
        let generatedTokens = tokenizer.encode(text: generatedText)

        // Token count must match logits count - each generation step produces one token and one logits vector
        guard generatedTokens.count == logits.count else {
            throw LogitsWriterError.tokenCountMismatch(
                tokenCount: generatedTokens.count,
                logitsCount: logits.count
            )
        }

        let tokenCount = generatedTokens.count

        // Only build top-K array if we need it for printing or saving top-K
        let needsTopKArray = printToConsole || !saveLogitsLength.isFull
        var tokenLogitsArray: [TokenLogits] = []

        guard needsTopKArray else {
            // Full mode without console printing - skip top-K extraction entirely
            if let jsonPath = saveJsonPath {
                try saveFullJSON(
                    logits: logits,
                    generatedTokens: generatedTokens,
                    tokenizer: tokenizer,
                    path: jsonPath
                )
                // Note: saveFullJSON already logs the save
            }
            return
        }

        let topK = saveLogitsLength.topKForConsole
        for index in 0..<tokenCount {
            let logitVector = logits[index]
            let tokenId = Int32(generatedTokens[index])
            let tokenText = tokenizer.decode(tokens: [generatedTokens[index]])

            // Extract top-K logits
            let topLogits = extractTopK(
                from: logitVector,
                tokenizer: tokenizer,
                k: topK
            )

            tokenLogitsArray.append(
                TokenLogits(
                    tokenId: tokenId,
                    tokenText: tokenText,
                    topLogits: topLogits
                ))

            // Print to console if requested
            if printToConsole {
                printTopK(
                    tokenId: tokenId,
                    tokenText: tokenText,
                    topLogits: topLogits,
                    stepNumber: index + 1
                )
            }
        }

        // Save logits to JSON if requested (at this point we know it's top-K mode)
        if let jsonPath = saveJsonPath {
            do {
                try saveTopKJSON(tokenLogits: tokenLogitsArray, path: jsonPath)
                CLILogger.log("Saved top-\(topK) logits to \(jsonPath)", component: "LogitsWriter")
            } catch {
                CLILogger.log("Failed to save logits to JSON: \(error)", component: "LogitsWriter")
                throw error
            }
        }
    }

    // MARK: - Continuation Evaluation Output

    /// Handle evaluation output for continuation mode
    /// - Parameters:
    ///   - result: The continuation evaluation result
    ///   - context: Original context string
    ///   - continuation: Original continuation string
    ///   - tokenizer: Tokenizer for decoding token IDs
    ///   - saveLogitsLength: Specifies whether to save top-K or full logits
    ///   - saveJsonPath: Optional path to save logits as JSON
    ///   - printToConsole: Whether to print evaluation summary to console
    public static func handleEvaluationOutput(
        result: ContinuationEvaluationResult,
        context: String,
        continuation: String,
        tokenizer: any Tokenizer,
        saveLogitsLength: LogitsLength,
        saveJsonPath: String?,
        printToConsole: Bool
    ) throws {
        // Print header
        if printToConsole {
            print("\n=== Continuation Evaluation ===")
            print("Context: \"\(truncateForDisplay(context, maxLength: 60))\"")
            print("Continuation: \"\(continuation)\"")
            print("")
            print("Context tokens: \(result.contextTokens.count)")
            print("Continuation tokens: \(result.continuationTokens.count)")
            print("")
        }

        // Decode continuation tokens for display
        let continuationTexts = result.continuationTokens.map { tokenId in
            tokenizer.decode(tokens: [Int(tokenId)])
        }

        // Get target probabilities
        let targetProbs = result.targetProbabilities()

        // Use the helper to determine top-K for extraction
        let topK = saveLogitsLength.topKForConsole

        // Process each position
        for (i, (logitsVec, targetToken)) in zip(result.logits, result.continuationTokens).enumerated() {
            let targetText = i < continuationTexts.count ? continuationTexts[i] : "?"
            let targetProb = i < targetProbs.count ? targetProbs[i] : 0.0

            // Extract top-K for this position
            let topLogits = extractTopK(from: logitsVec, tokenizer: tokenizer, k: topK)

            if printToConsole {
                printEvaluationPosition(
                    position: i,
                    targetToken: targetToken,
                    targetText: targetText,
                    targetProbability: targetProb,
                    topLogits: topLogits
                )
            }
        }

        // Print summary statistics
        if printToConsole {
            print("")
            print("--- Summary ---")
            print(String(format: "Total log-probability: %.4f", result.logProbability()))
            print(String(format: "Average log-probability: %.4f", result.averageLogProbability()))
            print(String(format: "Perplexity: %.4f", result.perplexity()))
        }

        // Save to JSON if requested
        if let jsonPath = saveJsonPath {
            try saveEvaluationJSON(
                result: result,
                context: context,
                continuation: continuation,
                continuationTexts: continuationTexts,
                targetProbs: targetProbs,
                tokenizer: tokenizer,
                saveLogitsLength: saveLogitsLength,
                path: jsonPath
            )
            CLILogger.log("Saved evaluation results to \(jsonPath)", component: "LogitsWriter")
        }
    }

    /// Print evaluation results for a single position
    private static func printEvaluationPosition(
        position: Int,
        targetToken: Int32,
        targetText: String,
        targetProbability: Double,
        topLogits: [TopLogitEntry]
    ) {
        let escapedTarget = escapeForDisplay(targetText)

        print(String(format: "Position %d: target='%@' (ID: %d)", position, escapedTarget, targetToken))
        print(String(format: "  Probability: %.4f", targetProbability))
        print("  Top-\(kDefaultConsoleTopK):")

        for (index, entry) in topLogits.enumerated().prefix(kDefaultConsoleTopK) {
            let escapedText = escapeForDisplay(entry.tokenText)
            let marker = entry.tokenId == targetToken ? " ✓" : ""
            print(
                String(
                    format: "    %d. '%@' (%d): %.4f%@",
                    index + 1, escapedText, entry.tokenId, entry.logit, marker))
        }
        print("")
    }

    /// Save evaluation results to JSON
    private static func saveEvaluationJSON(
        result: ContinuationEvaluationResult,
        context: String,
        continuation: String,
        continuationTexts: [String],
        targetProbs: [Double],
        tokenizer: any Tokenizer,
        saveLogitsLength: LogitsLength,
        path: String
    ) throws {
        let isFull = saveLogitsLength.isFull
        let topK = saveLogitsLength.topKForConsole

        // Build positions array
        var positions: [[String: Any]] = []

        for (i, (logitsVec, targetToken)) in zip(result.logits, result.continuationTokens).enumerated() {
            let targetText = i < continuationTexts.count ? continuationTexts[i] : ""
            let targetProb = i < targetProbs.count ? targetProbs[i] : 0.0
            let targetLogProb = log(max(targetProb, 1e-10))

            var position: [String: Any] = [
                "index": i,
                "target_token": targetToken,
                "target_text": targetText,
                "target_probability": targetProb,
                "target_log_prob": targetLogProb,
            ]

            // Add top-K or full logits
            if isFull {
                position["logits_base64"] = encodeLogitsToBase64(logitsVec)
            } else {
                let topLogits = extractTopK(from: logitsVec, tokenizer: tokenizer, k: topK)
                position["top_k"] = topLogits.map { entry in
                    [
                        "token_id": entry.tokenId,
                        "token": entry.tokenText,
                        "logit": entry.logit,
                    ] as [String: Any]
                }
            }

            positions.append(position)
        }

        // Build full JSON structure
        let output: [String: Any] = [
            "context": context,
            "continuation": continuation,
            "context_tokens": result.contextTokens.map { Int($0) },
            "continuation_tokens": result.continuationTokens.map { Int($0) },
            "positions": positions,
            "total_log_probability": result.logProbability(),
            "average_log_probability": result.averageLogProbability(),
            "perplexity": result.perplexity(),
        ]

        // Serialize to JSON
        let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        let url = URL(fileURLWithPath: path)
        try data.write(to: url)
    }

    /// Truncate string for display with ellipsis
    private static func truncateForDisplay(_ string: String, maxLength: Int) -> String {
        if string.count <= maxLength {
            return string
        }
        let endIndex = string.index(string.startIndex, offsetBy: maxLength - 3)
        return String(string[..<endIndex]) + "..."
    }
}

// MARK: - Errors

public enum LogitsWriterError: Error, LocalizedError {
    case emptyLogits
    case inconsistentVocabSize(String)
    case tokenCountMismatch(tokenCount: Int, logitsCount: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyLogits:
            return "Cannot save empty logits array"
        case .inconsistentVocabSize(let message):
            return "Inconsistent vocabulary size: \(message)"
        case .tokenCountMismatch(let tokenCount, let logitsCount):
            return "Token count (\(tokenCount)) does not match logits count (\(logitsCount))"
        }
    }
}
