// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAILanguageModels
import Foundation

// MARK: - Raw Token Scoring

/// Decides whether a `--raw-tokens` run scores a forced continuation, and reads the ids it scores.
///
/// `--raw-tokens` is always the context. A run scores only when `--raw-continuation` names the ids
/// to score, which mirrors `--prompt` and `--continuation` in the text mode. Scoring used to follow
/// from `--raw-tokens` plus a logits flag alone and took `tokens[0]` as the whole context, so no run
/// could generate from raw tokens and save the logits of what it generated.
enum RawTokenScoring {
    /// - Parameters:
    ///   - rawContinuationPath: The `--raw-continuation` file, in the `{"tokens": [...]}` format of
    ///     `--raw-tokens`, or `nil` when the run generates.
    ///   - rawTokensPath: The `--raw-tokens` file that gives the context, or `nil`.
    ///   - textContinuationGiven: Whether the text `--continuation` is also given.
    ///   - logitsRequested: Whether `--print-logits` or `--save-logits` is given.
    /// - Returns: The ids to score as a forced continuation of the whole `--raw-tokens` context, or
    ///   `nil` when the run generates from that context.
    /// - Throws: `ContinuationOptionError` when `--raw-continuation` is given with the text
    ///   `--continuation`, without `--raw-tokens`, without a logits output, or when either file
    ///   cannot be read or holds no ids. Both files are read here so the run refuses before the
    ///   model loads.
    static func continuation(
        rawContinuationPath: String?,
        rawTokensPath: String?,
        textContinuationGiven: Bool,
        logitsRequested: Bool
    ) throws -> [Int32]? {
        guard let path = rawContinuationPath else { return nil }
        guard !textContinuationGiven else {
            throw ContinuationOptionError.rawContinuationConflictsWithContinuation
        }
        guard let contextPath = rawTokensPath else {
            throw ContinuationOptionError.rawContinuationRequiresRawTokens
        }
        guard logitsRequested else {
            throw ContinuationOptionError.rawContinuationRequiresLogitsOutput
        }

        guard !(try readTokens(at: contextPath)).isEmpty else {
            throw ContinuationOptionError.emptyRawContext(path: contextPath)
        }
        let tokens = try readTokens(at: path)
        guard !tokens.isEmpty else {
            throw ContinuationOptionError.emptyRawContinuation(path: path)
        }
        return tokens
    }

    private static func readTokens(at path: String) throws -> [Int32] {
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        do {
            return try JSONDecoder().decode(RawTokensInput.self, from: Data(contentsOf: url)).tokens
        } catch {
            throw ContinuationOptionError.unreadableRawTokens(path: path, reason: error.localizedDescription)
        }
    }
}

// MARK: - Errors

/// Why an `llm-runner` run refuses its continuation options.
enum ContinuationOptionError: Error, LocalizedError {
    case requiresDisabledChatTemplate
    case requiresLogitsOutput
    case engineDoesNotSupportLogits
    case emptyContinuation
    case rawTokensNotSupported
    case rawContinuationRequiresRawTokens
    case rawContinuationRequiresLogitsOutput
    case rawContinuationConflictsWithContinuation
    case unreadableRawTokens(path: String, reason: String)
    case emptyRawContext(path: String)
    case emptyRawContinuation(path: String)

    var errorDescription: String? {
        switch self {
        case .requiresDisabledChatTemplate:
            return "--continuation requires --apply-chat-template=false"
        case .requiresLogitsOutput:
            return "--continuation requires --print-logits or --save-logits"
        case .engineDoesNotSupportLogits:
            return "The current inference engine does not support returning logits"
        case .emptyContinuation:
            return "Continuation string cannot be empty"
        case .rawTokensNotSupported:
            return "--continuation requires text prompt (--prompt or --prompt-file), not --raw-tokens"
        case .rawContinuationRequiresRawTokens:
            return "--raw-continuation requires --raw-tokens"
        case .rawContinuationRequiresLogitsOutput:
            return "--raw-continuation requires --print-logits or --save-logits"
        case .rawContinuationConflictsWithContinuation:
            return "--raw-continuation cannot be used with --continuation"
        case .unreadableRawTokens(let path, let reason):
            return "Cannot read token file \(path): \(reason) "
                + "Expected format: {\"tokens\": [1, 2, 3, ...]}"
        case .emptyRawContext(let path):
            return "--raw-tokens file \(path) holds no tokens, so there is no context to score against"
        case .emptyRawContinuation(let path):
            return "--raw-continuation file \(path) holds no tokens"
        }
    }
}
