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
    /// The ids a `--raw-continuation` run scores.
    struct Request: Equatable {
        /// Context tokens.
        let context: [Int32]
        /// Continuation tokens, each one scored.
        let continuation: [Int32]
    }

    /// Reads the tokens to score, refusing options that conflict, before the model loads.
    ///
    /// - Parameters:
    ///   - rawContinuationPath: File of continuation tokens to score, or `nil` when the run generates.
    ///   - rawTokensPath: File of context tokens, or `nil`.
    ///   - textContinuationGiven: Whether a text continuation is also given.
    ///   - logitsRequested: Whether the run prints or saves logits.
    ///   - conflictingFlags: Options on this run that take their own path and would skip the scoring.
    /// - Returns: The context and continuation tokens, or `nil` when the run generates.
    /// - Throws: `ContinuationOptionError` when the options conflict, or when a file cannot be read or
    ///   holds no tokens.
    static func request(
        rawContinuationPath: String?,
        rawTokensPath: String?,
        textContinuationGiven: Bool,
        logitsRequested: Bool,
        conflictingFlags: [String] = []
    ) throws -> Request? {
        guard let path = rawContinuationPath else {
            // A file that cannot be read is left to the prompt resolver, which reports it as before.
            if let contextPath = rawTokensPath, let context = try? readTokens(at: contextPath), context.isEmpty {
                throw ContinuationOptionError.emptyRawTokens(path: contextPath)
            }
            return nil
        }
        guard !textContinuationGiven else {
            throw ContinuationOptionError.rawContinuationConflicts(flag: "--continuation")
        }
        if let flag = conflictingFlags.first {
            throw ContinuationOptionError.rawContinuationConflicts(flag: flag)
        }
        guard let contextPath = rawTokensPath else {
            throw ContinuationOptionError.rawContinuationRequiresRawTokens
        }
        guard logitsRequested else {
            throw ContinuationOptionError.rawContinuationRequiresLogitsOutput
        }
        let context = try readTokens(at: contextPath)
        guard !context.isEmpty else {
            throw ContinuationOptionError.emptyRawContext(path: contextPath)
        }
        let continuation = try readTokens(at: path)
        guard !continuation.isEmpty else {
            throw ContinuationOptionError.emptyRawContinuation(path: path)
        }
        return Request(context: context, continuation: continuation)
    }

    /// Produces a notice when a run asks for behaviour that is no longer available: raw tokens with
    /// a logits output used to score a continuation, and now generate. The notice says how to split
    /// the input to score instead. Nil when it does not apply.
    static func generationNotice(
        rawTokensGiven: Bool, rawContinuationGiven: Bool, textContinuationGiven: Bool, logitsRequested: Bool
    ) -> String? {
        guard rawTokensGiven, !rawContinuationGiven, !textContinuationGiven, logitsRequested else { return nil }
        return "Note: --raw-tokens is the context, so this run generates and saves the logits of each "
            + "generated token. To score a forced continuation instead, pass its ids with "
            + "--raw-continuation. Runs that relied on --raw-tokens [t0, t1, ..., tn] being scored "
            + "should pass --raw-tokens [t0] --raw-continuation [t1, ..., tn]."
    }

    /// Checks the given request against the model, to ensure every token is in its vocabulary and
    /// the context and continuation together fit its context length. Needs only the bundle's
    /// config, so it runs before the model loads.
    static func checkFits(_ request: Request, vocabSize: Int, maxContextLength: Int) throws {
        for (flag, ids) in [("--raw-tokens", request.context), ("--raw-continuation", request.continuation)] {
            if let id = ids.first(where: { $0 < 0 || Int($0) >= vocabSize }) {
                throw ContinuationOptionError.tokenOutsideVocabulary(flag: flag, id: id, vocabSize: vocabSize)
            }
        }
        let total = request.context.count + request.continuation.count
        guard total <= maxContextLength else {
            throw ContinuationOptionError.exceedsContextLength(
                context: request.context.count, continuation: request.continuation.count,
                maxContextLength: maxContextLength)
        }
    }

    /// Masks each continuation position's logits with a JSON schema's grammar, so log-probabilities
    /// are over the tokens the schema allows there. Once the JSON is complete, only the grammar's stop
    /// tokens are allowed. The context is not constrained.
    ///
    /// - Throws: `ContinuationOptionError` when a continuation token is one the schema does not allow
    ///   at its position, or comes after the grammar has ended.
    static func maskedByGrammar(
        _ result: ContinuationEvaluationResult, session: inout ConstrainedGenerationSession
    ) throws -> ContinuationEvaluationResult {
        var masked: [[LogitsScalarType]] = []
        masked.reserveCapacity(result.logits.count)
        for (position, (logits, id)) in zip(result.logits, result.continuationTokens).enumerated() {
            var row = logits
            guard session.applyMask(to: &row) else {
                throw ContinuationOptionError.continuationPastSchemaEnd(position: position, id: id)
            }
            guard session.acceptToken(id) else {
                throw ContinuationOptionError.continuationViolatesSchema(position: position, id: id)
            }
            masked.append(row)
        }
        return ContinuationEvaluationResult(
            contextTokens: result.contextTokens, continuationTokens: result.continuationTokens, logits: masked)
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
enum ContinuationOptionError: Error, LocalizedError, Equatable {
    case requiresDisabledChatTemplate
    case requiresLogitsOutput
    case engineDoesNotSupportLogits
    case emptyContinuation
    case rawTokensNotSupported
    case rawContinuationRequiresRawTokens
    case rawContinuationRequiresLogitsOutput
    case rawContinuationConflicts(flag: String)
    case unreadableRawTokens(path: String, reason: String)
    case emptyRawContext(path: String)
    case emptyRawTokens(path: String)
    case emptyRawContinuation(path: String)
    case tokenOutsideVocabulary(flag: String, id: Int32, vocabSize: Int)
    case exceedsContextLength(context: Int, continuation: Int, maxContextLength: Int)
    case continuationViolatesSchema(position: Int, id: Int32)
    case continuationPastSchemaEnd(position: Int, id: Int32)

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
        case .rawContinuationConflicts(let flag):
            return "--raw-continuation cannot be used with \(flag)"
        case .unreadableRawTokens(let path, let reason):
            return "Cannot read token file \(path): \(reason) "
                + "Expected format: {\"tokens\": [1, 2, 3, ...]}"
        case .emptyRawContext(let path):
            return "--raw-tokens file \(path) holds no tokens, so there is no context to score against"
        case .emptyRawTokens(let path):
            return "--raw-tokens file \(path) holds no tokens, so there is nothing to generate from"
        case .emptyRawContinuation(let path):
            return "--raw-continuation file \(path) holds no tokens"
        case .tokenOutsideVocabulary(let flag, let id, let vocabSize):
            return "\(flag) holds token id \(id), outside the model's vocabulary of \(vocabSize)"
        case .exceedsContextLength(let context, let continuation, let maxContextLength):
            return "--raw-tokens (\(context) ids) plus --raw-continuation (\(continuation) ids) is "
                + "\(context + continuation) ids, more than the model's context length of \(maxContextLength)"
        case .continuationViolatesSchema(let position, let id):
            return "--raw-continuation id \(id) at position \(position) is not allowed by --json-schema"
        case .continuationPastSchemaEnd(let position, let id):
            return "--raw-continuation id \(id) at position \(position) comes after the --json-schema JSON is complete"
        }
    }
}
