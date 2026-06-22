// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import AVFoundation
import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

// MARK: - SpeechBundle

/// Locates and loads the assets inside a CoreAISpeech model bundle directory.
///
/// A bundle directory contains:
///   encoder.aimodel      — audio features → encoder hidden states
///   decoder.aimodel      — autoregressive decoder with persistent state
///   generation_config.json (optional) — prefix, EOT token, etc.
///
/// The tokenizer is loaded from the local HF cache if it can be found there.
public struct SpeechBundle: Sendable {
    public let encoder: AIModel
    public let decoder: AIModel
    public let tokenizer: (any Tokenizer)?
    public let generationConfig: GenerationConfig

    public init(at url: URL) async throws {
        let encURL = url.appending(path: "encoder.aimodel")
        let decURL = url.appending(path: "decoder.aimodel")
        guard FileManager.default.fileExists(atPath: encURL.path),
            FileManager.default.fileExists(atPath: decURL.path)
        else {
            throw SpeechError.missingModel(
                "bundle at \(url.lastPathComponent) must contain encoder.aimodel and decoder.aimodel")
        }
        encoder = try await AIModel(contentsOf: encURL)
        decoder = try await AIModel(contentsOf: decURL)

        // Load generation config from bundle if present, otherwise use Whisper defaults
        let cfgURL = url.appending(path: "generation_config.json")
        generationConfig = (try? GenerationConfig(from: cfgURL)) ?? .whisper

        // Tokenizer — look in bundle first, then fall back to HF cache
        tokenizer = try? await Self.loadTokenizer(bundleURL: url, config: generationConfig)
    }

    private static func loadTokenizer(
        bundleURL: URL, config: GenerationConfig
    ) async throws -> (any Tokenizer)? {
        // 1. Try tokenizer files in the bundle itself
        if FileManager.default.fileExists(atPath: bundleURL.appending(path: "tokenizer.json").path) {
            return try? await AutoTokenizer.from(modelFolder: bundleURL)
        }
        // 2. Fall back to local HF cache using the model name from config
        if let name = config.tokenizerName {
            let cacheRoot = FileManager.default.homeDirectoryForCurrentUser
                .appending(path: ".cache/huggingface/hub")
            let folderName = "models--" + name.replacingOccurrences(of: "/", with: "--")
            let snapshotsDir = cacheRoot.appending(path: "\(folderName)/snapshots")
            if let snapshot = try? FileManager.default.contentsOfDirectory(
                atPath: snapshotsDir.path
            ).first {
                return try? await AutoTokenizer.from(
                    modelFolder: snapshotsDir.appending(path: snapshot))
            }
        }
        return nil
    }
}

// MARK: - GenerationConfig

/// Model-specific generation parameters, read from generation_config.json in the bundle.
public struct GenerationConfig: Sendable {
    /// Tokens prepended to every decode sequence before free generation.
    public let forcedPrefix: [Int32]
    /// Token that signals end of transcription.
    public let eotToken: Int32
    /// Maximum tokens to generate per call.
    public let maxDecodeSteps: Int
    /// HuggingFace model name for loading the tokenizer from cache.
    public let tokenizerName: String?

    /// Whisper large-v3-turbo defaults.
    public static let whisper = GenerationConfig(
        forcedPrefix: [50258, 50259, 50360, 50364],  // BOS <|en|> <|transcribe|> <|notimestamps|>
        eotToken: 50257,
        maxDecodeSteps: 50,
        tokenizerName: "openai/whisper-large-v3-turbo"
    )

    init(forcedPrefix: [Int32], eotToken: Int32, maxDecodeSteps: Int, tokenizerName: String?) {
        self.forcedPrefix = forcedPrefix
        self.eotToken = eotToken
        self.maxDecodeSteps = maxDecodeSteps
        self.tokenizerName = tokenizerName
    }

    init(from url: URL) throws {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        forcedPrefix = (json["forced_decoder_ids"] as? [Int]).map { $0.map { Int32($0) } } ?? Self.whisper.forcedPrefix
        eotToken = (json["eos_token_id"] as? Int).map { Int32($0) } ?? Self.whisper.eotToken
        maxDecodeSteps = (json["max_new_tokens"] as? Int) ?? Self.whisper.maxDecodeSteps
        tokenizerName = json["tokenizer_name"] as? String ?? Self.whisper.tokenizerName
    }
}

// MARK: - SpeechError

public enum SpeechError: Error, CustomStringConvertible {
    case missingModel(String)
    case missingTokenizer
    case invalidAudio(String)

    public var description: String {
        switch self {
        case .missingModel(let msg): return "Missing model: \(msg)"
        case .missingTokenizer:
            return "Tokenizer not found — ensure the model bundle includes a tokenizer or the HF cache is populated"
        case .invalidAudio(let msg): return "Invalid audio: \(msg)"
        }
    }
}
