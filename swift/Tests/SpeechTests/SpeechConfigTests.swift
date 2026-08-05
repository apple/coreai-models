// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

// MARK: - GenerationConfig

/// Covers the review comment asking for `GenerationConfig` to move off manual `JSONSerialization`
/// onto `Decodable`. The interesting cases are the failure modes, because a throw here makes the
/// caller fall back to `.whisper` wholesale rather than per field.
@Suite("GenerationConfig decoding")
struct GenerationConfigTests {
    private func config(_ json: String) throws -> GenerationConfig {
        try GenerationConfig(json: Data(json.utf8))
    }

    @Test("A full config decodes every field")
    func fullConfigDecodes() throws {
        let c = try config(
            """
            {"forced_decoder_ids": [1, 2, 3], "eos_token_id": 9,
             "max_new_tokens": 7, "tokenizer_name": "acme/model"}
            """)
        #expect(c.forcedPrefix == [1, 2, 3])
        #expect(c.eotToken == 9)
        #expect(c.maxDecodeSteps == 7)
        #expect(c.tokenizerName == "acme/model")
    }

    @Test("Absent fields fall back to the Whisper defaults")
    func absentFieldsFallBack() throws {
        let c = try config(#"{"eos_token_id": 42}"#)
        #expect(c.eotToken == 42)
        #expect(c.forcedPrefix == GenerationConfig.whisper.forcedPrefix)
        #expect(c.maxDecodeSteps == GenerationConfig.whisper.maxDecodeSteps)
        #expect(c.tokenizerName == GenerationConfig.whisper.tokenizerName)
    }

    @Test("An explicit null falls back")
    func explicitNullFallsBack() throws {
        let c = try config(#"{"forced_decoder_ids": null}"#)
        #expect(c.forcedPrefix == GenerationConfig.whisper.forcedPrefix)
    }

    @Test("An empty array is honoured rather than replaced")
    func emptyArrayIsHonoured() throws {
        // Easy to regress into `?? whisper.forcedPrefix` firing on empty rather than only on nil.
        let c = try config(#"{"forced_decoder_ids": []}"#)
        #expect(c.forcedPrefix.isEmpty)
    }

    @Test("Unknown keys are ignored")
    func unknownKeysIgnored() throws {
        let c = try config(#"{"eos_token_id": 5, "suppress_tokens": [1, 2], "begin": [220]}"#)
        #expect(c.eotToken == 5)
    }

    /// HuggingFace writes `forced_decoder_ids` as `[[position, tokenId]]` pairs and omits
    /// `decoder_start_token_id` from them, so such a file cannot be dropped in as-is. The decode
    /// throws, and the caller's `try?` then discards *every* field, not just this one.
    @Test("HuggingFace pair-form forced_decoder_ids fails to decode")
    func huggingFacePairFormFails() {
        let json = Data(#"{"forced_decoder_ids": [[1, 50259], [2, 50360]], "eos_token_id": 1}"#.utf8)
        #expect(throws: (any Error).self) { try GenerationConfig(json: json) }

        let effective = (try? GenerationConfig(json: json)) ?? .whisper
        #expect(effective.forcedPrefix == GenerationConfig.whisper.forcedPrefix)
        // The consequence worth pinning: the sibling `eos_token_id: 1` is lost too.
        #expect(effective.eotToken == GenerationConfig.whisper.eotToken)
    }

    @Test("Malformed JSON throws")
    func malformedJSONThrows() {
        #expect(throws: (any Error).self) { try GenerationConfig(json: Data("not json".utf8)) }
    }

    @Test("Loading from a URL matches loading from data")
    func urlMatchesData() throws {
        let json = #"{"eos_token_id": 11, "max_new_tokens": 3}"#
        try withTempDirectory { dir in
            let url = dir.appending(path: "generation_config.json")
            try Data(json.utf8).write(to: url)
            let fromURL = try GenerationConfig(from: url)
            let fromData = try config(json)
            #expect(fromURL.eotToken == fromData.eotToken)
            #expect(fromURL.maxDecodeSteps == fromData.maxDecodeSteps)
            #expect(fromURL.forcedPrefix == fromData.forcedPrefix)
        }
    }

    @Test("Whisper defaults are pinned")
    func whisperDefaultsPinned() {
        let w = GenerationConfig.whisper
        #expect(w.forcedPrefix == [50_258, 50_259, 50_360, 50_364])
        #expect(w.eotToken == 50_257)
        #expect(w.maxDecodeSteps == 50)
        #expect(w.tokenizerName == "openai/whisper-large-v3-turbo")
    }
}

// MARK: - Architecture detection

@Suite("Bundle architecture detection")
struct BundleArchitectureTests {
    private func architecture(_ json: String) -> SpeechRecognitionBundle.Architecture {
        SpeechRecognitionBundle.architecture(from: Data(json.utf8))
    }

    @Test("Explicit architectures are recognized")
    func explicitArchitectures() {
        #expect(architecture(#"{"config": {"architecture": "parakeet_tdt"}}"#) == .parakeetTDT)
        #expect(architecture(#"{"config": {"architecture": "whisper"}}"#) == .whisper)
    }

    /// Everything unparseable degrades to `.whisper`, because legacy bundles predate the field.
    /// Worth pinning: an *unknown* value also degrades rather than throwing, since the synthesized
    /// `Decodable` rejects the raw value and `try?` swallows the whole payload.
    @Test(
        "Anything unrecognized degrades to whisper",
        arguments: [
            #"{"kind": "speech_recognizer"}"#,
            #"{"config": {"vocab_size": 1024}}"#,
            #"{"config": {"architecture": "conformer_ctc"}}"#,
            #"{"#,
            "",
        ])
    func unrecognizedDegradesToWhisper(json: String) {
        #expect(architecture(json) == .whisper)
    }
}

// MARK: - ParakeetTDTConfig

@Suite("ParakeetTDTConfig decoding")
struct ParakeetTDTConfigTests {
    private static let validJSON = """
        {"config": {
            "vocab_size": 1025, "blank_token_id": 1024, "decoder_hidden_size": 640,
            "num_decoder_layers": 2, "max_symbols_per_step": 10, "durations": [0, 1, 2, 3, 4],
            "encoder": {"num_mel_bins": 128, "subsampling_factor": 8}
        }}
        """

    @Test("A full config block decodes with snake_case keys")
    func fullConfigDecodes() throws {
        let c = try ParakeetTDTConfig.decode(fromMetadata: Data(Self.validJSON.utf8))
        #expect(c.vocabSize == 1_025)
        #expect(c.blankTokenId == 1_024)
        #expect(c.decoderHiddenSize == 640)
        #expect(c.numDecoderLayers == 2)
        #expect(c.maxSymbolsPerStep == 10)
        #expect(c.durations == [0, 1, 2, 3, 4])
        #expect(c.encoderNumMelBins == 128)
        #expect(c.encoderSubsamplingFactor == 8)
    }

    @Test("A missing config block reports the missing field")
    func missingConfigBlockThrows() {
        #expect(throws: (any Error).self) {
            try ParakeetTDTConfig.decode(fromMetadata: Data("{}".utf8))
        }
        do {
            _ = try ParakeetTDTConfig.decode(fromMetadata: Data("{}".utf8))
            Issue.record("expected a throw")
        } catch {
            #expect(String(describing: error).contains("config"))
        }
    }

    @Test("Every field in the config block is required")
    func fieldsAreRequired() throws {
        // Drop one key at a time from the valid document; each omission must throw rather than
        // silently defaulting, since a wrong vocab size or duration list corrupts decoding.
        for key in [
            "vocab_size", "blank_token_id", "decoder_hidden_size", "num_decoder_layers",
            "max_symbols_per_step", "durations", "encoder",
        ] {
            var object =
                try #require(
                    JSONSerialization.jsonObject(with: Data(Self.validJSON.utf8))
                        as? [String: Any])
            var config = try #require(object["config"] as? [String: Any])
            config.removeValue(forKey: key)
            object["config"] = config
            let data = try JSONSerialization.data(withJSONObject: object)
            #expect(throws: (any Error).self, "omitting \(key) should throw") {
                try ParakeetTDTConfig.decode(fromMetadata: data)
            }
        }
    }

    @Test("Extra keys are ignored")
    func extraKeysIgnored() throws {
        var object =
            try #require(
                JSONSerialization.jsonObject(with: Data(Self.validJSON.utf8)) as? [String: Any])
        var config = try #require(object["config"] as? [String: Any])
        config["architecture"] = "parakeet_tdt"
        object["config"] = config
        object["metadata_version"] = "0.2"
        let c = try ParakeetTDTConfig.decode(
            fromMetadata: try JSONSerialization.data(withJSONObject: object))
        #expect(c.vocabSize == 1_025)
    }
}

// MARK: - SpeechError

@Suite("SpeechError")
struct SpeechErrorTests {
    @Test("Descriptions embed their payload")
    func descriptionsEmbedPayload() {
        // Substring assertions only — verbatim message equality would break on any rewording.
        #expect(SpeechError.missingModel("encoder").description.contains("encoder"))
        #expect(SpeechError.invalidAudio("bad rate").description.contains("bad rate"))
        #expect(SpeechError.incompatibleResources("mismatch").description.contains("mismatch"))
        #expect(SpeechError.missingTokenizer.description.lowercased().contains("tokenizer"))
    }
}
