// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

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
