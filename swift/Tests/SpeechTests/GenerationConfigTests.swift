// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

@Suite("GenerationConfig")
struct GenerationConfigTests {
    @Test("Parses forced_decoder_ids from HuggingFace format")
    func parseForcedDecoderIds() throws {
        let json: [String: Any] = [
            "forced_decoder_ids": [[1, 50258], [2, 50259], [3, 50360], [4, 50364]],
            "eos_token_id": 50257,
            "max_new_tokens": 100,
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gen_config.json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let config = try GenerationConfig(from: url)
        #expect(config.forcedPrefix == [50258, 50259, 50360, 50364])
        #expect(config.eotToken == 50257)
        #expect(config.maxDecodeSteps == 100)
    }

    @Test("Falls back to Whisper defaults when forced_decoder_ids is missing")
    func fallbackOnMissingField() throws {
        let json: [String: Any] = ["eos_token_id": 50257]
        let data = try JSONSerialization.data(withJSONObject: json)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gen_config2.json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let config = try GenerationConfig(from: url)
        #expect(config.forcedPrefix == GenerationConfig.whisper.forcedPrefix)
    }

    @Test("Falls back to Whisper defaults when forced_decoder_ids has wrong format")
    func fallbackOnWrongFormat() throws {
        let json: [String: Any] = [
            "forced_decoder_ids": [50258, 50259, 50360]  // flat array, not pairs
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gen_config3.json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let config = try GenerationConfig(from: url)
        #expect(config.forcedPrefix == GenerationConfig.whisper.forcedPrefix)
    }

    @Test("Skips null token IDs in forced_decoder_ids pairs")
    func skipsNullTokenIds() throws {
        let json: [String: Any] = [
            "forced_decoder_ids": [[1, NSNull()], [2, 50360], [3, 50364]]
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gen_config4.json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let config = try GenerationConfig(from: url)
        #expect(config.forcedPrefix == [50360, 50364])
    }

    @Test("Handles out-of-order positions")
    func outOfOrderPositions() throws {
        let json: [String: Any] = [
            "forced_decoder_ids": [[3, 50360], [1, 50258], [2, 50259]]
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gen_config5.json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let config = try GenerationConfig(from: url)
        #expect(config.forcedPrefix == [50258, 50259, 50360])
    }
}
