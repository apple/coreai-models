// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// A tokenizer directory with one token per byte, so decoding what it encodes gives back the exact
/// text: tests can read a rendered chat template back from its tokens.
public enum ByteLevelTokenizerFixture {
    /// Writes `tokenizer.json` and `tokenizer_config.json` (with `chatTemplate`) to a new temporary
    /// directory and returns it. The caller removes it.
    ///
    /// `additionalSpecialTokens` get the ids after the byte tokens, from `firstAdditionalTokenId` on.
    public static func directory(chatTemplate: String, additionalSpecialTokens: [String] = []) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("byte-level-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let special = ["<pad>", "<unk>", "<eos>", "<bos>"]
        var vocab: [String: Int] = [:]
        for (id, token) in special.enumerated() { vocab[token] = id }
        for (offset, character) in byteCharacters.enumerated() { vocab[String(character)] = special.count + offset }
        for (offset, token) in additionalSpecialTokens.enumerated() { vocab[token] = firstAdditionalTokenId + offset }
        let byteLevel: [String: Any] = [
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false,
        ]
        let tokenizer: [String: Any] = [
            "version": "1.0",
            "added_tokens": (special + additionalSpecialTokens).map { content in
                [
                    "id": vocab[content]!, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
                    "normalized": false, "special": true,
                ] as [String: Any]
            },
            "pre_tokenizer": byteLevel,
            "decoder": byteLevel,
            "model": [
                "type": "BPE", "unk_token": "<unk>", "continuing_subword_prefix": "", "end_of_word_suffix": "",
                "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": [] as [String],
            ] as [String: Any],
        ]
        let config: [String: Any] = [
            "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<bos>", "eos_token": "<eos>",
            "unk_token": "<unk>", "pad_token": "<pad>", "model_max_length": 1_000_000,
            "chat_template": chatTemplate,
        ]
        try JSONSerialization.data(withJSONObject: tokenizer)
            .write(to: directory.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        return directory
    }

    /// Id of the first of `directory`'s `additionalSpecialTokens`.
    public static let firstAdditionalTokenId = 4 + 256

    /// The character byte-level BPE stands each byte for (GPT-2's `bytes_to_unicode`).
    private static let byteCharacters: [Character] = {
        // Printable bytes stand for themselves; the rest map, in order, to scalars from 256 up.
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var scalars = bytes
        var next = 256
        for byte in 0..<256 where !bytes.contains(byte) {
            bytes.append(byte)
            scalars.append(next)
            next += 1
        }
        let byScalar = Dictionary(uniqueKeysWithValues: zip(bytes, scalars))
        return (0..<256).map { Character(Unicode.Scalar(UInt32(byScalar[$0]!))!) }
    }()
}
