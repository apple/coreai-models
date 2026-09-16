// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIShared
@testable import CoreAIVideoSegmenter

@Suite("CLIPTokenizer attention mask")
struct TokenizerTests {
    /// A vocabulary of whole single-letter words, with the ids real CLIP gives them.
    ///
    /// CLIP's BPE appends `</w>` to the last character of a token and then merges pairs;
    /// with no merge table a one-letter word is already terminal, so it looks up directly.
    /// That keeps this suite self-contained: it tests the masking rule, not BPE.
    private func tokenizer() throws -> CLIPTokenizer {
        try CLIPTokenizer(
            vocab: [
                "<|startoftext|>": 49406,
                "<|endoftext|>": 49407,
                "a</w>": 320,
                "b</w>": 736,
                "c</w>": 1615,
            ],
            merges: [])
    }

    @Test("A one-word prompt is SOT, the word, EOT, then padding")
    func singleWord() throws {
        let (ids, mask) = try tokenizer().encodeWithMask("a", contextLength: 32)
        #expect(ids.prefix(4) == [49406, 320, 49407, 49407])
        #expect(ids.count == 32)
        #expect(mask.prefix(4) == [1, 1, 1, 0])
        #expect(mask.reduce(0, +) == 3)
    }

    @Test("The mask distinguishes a real EOT from the pad tokens after it")
    func maskMarksRealTokens() throws {
        // The pad token *is* `<|endoftext|>`, so the ids alone cannot tell the closing EOT
        // from padding. This is the whole reason `encodeWithMask` exists.
        let (ids, mask) = try tokenizer().encodeWithMask("a b c", contextLength: 32)
        #expect(ids.prefix(5) == [49406, 320, 736, 1615, 49407])
        #expect(mask.prefix(6) == [1, 1, 1, 1, 1, 0])
        #expect(mask.reduce(0, +) == 5)
        #expect(ids[4] == ids[5], "the real EOT and the first pad share an id")
        #expect(mask[4] != mask[5], "but not a mask value")
    }

    @Test("Case and surrounding whitespace are normalized away")
    func normalization() throws {
        let plain = try tokenizer().encodeWithMask("a b", contextLength: 32)
        let noisy = try tokenizer().encodeWithMask("  A   B  ", contextLength: 32)
        #expect(plain.ids == noisy.ids)
        #expect(plain.attentionMask == noisy.attentionMask)
    }

    @Test("An over-long prompt is truncated with EOT forced into the last slot")
    func truncation() throws {
        // HF would emit a longer row and warn; the graph is traced at a fixed length, so
        // truncating is the only option that runs. The mask goes all-ones because every
        // slot then holds a real token.
        let (ids, mask) = try tokenizer().encodeWithMask("a b c a b c", contextLength: 5)
        #expect(ids.count == 5)
        #expect(ids.last == 49407)
        #expect(mask == [1, 1, 1, 1, 1])
    }

    @Test("encode agrees with encodeWithMask on the ids")
    func encodeMatches() throws {
        let tokenizer = try tokenizer()
        // Both share one implementation, so this pins the padded, exact-fit and truncated
        // lengths together rather than only the common case.
        for contextLength in [32, 5, 3] {
            #expect(
                tokenizer.encode("a b c", contextLength: contextLength)
                    == tokenizer.encodeWithMask("a b c", contextLength: contextLength).ids,
                "contextLength \(contextLength)")
        }
    }

    @Test("A non-positive context length yields nothing instead of trapping")
    func emptyContext() throws {
        let tokenizer = try tokenizer()
        #expect(tokenizer.encode("a b c", contextLength: 0).isEmpty)
        let (ids, mask) = tokenizer.encodeWithMask("a b c", contextLength: -1)
        #expect(ids.isEmpty)
        #expect(mask.isEmpty)
    }
}

@Suite("CLIPTokenizer BPE")
struct TokenizerBPETests {
    /// A toy vocabulary whose merge table actually merges.
    ///
    /// The suite above passes `merges: []` and single-letter words, so `bpe` returns at its
    /// `chars.count == 1` early exit and the merge loop never runs. These fixtures are the
    /// smallest thing that exercises it.
    private enum Toy {
        /// Merge rank is list position; the lowest-ranked matching pair merges first.
        static let merges: [(String, String)] = [
            ("b", "c</w>"),  // rank 0
            ("a", "b"),  // rank 1
            ("a", "t</w>"),  // rank 2
            ("c", "at</w>"),  // rank 3
        ]

        /// Holds an id for both outcomes of the "abc" race below, so a wrong merge order
        /// shows up as different ids rather than as tokens silently dropped by `compactMap`.
        static let vocab: [String: Int32] = [
            "<|startoftext|>": 49406,
            "<|endoftext|>": 49407,
            "a": 100,
            "ab": 101,
            "c</w>": 102,
            "bc</w>": 103,
            "cat</w>": 200,
        ]

        static func tokenizer() throws -> CLIPTokenizer {
            try CLIPTokenizer(vocab: vocab, merges: merges)
        }
    }

    @Test("A word whose merges chain all the way collapses to a single token")
    func mergesToSingleToken() throws {
        // `["c", "a", "t</w>"]`, rank 2 gives `["c", "at</w>"]`, rank 3 gives `["cat</w>"]`.
        // This is the shape "person" has in the real vocabulary: one id, not six.
        let (ids, mask) = try Toy.tokenizer().encodeWithMask("cat", contextLength: 8)
        #expect(ids.prefix(3) == [49406, 200, 49407])
        #expect(mask.reduce(0, +) == 3)
    }

    @Test("The lowest-rank pair merges first, even when a later pair also matches")
    func mergeRankOrder() throws {
        // "abc" is `["a", "b", "c</w>"]` and both ("a","b") and ("b","c</w>") match. Rank 0
        // wins, giving `["a", "bc</w>"]` = [100, 103]; merging ("a","b") first would give
        // `["ab", "c</w>"]` = [101, 102].
        let ids = try Toy.tokenizer().encode("abc", contextLength: 8)
        #expect(ids.prefix(4) == [49406, 100, 103, 49407])
    }

    @Test("A tokenizer.json on disk loads to the same tokenizer as the in-memory init")
    func loadsFromFolder() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        // A one-element entry among the merges: `init(folder:)` skips malformed pairs, and
        // skipping must not shift the ranks of the valid ones that follow.
        let merges: [[String]] = [["junk"]] + Toy.merges.map { [$0.0, $0.1] }
        let json = try JSONSerialization.data(
            withJSONObject: [
                "model": ["vocab": Toy.vocab.mapValues(Int.init), "merges": merges]
            ])
        try json.write(to: folder.appending(path: "tokenizer.json"))

        let loaded = try CLIPTokenizer(folder: folder)
        let expected = try Toy.tokenizer()
        for prompt in ["cat", "abc", "c", "a cat"] {
            #expect(
                loaded.encode(prompt, contextLength: 8) == expected.encode(prompt, contextLength: 8),
                "\(prompt)")
        }
    }
}

@Suite("Temporal position index")
struct TemporalIndexTests {
    @Test("A conditioning frame maps to the last row, not to -1")
    func conditioningWraps() {
        // HF writes `memory_temporal_positional_encoding[offset - 1]`, and a conditioning
        // frame's offset is 0, which in Python is the last row. Getting this wrong gives
        // every conditioning memory the wrong temporal encoding, which the graph cannot flag
        // because the index is still in range.
        #expect(MemoryBankPacker.temporalIndex(forOffset: 0, numMaskmem: 7) == 6)
    }

    @Test("Recent frames map to offset minus one")
    func recentFrames() {
        #expect(MemoryBankPacker.temporalIndex(forOffset: 1, numMaskmem: 7) == 0)
        #expect(MemoryBankPacker.temporalIndex(forOffset: 6, numMaskmem: 7) == 5)
    }
}

@Suite("VideoSegmenterBundle metadata")
struct BundleMetadataTests {
    /// A minimal `video_segmenter` metadata document, plus whatever extra top-level blocks
    /// the caller needs.
    private func bundle(
        imageSize: Int = 1008, extraBlocks: String = ""
    ) throws -> VideoSegmenterBundle {
        try bundle(
            """
            {
              "metadata_version": "0.2",
              "kind": "video_segmenter",
              "name": "sam3_video_float16",
              "assets": {"main": "sam3_video_float16.aimodel"},
              "runtime": {
                "image_size": \(imageSize), "spatial_slots": 10,
                "ptr_slots": 24, "max_text_seq_len": 32
              }\(extraBlocks)
            }
            """)
    }

    private func bundle(_ json: String) throws -> VideoSegmenterBundle {
        try VideoSegmenterBundle(
            bundle: ModelBundle(
                raw: Data(json.utf8),
                bundlePath: URL(fileURLWithPath: "/tmp/does-not-need-to-exist")))
    }

    @Test("The runtime block is parsed")
    func geometry() throws {
        let parsed = try bundle()
        #expect(parsed.geometry.imageSize == 1008)
        #expect(parsed.geometry.spatialSlots == 10)
        #expect(parsed.geometry.ptrSlots == 24)
        #expect(parsed.geometry.maxTextSeqLen == 32)
    }

    @Test("A bundle without a tracking block keeps the upstream defaults")
    func defaultsWithoutTracking() throws {
        // The shipped export predates the tracking block, so this is the path it takes.
        let parameters = try bundle().parameters()
        #expect(parameters.hotstartDelay == 15)
        #expect(parameters.scoreThresholdDetection == 0.5)
        #expect(parameters.numMaskmem == 7)
        #expect(parameters.maxCondFrameNum == 4)
    }

    @Test("A partial tracking block overrides only what it names")
    func partialOverride() throws {
        let parsed = try bundle(
            imageSize: 336,
            extraBlocks: """
                ,
                  "tracking": {"hotstart_delay": 0, "score_threshold_detection": 0.25}
                """)
        let parameters = try parsed.parameters()
        #expect(parameters.hotstartDelay == 0)
        #expect(parameters.scoreThresholdDetection == 0.25)
        #expect(parameters.newDetThresh == 0.7, "untouched keys keep their default")
        #expect(parsed.geometry.imageSize == 336)
    }

    @Test("Caller overrides apply underneath the bundle's own values")
    func bundleWinsOverCaller() throws {
        var caller = VideoSegmentationParameters.default
        caller.hotstartDelay = 99
        caller.maxNumObjects = 5

        let parameters = try bundle(extraBlocks: #","tracking": {"hotstart_delay": 3}"#)
            .parameters(overriding: caller)
        #expect(parameters.hotstartDelay == 3, "the bundle knows its own checkpoint")
        #expect(
            parameters.maxNumObjects == 5,
            "the caller's value survives where the bundle is silent")
    }

    @Test("A missing runtime block is rejected with an actionable message")
    func missingRuntime() throws {
        #expect(throws: VideoSegmentationError.self) {
            try bundle(
                """
                {"metadata_version": "0.2", "kind": "video_segmenter", "name": "x",
                 "assets": {"main": "x.aimodel"}}
                """)
        }
    }

    @Test("A malformed tracking block is reported rather than replaced by defaults")
    func malformedTracking() throws {
        let parsed = try bundle(extraBlocks: #","tracking": {"hotstart_delay": "soon"}"#)
        #expect(throws: VideoSegmentationError.self) {
            try parsed.parameters()
        }
    }

    @Test("A segmenter bundle is not accepted as a video segmenter")
    func kindMismatch() throws {
        #expect(throws: ModelBundle.BundleError.self) {
            try bundle(
                """
                {"metadata_version": "0.2", "kind": "segmenter", "name": "x",
                 "assets": {"main": "x.aimodel"},
                 "runtime": {"image_size": 1008, "spatial_slots": 10, "ptr_slots": 24,
                             "max_text_seq_len": 32}}
                """)
        }
    }
}
