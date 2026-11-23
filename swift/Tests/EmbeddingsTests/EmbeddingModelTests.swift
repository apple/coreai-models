// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Testing

@testable import CoreAIEmbeddings

@Suite struct EmbeddingModelTests {
    @Test func clipTextBatchesAndOrder() async throws {
        let log = CallLog()
        var model = try FakeModels.model("clip-vit-b32", log: log) { _, _, b, inputs in
            // The row's first word id, so the output keeps the input order visible.
            let ids = inputs["input_ids"]!.ints!
            return [Float(ids[b * 77 + 1])] + [Float](repeating: 0, count: 511)
        }
        let texts = (0..<70).map { String(repeating: "x", count: $0 % 5 + 1) }
        let vectors = try await model.embed(texts: texts)
        #expect(vectors.map { $0.vector[0] } == texts.map { Float($0.count + 10) })
        let calls = await log.calls(of: "text_encoder")
        #expect(calls.map { $0.inputs["input_ids"]!.shape } == [[64, 77], [6, 77]])
        let first = calls[0].inputs
        #expect(Array(first["input_ids"]!.ints!.prefix(4)) == [49406, 11, 49407, 49407])
        #expect(Array(first["attention_mask"]!.ints!.prefix(4)) == [1, 1, 1, 0])
        #expect(vectors.allSatisfy { $0.space.id == "openai/clip-vit-base-patch32" })
        let logitScale = model.logitScale
        #expect(abs((logitScale ?? 0) - 100) < 0.01)
    }

    @Test func clipImageCrops() async throws {
        let log = CallLog()
        var model = try FakeModels.model("clip-vit-b32", log: log)
        // Red in the middle, green in the 40 columns at each side: 300x200 resizes to 336x224 and
        // the center crop starts at column 56, past the green.
        let image = try Fixtures.image(width: 300, height: 200) { x, _ in x < 40 || x >= 260 ? (0, 255, 0) : (255, 0, 0)
        }
        _ = try await model.embed(images: [image, image])
        let input = try #require(await log.calls(of: "image_encoder").first?.inputs["pixel_values"])
        #expect(input.shape == [2, 3, 224, 224])
        let r = (1 - 0.48145466) / Float(0.26862954)
        #expect(abs(input.floats[0] - r) < 1e-5 && abs(input.floats[223] - r) < 1e-5)
        #expect(abs(input.floats[224 * 224] - (0 - 0.4578275) / 0.26130258) < 1e-5)
    }

    @Test func embeddingGemma2NeuralEngineImage() async throws {
        let log = CallLog()
        var model = try FakeModels.model("embeddinggemma-2-ios", log: log) { component, output, _, _ in
            output == "soft_tokens" ? (0..<(280 * 512)).map { Float($0 / 512 + 1) } : nil
        }
        let image = try Fixtures.image(width: 64, height: 64) { x, y in (UInt8(x), UInt8(y), 0) }
        _ = try await model.embed(images: [image])
        let pixels = try #require(await log.calls(of: "image_encoder").first?.inputs)
        #expect(pixels["pixel_values"]?.shape == [1, 2520, 768])
        // Patch 1 is the second one along x, so its red (x) channel outweighs its green (y) one.
        let patch = try #require(pixels["pixel_values"]?.floats[768..<1536])
        let channel = { (c: Int) in stride(from: 768 + c, to: 1536, by: 3).map { patch[$0] }.reduce(0, +) }
        #expect(channel(0) > channel(1) + 0.5)
        // 64x64 resizes to 768x768: 48x48 patches, 256 soft tokens.
        let positions = try #require(pixels["position_ids"]?.ints)
        #expect(Array(positions.prefix(4)) == [0, 0, 1, 0])
        #expect(positions[2304 * 2] == -1)
        let table = try #require(await log.calls(of: "token_table").first)
        let ids = try #require(table.inputs["input_ids"]?.ints)
        #expect(ids.count == 260 && ids.prefix(2) == [2, 255999] && ids.suffix(2) == [258882, 1])
        let encoder = try #require(await log.calls(of: "encoder").first)
        #expect(encoder.function == "embeds_len512_b1")
        let embeds = try #require(encoder.inputs["inputs_embeds"]?.floats)
        #expect(embeds[2 * 512] == 1 && embeds[257 * 512] == 256)
        #expect(embeds[1 * 512] == 1 && embeds[258 * 512] == 1)
        #expect(encoder.inputs["bias"]?.shape == [1, 512, 1, 512])
        let pool = try #require(encoder.inputs["pool_weights"]?.floats)
        #expect(abs(pool.reduce(0, +) - 1) < 1e-5 && pool[260] == 0)
    }

    @Test func embeddingGemma2GPUImageUsesSharedAsset() async throws {
        let log = CallLog()
        var model = try FakeModels.model("embeddinggemma-2", log: log)
        let image = try Fixtures.image(width: 64, height: 64) { _, _ in (9, 9, 9) }
        _ = try await model.embed(images: [image])
        let calls = await log.calls
        #expect(calls.map(\.component) == ["image_encoder", "token_table", "encoder"])
        #expect(calls[2].function == "embeds_len288_b1")
        let mask = try #require(calls[2].inputs["attention_mask"]?.ints)
        #expect(mask.prefix(260).allSatisfy { $0 == 1 } && mask.dropFirst(260).allSatisfy { $0 == 0 })
        // One load per asset: the text tower reuses the program of `token_table` and `encoder`.
        _ = try await model.embed(images: [image])
        _ = try await model.embed(texts: ["a"])
        #expect(await log.loads == 2)
    }

    @Test func widthsAndCosine() async throws {
        var model = try FakeModels.model("embeddinggemma-2") { _, _, _, _ in [3, 4] + [Float](repeating: 0, count: 766)
        }
        let full = try await model.embed(texts: ["a"])[0]
        let small = try await model.embed(texts: ["a"], dimensions: 128)[0]
        #expect(small.dimension == 128 && abs(small.vector[0] - 0.6) < 1e-6 && abs(small.vector[1] - 0.8) < 1e-6)
        await #expect(throws: EmbeddingError.self) { try await model.embed(texts: ["a"], dimensions: 100) }

        #expect(try full.cosine(full) > 0.9999)
        #expect(throws: EmbeddingError.self) { try full.cosine(small) }
        let other = Embedding(vector: full.vector, space: EmbeddingSpace(id: "other/model", revision: "abc"))
        #expect(throws: EmbeddingError.self) { try full.cosine(other) }
    }

    @Test func mixedInputsKeepTheirOrder() async throws {
        let log = CallLog()
        var model = try FakeModels.model("clip-vit-b32", log: log) { component, _, b, _ in
            [component == "text_encoder" ? Float(b + 1) : Float(-(b + 1))] + [Float](repeating: 0, count: 511)
        }
        let image = try Fixtures.image(width: 8, height: 8) { _, _ in (0, 0, 0) }
        let vectors = try await model.embed([.image(image), .text("a"), .image(image), .text("b c")])
        #expect(vectors.map { $0.vector[0] } == [-1, 1, -2, 2])
        #expect(await log.calls.map(\.component) == ["text_encoder", "image_encoder"])
    }

    @Test func truncationKeepsEosAndSetsStatus() async throws {
        let log = CallLog()
        let block = try Fixtures.block("clip-vit-b32")
        let text = try await FakeModels.clipText(block.text)
        var model = try FakeModels.model("clip-vit-b32", log: log, text: text)
        let long = Array(repeating: "w", count: 100).joined(separator: " ")
        let image = try Fixtures.image(width: 8, height: 8) { _, _ in (0, 0, 0) }
        let vectors = try await model.embed([.text(long), .text("a b"), .image(image)])
        #expect(vectors.map(\.status) == [.truncated(tokens: 102, limit: 77), .complete, .complete])
        // The cut and the in-limit ids are those of `CLIPTokenizer.encodeWithMask`.
        let clip = try CLIPTokenizer(vocab: FakeModels.clipVocab, merges: [])
        let inputs = try #require(await log.calls(of: "text_encoder").first?.inputs)
        let ids = try #require(inputs["input_ids"]?.ints)
        #expect(Array(ids[0..<77]) == clip.encodeWithMask(long).ids)
        #expect(ids.count == 2 * 77 && ids.last == 49407)
        let short = clip.encodeWithMask("a b")
        #expect(text.tokenize("a b").ids == Array(short.ids.prefix(Int(short.attentionMask.reduce(0, +)))))
        // Decomposed `é` (e + U+0301) is composed first, as Hugging Face's NFC normalizer does.
        #expect(text.tokenize("cafe\u{301}").ids == [49406, 15304, 49407])
        #expect(inputs["attention_mask"]?.ints?.prefix(77).reduce(0, +) == 77)

        var gemma = try FakeModels.model("embeddinggemma-2") { _, _, _, _ in [3, 4] + [Float](repeating: 0, count: 766)
        }
        let cut = try await gemma.embed(texts: [Array(repeating: "w", count: 3000).joined(separator: " ")])[0]
        #expect(cut.status == .truncated(tokens: 3002, limit: 2048))
        #expect(try cut.cosine(Embedding(vector: cut.vector, space: cut.space)) > 0.9999)
    }

    /// The audio tower is skipped until audio input lands; the text tower runs.
    @Test func clapRunsText() async throws {
        let clap: (inout [String: Any]) -> Void = {
            $0["towers"] = [
                ["modality": "text", "pipeline": ["text_encoder"]],
                ["modality": "audio", "pipeline": ["audio_encoder"]],
            ]
            $0["image"] = nil
            $0.edit("components") {
                $0["image_encoder"] = nil
                $0["audio_encoder"] = [
                    "asset": "audio_encoder", "inputs": ["input_features": "float32[B,1,1001,64]"],
                    "outputs": ["embedding": "float32[B,512]"], "compute_units": ["gpu"],
                    "functions": [["function": "audio_encoder", "batch": ["min": 1, "max": 64]]],
                ]
            }
            $0["audio"] = [
                "sample_rate": 48000, "n_fft": 1024, "win_length": 1024, "hop_length": 480, "n_mel_bins": 64,
                "n_frames": 1001, "f_min": 50, "f_max": 14000,
            ]
            $0.edit("text") {
                $0["bos_token_id"] = 0
                $0["eos_token_id"] = 2
                $0["pad_token_id"] = 1
            }
        }
        #expect(try Fixtures.block("clip-vit-b32", editing: clap).towers.map(\.modality) == [.text])
        var model = try FakeModels.model("clip-vit-b32", editing: clap)
        #expect(model.modalities == [.text])
        #expect(try await model.embed(texts: ["a dog"]).count == 1)
    }

    @Test func textWithoutBos() throws {
        let block = try Fixtures.block("clip-vit-b32") {
            $0.edit("text") {
                $0["bos_token_id"] = NSNull()
                $0["add_bos_token"] = false
            }
        }
        #expect(block.text.bosTokenId == nil && !block.text.addBosToken)
    }

    @Test func refusesBadTextBlock() {
        let cases: [(path: String, key: String, value: Any, message: String)] = [
            ("text.truncation", "max_tokens", 78, "exceeds text_encoder's longest function, 77 tokens"),
            ("text.truncation", "max_tokens", 0, "max_tokens must be at least 1"),
            ("text", "default_prompt_name", "missing", "default_prompt_name 'missing' is not in prompts"),
        ]
        for c in cases {
            let error = #expect(throws: EmbeddingError.self) {
                try Fixtures.block("clip-vit-b32") { $0.edit(c.path) { $0[c.key] = c.value } }
            }
            #expect(error?.description.contains(c.message) == true, "\(c.key) = \(c.value)")
        }
    }

    @Test func defaultPromptIsPrepended() throws {
        let block = try Fixtures.block("embeddinggemma-2") { $0.edit("text") { $0["default_prompt_name"] = "query" } }
        #expect(block.text.prompt == "task: search result | query: ")
        let text = FakeModels.text(block.text)
        #expect(text.tokenize("cats").ids == text.encode("task: search result | query: cats").map(Int32.init))
    }

    @Test func missingTowerAndNonFiniteOutput() async throws {
        var text = try FakeModels.model("clip-vit-b32", editing: Fixtures.clipTextOnly)
        let image = try Fixtures.image(width: 8, height: 8) { _, _ in (0, 0, 0) }
        let error = await #expect(throws: EmbeddingError.self) { try await text.embed(images: [image]) }
        #expect(error?.description == "the bundle has no image tower")
        var broken = try FakeModels.model("clip-vit-b32") { _, _, _, _ in [.nan] }
        await #expect(throws: EmbeddingError.self) { try await broken.embed(texts: ["a"]) }
    }
}

@Suite struct QueuedEmbeddingModelTests {
    @Test(.timeLimit(.minutes(1))) func servesCallersOneAtATimeInOrder() async throws {
        let log = CallLog()
        let queued = try QueuedEmbeddingModel(
            FakeModels.model("clip-vit-b32", log: log) { _, _, b, inputs in
                [Float(inputs["input_ids"]!.ints![b * 77 + 1])] + [Float](repeating: 0, count: 511)
            })
        await log.hold()
        var calls: [Task<[Embedding], Error>] = []
        for words in 1...4 {
            calls.append(Task { try await queued.embed(texts: [String(repeating: "x", count: words)]) })
            // The first call runs and is held; each later one waits behind the ones before it.
            while await log.calls.isEmpty || queued.waitingCount < words - 1 { await Task.yield() }
        }
        await log.release()
        for (words, call) in calls.enumerated() {
            #expect(try await call.value.map { $0.vector[0] } == [Float(words + 11)])
        }
        #expect(await log.calls.map { $0.inputs["input_ids"]!.ints![1] } == [11, 12, 13, 14])
        #expect(await log.mostRunning == 1)

        // A call that throws puts the model back.
        await #expect(throws: EmbeddingError.self) { try await queued.embed(texts: ["a"], dimensions: 100) }
        #expect(try await queued.embed(texts: ["a"]).count == 1)
    }

    /// A cancelled call throws `CancellationError` before any program call, whether it was waiting
    /// behind another call or cancelled before it started; the other calls still run.
    @Test(.timeLimit(.minutes(1))) func cancellation() async throws {
        let log = CallLog()
        let queued = try QueuedEmbeddingModel(FakeModels.model("clip-vit-b32", log: log))
        await log.hold()
        let running = Task { try await queued.embed(texts: ["a"]) }
        while await log.calls.isEmpty { await Task.yield() }
        let waiting = Task { try await queued.embed(texts: ["b"]) }
        while queued.waitingCount < 1 { await Task.yield() }
        waiting.cancel()
        await #expect(throws: CancellationError.self) { try await waiting.value }
        await log.release()
        #expect(try await running.value.count == 1)

        let calls = await log.calls.count
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await queued.embed(texts: ["c"])
        }
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        #expect(await log.calls.count == calls)
    }
}

@Suite struct CoreAIProgramTests {
    @Test func fillConvertsToTheArrayType() throws {
        var half = NDArray(shape: [2], scalarType: .float16)
        try CoreAIProgram.fill(&half, with: HostTensor(float32: [1.5, -2], shape: [2]), name: "x")
        #expect(readNDArray(half, as: Float16.self, count: 2) == [1.5, -2])
        var float = NDArray(shape: [2], scalarType: .float32)
        try CoreAIProgram.fill(&float, with: HostTensor(int32: [3, -4], shape: [2]), name: "x")
        #expect(readNDArray(float, as: Float.self, count: 2) == [3, -4])
        var int = NDArray(shape: [2], scalarType: .int32)
        #expect(throws: EmbeddingError.self) {
            try CoreAIProgram.fill(&int, with: HostTensor(float32: [1, 2], shape: [2]), name: "x")
        }
    }
}
