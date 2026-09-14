// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// Covers the shared `runChunkedPrefill` loop that both sequential engines call: how a prompt
/// is split into per-chunk `processChunk` calls, which chunks are marked held-back, and which
/// chunk's logits are returned. Uses a recording closure so the loop can be exercised without a
/// live graph.
@Suite("Chunked Prefill")
struct ChunkedPrefillTests {
    /// Records the token slices and held-back flags `runChunkedPrefill` hands to `processChunk`.
    private final class Recorder {
        var calls: [(tokens: [Int32], heldBack: Bool)] = []
    }

    /// One logit row per token, filled with the token's value so `lastTokenLogits` can be checked.
    private func rows(for chunk: ArraySlice<Int32>, vocabSize: Int) -> [LogitsScalarType] {
        var out: [LogitsScalarType] = []
        for token in chunk {
            out.append(contentsOf: Array(repeating: LogitsScalarType(Int(token)), count: vocabSize))
        }
        return out
    }

    // MARK: - VLM case (heldBack == 0)

    @Test("With nothing held back, every chunk runs and the last token's logits come back")
    func heldBackZeroProcessesEveryChunk() async throws {
        let tokens: [Int32] = Array(0..<10)
        let vocabSize = 3
        let recorder = Recorder()

        let result = try await runChunkedPrefill(
            tokens: tokens[...],
            chunkSize: 4,
            heldBack: 0,
            vocabSize: vocabSize
        ) { chunk, isHeldBack in
            recorder.calls.append((Array(chunk), isHeldBack))
            return self.rows(for: chunk, vocabSize: vocabSize)
        }

        // Contiguous chunks of width 4 covering all 10 tokens, none held back.
        #expect(recorder.calls.map(\.tokens) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        #expect(recorder.calls.allSatisfy { !$0.heldBack })

        // Last token is 9, so the returned row is [9, 9, 9].
        #expect(result == Array(repeating: LogitsScalarType(9), count: vocabSize))
    }

    @Test("With nothing held back, a single-chunk prompt runs once")
    func heldBackZeroSingleChunk() async throws {
        let tokens: [Int32] = [5, 6]
        let recorder = Recorder()

        let result = try await runChunkedPrefill(
            tokens: tokens[...], chunkSize: 8, heldBack: 0, vocabSize: 2
        ) { chunk, isHeldBack in
            recorder.calls.append((Array(chunk), isHeldBack))
            return self.rows(for: chunk, vocabSize: 2)
        }

        #expect(recorder.calls.map(\.tokens) == [[5, 6]])
        #expect(recorder.calls[0].heldBack == false)
        #expect(result == [LogitsScalarType(6), LogitsScalarType(6)])
    }

    // MARK: - Text case (heldBack > 0)

    @Test("With one held back, earlier chunks fill KV and the tail carries the logits")
    func heldBackOneRoutesTailThroughMain() async throws {
        let tokens: [Int32] = Array(0..<10)
        let vocabSize = 3
        let recorder = Recorder()

        let result = try await runChunkedPrefill(
            tokens: tokens[...],
            chunkSize: 4,
            heldBack: 1,
            vocabSize: vocabSize
        ) { chunk, isHeldBack in
            recorder.calls.append((Array(chunk), isHeldBack))
            // Prefill-graph chunks produce no logits; only the held-back tail does.
            return isHeldBack ? self.rows(for: chunk, vocabSize: vocabSize) : []
        }

        // Nine tokens prefilled as [4, 4, 1] (not held back), then token 9 held back.
        #expect(recorder.calls.map(\.tokens) == [[0, 1, 2, 3], [4, 5, 6, 7], [8], [9]])
        #expect(recorder.calls.map(\.heldBack) == [false, false, false, true])

        // The held-back token 9 supplies the logits.
        #expect(result == Array(repeating: LogitsScalarType(9), count: vocabSize))
    }

    @Test("With one held back and a one-token prompt, nothing is prefilled")
    func heldBackOneSingleToken() async throws {
        let tokens: [Int32] = [7]
        let recorder = Recorder()

        let result = try await runChunkedPrefill(
            tokens: tokens[...], chunkSize: 4, heldBack: 1, vocabSize: 2
        ) { chunk, isHeldBack in
            recorder.calls.append((Array(chunk), isHeldBack))
            return isHeldBack ? self.rows(for: chunk, vocabSize: 2) : []
        }

        // The lone token is the held-back tail; the prefill loop never runs.
        #expect(recorder.calls.map(\.tokens) == [[7]])
        #expect(recorder.calls[0].heldBack == true)
        #expect(result == [LogitsScalarType(7), LogitsScalarType(7)])
    }

    @Test("With one held back, an exact multiple leaves only the tail")
    func heldBackOneExactMultiple() async throws {
        // 9 tokens: 8 prefilled as two width-4 chunks, token 8 held back.
        let tokens: [Int32] = Array(0..<9)
        let recorder = Recorder()

        _ = try await runChunkedPrefill(
            tokens: tokens[...], chunkSize: 4, heldBack: 1, vocabSize: 1
        ) { chunk, isHeldBack in
            recorder.calls.append((Array(chunk), isHeldBack))
            return isHeldBack ? self.rows(for: chunk, vocabSize: 1) : []
        }

        #expect(recorder.calls.map(\.tokens) == [[0, 1, 2, 3], [4, 5, 6, 7], [8]])
        #expect(recorder.calls.map(\.heldBack) == [false, false, true])
    }

    // MARK: - Coverage invariants

    @Test("Chunks are contiguous and cover the whole prompt")
    func chunksCoverPrompt() async throws {
        for heldBack in [0, 1] {
            for count in [1, 2, 7, 8, 9, 16, 17] {
                let tokens: [Int32] = Array(0..<Int32(count))
                let recorder = Recorder()

                _ = try await runChunkedPrefill(
                    tokens: tokens[...], chunkSize: 4, heldBack: heldBack, vocabSize: 1
                ) { chunk, isHeldBack in
                    recorder.calls.append((Array(chunk), isHeldBack))
                    return isHeldBack ? self.rows(for: chunk, vocabSize: 1) : []
                }

                let visited = recorder.calls.flatMap(\.tokens)
                #expect(visited == tokens, "heldBack=\(heldBack) count=\(count)")

                // Exactly the trailing `min(heldBack, count)` tokens are marked held back.
                let heldCalls = recorder.calls.filter(\.heldBack)
                let expectedHeld = Swift.min(heldBack, count)
                #expect(heldCalls.flatMap(\.tokens).count == expectedHeld)
            }
        }
    }
}
