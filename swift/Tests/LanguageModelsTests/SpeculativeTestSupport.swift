// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import TestUtilities
import Tokenizers

@testable import CoreAILanguageModels

// In-memory models and drafters for differential speculative-decoding tests.

/// Small deterministic generator, so a failing case reproduces from its seed.
struct SplitMix: RandomNumberGenerator {
    var state: UInt64
    init(_ seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// A model whose logit row depends on the hash of the whole prefix. `levels` bounds the number
/// of distinct logit values, so a small `levels` makes ties dominate.
struct HashModel {
    var vocab: Int
    var levels: Int
    var salt: UInt64

    func step(_ h: UInt64, _ token: Int32) -> UInt64 {
        var r = SplitMix(h ^ (UInt64(bitPattern: Int64(token)) &* 0x1000_0000_01B3) ^ salt)
        return r.next()
    }

    func row(_ h: UInt64) -> [Float] {
        (0..<vocab).map { v in
            var r = SplitMix(h &+ UInt64(v) &* 7919 &+ salt)
            return Float(Int(r.next() % UInt64(levels)))
        }
    }
}

/// The output type of a test double that never generates through ``InferenceEngine/generate``.
struct EmptyOutputSequence: InferenceOutputSequence {
    typealias Element = InferenceOutput
    typealias Failure = any Error

    private let store = StopReasonStore()
    var stopReason: StopReason? { store.stopReason }
    func setStopReason(_ reason: StopReason) { store.set(reason) }

    struct Iterator: AsyncIteratorProtocol {
        mutating func next() async throws -> InferenceOutput? { nil }
    }
    func makeAsyncIterator() -> Iterator { Iterator() }
}

/// A ``SpeculativeEngine`` over an in-memory model that yields one logit row per position.
///
/// Subclasses keep the per-position state: ``push(_:)`` appends a token and returns the row that
/// predicts the next one, ``truncate(to:)`` drops positions. Greedy verification takes the first
/// maximum (a NaN ranks as negative infinity), top-k verification the `k` largest, highest first
/// and ties by lowest index, as the engines do.
class RowEngine: SpeculativeEngine, @unchecked Sendable {
    struct Config: Codable, InferenceConfiguration {
        var maxContextLength: Int
    }

    let config: Config
    var maxContextLength: Int { config.maxContextLength }
    let alignedBlockSize: Int?
    let verifyTileWidth: Int?
    let maxVerifyBlockLength: Int
    private(set) var committedTokens: [Int32] = []
    /// Simulates an engine with recurrent state: a rewind to part of the committed tokens clears the
    /// cache and leaves the cursor at zero, silently.
    var rewindsOnlyToZero = false
    /// What ``hasRecurrentState`` reports.
    var reportsRecurrentState = false
    var hasRecurrentState: Bool { reportsRecurrentState }

    init(maxContextLength: Int, block: Int?, tile: Int?, maxVerifyBlockLength: Int) {
        self.config = Config(maxContextLength: maxContextLength)
        self.alignedBlockSize = block
        self.verifyTileWidth = tile
        self.maxVerifyBlockLength = maxVerifyBlockLength
    }

    func push(_ token: Int32) -> [Float] { fatalError("subclass") }
    func truncate(to count: Int) { fatalError("subclass") }

    var processedTokenCount: Int { committedTokens.count }

    // The rest of ``InferenceEngine`` is defaulted: this double only drives the speculative primitives.
    func generate(
        with input: [TokenId], samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> EmptyOutputSequence {
        throw InferenceRuntimeError.invalidState("RowEngine implements only the speculative primitives")
    }

    func reset(to tokenIndex: Int) async throws {
        guard tokenIndex >= 0, tokenIndex <= committedTokens.count else {
            throw InferenceRuntimeError.invalidArgument(
                "reset(to: \(tokenIndex)) out of range [0, \(committedTokens.count)]")
        }
        // Nothing to roll back; a reset to zero of an empty cache still clears the state.
        guard tokenIndex < committedTokens.count || tokenIndex == 0 else { return }
        let kept = rewindsOnlyToZero ? 0 : tokenIndex
        truncate(to: kept)
        committedTokens.removeLast(committedTokens.count - kept)
    }

    /// Feeds `tokens` and returns the rows of the last `logitsCount` positions.
    private func run(_ tokens: [Int32], logitsCount: Int) throws -> [[Float]] {
        guard committedTokens.count + tokens.count <= maxContextLength else {
            throw InferenceRuntimeError.invalidState(
                "block of \(tokens.count) at \(committedTokens.count) overflows the \(maxContextLength)-token KV")
        }
        if logitsCount > 0, tokens.count > maxVerifyBlockLength {
            throw InferenceRuntimeError.invalidState(
                "verify block of \(tokens.count) exceeds the widest graph (\(maxVerifyBlockLength))")
        }
        var rows: [[Float]] = []
        for (i, t) in tokens.enumerated() {
            let row = push(t)
            committedTokens.append(t)
            if i >= tokens.count - logitsCount { rows.append(row) }
        }
        return rows
    }

    func advance(_ tokens: [Int32]) async throws {
        guard !tokens.isEmpty else { return }
        _ = try run(tokens, logitsCount: 0)
    }

    func forwardPerPositionArgmax(_ tokens: [Int32]) async throws -> [Int32] {
        try run(tokens, logitsCount: tokens.count).map(argmax)
    }

    func forwardWithPerPositionLogits(_ tokens: [Int32]) async throws -> [[LogitsScalarType]] {
        try run(tokens, logitsCount: tokens.count).map { $0.map { LogitsScalarType($0) } }
    }

    func forwardProposalGreedy(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (argmax: [Int32], features: DrafterFeatures?) {
        let tokens = [anchor] + proposal.tokens
        return (try run(tokens, logitsCount: tokens.count).map(argmax), nil)
    }

    func forwardProposal(
        anchor: Int32, proposal: DraftProposal
    ) async throws -> (logits: [[LogitsScalarType]], features: DrafterFeatures?) {
        let tokens = [anchor] + proposal.tokens
        return (try run(tokens, logitsCount: tokens.count).map { $0.map { LogitsScalarType($0) } }, nil)
    }

    /// Room left in the aligned block after the anchor at the next position.
    var oneForwardDraftBudget: Int {
        guard let block = alignedBlockSize, block > 0 else { return Int.max }
        return block - 1 - (committedTokens.count % block)
    }

    private func argmax(_ row: [Float]) -> Int32 {
        var best = 0
        var bestValue = -Float.infinity
        for i in row.indices where row[i] > bestValue {
            bestValue = row[i]
            best = i
        }
        return Int32(best)
    }
}

/// An engine over a ``HashModel`` with a bounded KV capacity.
///
/// A verify block wider than `maxVerifyBlockLength` throws, as an engine with a fixed widest graph
/// does.
final class HashEngine: RowEngine, @unchecked Sendable {
    let model: HashModel
    private(set) var hashes: [UInt64] = []

    init(
        model: HashModel, maxContextLength: Int = 100_000, block: Int? = nil, tile: Int? = nil,
        maxVerifyBlockLength: Int = Int.max
    ) {
        self.model = model
        super.init(
            maxContextLength: maxContextLength, block: block, tile: tile,
            maxVerifyBlockLength: maxVerifyBlockLength)
    }

    override func push(_ token: Int32) -> [Float] {
        let h = model.step(hashes.last ?? 0, token)
        hashes.append(h)
        return model.row(h)
    }

    override func truncate(to count: Int) { hashes.removeLast(hashes.count - count) }
}

/// First index of the maximum.
func firstMax(_ row: [Float]) -> Int32 {
    var best = 0
    for i in 1..<row.count where row[i] > row[best] { best = i }
    return Int32(best)
}

/// Plain greedy reference (first maximum), stopping inclusive at a stop token. `capacity` is the
/// KV length: a token is fed back only while it fits, and the last emitted token need not be.
func plainGreedy(
    model: HashModel, prompt: [Int32], maxNew: Int, stops: Set<Int32>, capacity: Int = Int.max
) -> [Int32] {
    var h: UInt64 = 0
    var processed = 0
    for t in prompt {
        h = model.step(h, t)
        processed += 1
    }
    var out: [Int32] = []
    while out.count < maxNew {
        let next = firstMax(model.row(h))
        out.append(next)
        if stops.contains(next) { break }
        if processed + 1 > capacity { break }
        h = model.step(h, next)
        processed += 1
    }
    return out
}

/// Proposes the true continuation with random corruption.
final class FuzzDrafter: DrafterEngine, @unchecked Sendable {
    let model: HashModel
    let maxContextLength: Int
    var rng: SplitMix

    init(model: HashModel, seed: UInt64, maxContextLength: Int = 100_000) {
        self.model = model
        self.rng = SplitMix(seed)
        self.maxContextLength = maxContextLength
    }

    func reset(to tokenIndex: Int) async throws {}
    func commit(confirmedLength: Int, features: DrafterFeatures?) async throws {}

    func draftAll(
        confirmedPrefix: [Int32], count: Int, policy: VerificationPolicy,
        using generator: inout some RandomNumberGenerator
    ) async throws -> DraftProposal {
        let mode = Int(rng.next() % 5)
        if mode == 0 { return DraftProposal(tokens: []) }
        var h: UInt64 = 0
        for t in confirmedPrefix { h = model.step(h, t) }
        var tokens: [Int32] = []
        let accept = Int(rng.next() % UInt64(count + 1))
        for i in 0..<count {
            var t = firstMax(model.row(h))
            if mode == 1 || i == accept || (i > accept && rng.next() % 2 == 0) {
                t = Int32(rng.next() % UInt64(model.vocab))
            }
            tokens.append(t)
            h = model.step(h, t)
        }
        return DraftProposal(tokens: tokens)
    }
}

/// An engine whose logits depend only on the last token: `table[token]` is the next-token row.
final class TableEngine: RowEngine, @unchecked Sendable {
    let table: [[Float]]

    init(table: [[Float]]) {
        self.table = table
        super.init(maxContextLength: 100_000, block: nil, tile: nil, maxVerifyBlockLength: Int.max)
    }

    override func push(_ token: Int32) -> [Float] { table[Int(token)] }
    override func truncate(to count: Int) {}
}
