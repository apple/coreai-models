// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import Foundation
import Metal
import Testing

@testable import CoreAILanguageModels

@Suite("PrefixTree")
struct PrefixTreeTests {
    static let chunk = 4

    /// A chunk's KV as two recognisable blobs, so a match can be checked against what went in.
    static func kv(_ depth: Int, bytes: Int = 8) -> [Data] {
        [Data(repeating: UInt8(depth), count: bytes), Data(repeating: UInt8(100 + depth), count: bytes)]
    }

    /// Publishes every full chunk of `tokens`, as the engine does after running them.
    @discardableResult
    static func publish(_ tokens: [Int32], into tree: inout PrefixTree) -> [PrefixTree.Key] {
        var keys: [PrefixTree.Key] = []
        for depth in 1...(tokens.count / chunk) {
            let span = tokens[((depth - 1) * chunk)..<(depth * chunk)]
            guard let key = tree.insert(parent: keys.last, tokens: span, depth: depth, bytes: 16, chunk: kv(depth))
            else { break }
            keys.append(key)
        }
        return keys
    }

    static func tokens(_ count: Int, from start: Int32 = 0) -> [Int32] {
        (0..<count).map { start + Int32($0) }
    }

    @Test("A prompt that starts with a published one reuses every whole chunk they share")
    func sharedPrefixIsReused() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 1 << 20)
        Self.publish(Self.tokens(12), into: &tree)
        // Shares 10 tokens: two whole chunks, then a different third.
        let prompt = Self.tokens(10) + [99, 98, 97, 96, 95]
        let match = tree.match(prompt, limit: prompt.count - 1, needsSnapshot: false)
        #expect(match.reusedTokens == 8)
        #expect(match.chunks == [Self.kv(1), Self.kv(2)])
        #expect(match.keys.count == 2)
    }

    @Test("At least one token is left to compute, even when the whole prompt is cached")
    func limitLeavesOneToken() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 1 << 20)
        Self.publish(Self.tokens(8), into: &tree)
        let match = tree.match(Self.tokens(8), limit: 7, needsSnapshot: false)
        #expect(match.reusedTokens == 4)
    }

    @Test("A different first chunk misses, and so does a prompt shorter than a chunk")
    func misses() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 1 << 20)
        Self.publish(Self.tokens(8), into: &tree)
        let otherStart = tree.match(Self.tokens(8, from: 50), limit: 7, needsSnapshot: false)
        #expect(otherStart.isEmpty)
        let tooShort = tree.match(Self.tokens(3), limit: 2, needsSnapshot: false)
        #expect(tooShort.isEmpty)
    }

    @Test("Only a restored match counts as a hit; one passed over for the history counts as neither")
    func recordCountsWhatWasUsed() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 1 << 20)
        Self.publish(Self.tokens(8), into: &tree)
        let found = tree.match(Self.tokens(9), limit: 8, needsSnapshot: false)
        let missed = tree.match(Self.tokens(9, from: 50), limit: 8, needsSnapshot: false)
        #expect(tree.hits == 0 && tree.misses == 0)
        tree.record(found, restored: false)
        #expect(tree.hits == 0 && tree.misses == 0 && tree.reusedTokens == 0)
        tree.record(found, restored: true)
        tree.record(missed, restored: false)
        #expect(tree.hits == 1 && tree.misses == 1 && tree.reusedTokens == 8)
    }

    @Test("A chunk's name depends on every token before it, so the same chunk after another prefix is a different node")
    func keysChain() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 1 << 20)
        Self.publish(Self.tokens(8), into: &tree)
        let other = Self.tokens(4, from: 50) + Array(Self.tokens(8)[4..<8])
        let otherPrefix = tree.match(other, limit: other.count - 1, needsSnapshot: false)
        #expect(otherPrefix.isEmpty)
        #expect(
            PrefixTree.chain(PrefixTree.rootKey, Self.tokens(4)[...])
                != PrefixTree.chain(PrefixTree.chain(PrefixTree.rootKey, Self.tokens(4)[...]), Self.tokens(4)[...]))
    }

    @Test("On a model with recurrent state, reuse stops at the deepest snapshot on the path")
    func hybridStopsAtSnapshot() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 1 << 20)
        let keys = Self.publish(Self.tokens(16), into: &tree)
        let prompt = Self.tokens(16) + [99]
        // No snapshot yet: nothing usable, though four chunks matched.
        let unsnapshotted = tree.match(prompt, limit: 16, needsSnapshot: true)
        #expect(unsnapshotted.isEmpty)
        #expect(unsnapshotted.replayedTokens == 16)
        let state = [Data(repeating: 7, count: 16)]
        let attached = tree.attach(snapshot: state, to: keys[1])
        #expect(attached)
        #expect(tree.hasSnapshot(at: keys[1]) && !tree.hasSnapshot(at: keys[2]))
        let match = tree.match(prompt, limit: 16, needsSnapshot: true)
        #expect(match.reusedTokens == 8)
        #expect(match.replayedTokens == 8)
        #expect(match.snapshot == state)
        // Without needing one, every matched chunk is used and no snapshot handed out.
        let attention = tree.match(prompt, limit: 16, needsSnapshot: false)
        #expect(attention.reusedTokens == 16)
        #expect(attention.snapshot == nil)
    }

    @Test("A chunk only hangs from the node directly above it")
    func insertNeedsItsParent() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 1 << 20)
        let noParent = tree.insert(parent: nil, tokens: Self.tokens(4)[...], depth: 2, bytes: 16, chunk: Self.kv(2))
        #expect(noParent == nil)
        let unknownParent = tree.insert(
            parent: "nowhere", tokens: Self.tokens(4)[...], depth: 2, bytes: 16, chunk: Self.kv(2))
        #expect(unknownParent == nil)
        let partialChunk = tree.insert(
            parent: nil, tokens: Self.tokens(3)[...], depth: 1, bytes: 16, chunk: Self.kv(1))
        #expect(partialChunk == nil)
        #expect(tree.nodeCount == 0)
    }

    @Test("Publishing a span twice keeps the first copy, and does not read the chunk again")
    func republishingIsFree() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 1 << 20)
        let first = Self.publish(Self.tokens(4), into: &tree)
        var read = false
        let again = tree.insert(
            parent: nil, tokens: Self.tokens(4)[...], depth: 1, bytes: 16,
            chunk: {
                read = true
                return Self.kv(9)
            }())
        #expect(again == first.first)
        #expect(!read)
        let match = tree.match(Self.tokens(5), limit: 4, needsSnapshot: false)
        #expect(match.chunks == [Self.kv(1)])
    }

    @Test("Past the budget the least recently used leaf goes first, and never a chain's middle")
    func evictsColdestLeaf() {
        // Each node is 16 bytes; room for five.
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 80)
        Self.publish(Self.tokens(12), into: &tree)  // A: three nodes
        Self.publish(Self.tokens(8, from: 50), into: &tree)  // B: two nodes, now full
        // Use A, so B is the colder.
        tree.record(tree.match(Self.tokens(12) + [0], limit: 12, needsSnapshot: false), restored: true)
        Self.publish(Self.tokens(4, from: 90), into: &tree)  // C: one node, which evicts B's leaf
        #expect(tree.nodeCount == 5)
        #expect(tree.evictedNodes == 1)
        #expect(tree.bytesHeld <= 80)
        let chainA = tree.match(Self.tokens(12) + [0], limit: 12, needsSnapshot: false)
        #expect(chainA.reusedTokens == 12)
        let chainB = tree.match(Self.tokens(8, from: 50) + [0], limit: 8, needsSnapshot: false)
        #expect(chainB.reusedTokens == 4)
    }

    @Test("A chunk larger than the whole budget is refused, never read, and evicts nothing")
    func oversizedChunkIsRefused() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 20)
        Self.publish(Self.tokens(4), into: &tree)
        var read = false
        let oversized = tree.insert(
            parent: nil, tokens: Self.tokens(4, from: 9)[...], depth: 1, bytes: 128,
            chunk: {
                read = true
                return Self.kv(1, bytes: 64)
            }())
        #expect(oversized == nil)
        #expect(!read)
        #expect(tree.nodeCount == 1)
    }

    @Test("A chunk that can't fit beside the chain it extends evicts nothing")
    func infeasibleInsertEvictsNothing() {
        // Room for three 16-byte nodes: a chain of two, and one other.
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 48)
        let chain = Self.publish(Self.tokens(8), into: &tree)
        Self.publish(Self.tokens(4, from: 50), into: &tree)
        // A 40-byte third chunk can't fit beside the chain's 32 bytes, even with the other node gone.
        let tooBig = tree.insert(
            parent: chain[1], tokens: Self.tokens(12)[8...], depth: 3, bytes: 40, chunk: Self.kv(3, bytes: 20))
        #expect(tooBig == nil)
        #expect(tree.nodeCount == 3)
        #expect(tree.evictedNodes == 0)
    }

    @Test("A snapshot under a full budget evicts another chain, never the one it belongs to")
    func attachKeepsItsChain() {
        // Room for four 16-byte nodes; the snapshot needs one node's room.
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 64)
        Self.publish(Self.tokens(4, from: 50), into: &tree)
        let chain = Self.publish(Self.tokens(12), into: &tree)
        // Use the other chain, so the coldest leaf is this chain's own.
        tree.record(tree.match(Self.tokens(4, from: 50) + [0], limit: 4, needsSnapshot: false), restored: true)
        let attached = tree.attach(snapshot: [Data(count: 16)], to: chain[0], keepingChainTo: chain[2])
        #expect(attached)
        #expect(tree.nodeCount == 3)
        let chainMatch = tree.match(Self.tokens(12) + [0], limit: 12, needsSnapshot: false)
        #expect(chainMatch.reusedTokens == 12)
        let otherMatch = tree.match(Self.tokens(4, from: 50) + [0], limit: 4, needsSnapshot: false)
        #expect(otherMatch.isEmpty)
    }

    @Test("A snapshot for a node that is gone evicts nothing")
    func attachToMissingNode() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 32)
        Self.publish(Self.tokens(8), into: &tree)
        let attached = tree.attach(snapshot: [Data(count: 16)], to: "nowhere")
        #expect(!attached)
        #expect(tree.nodeCount == 2)
        #expect(tree.evictedNodes == 0)
    }

    @Test("Evicting a node frees its snapshot with it")
    func evictionFreesSnapshot() {
        var tree = PrefixTree(chunkTokens: Self.chunk, byteBudget: 48)
        let chain = Self.publish(Self.tokens(4), into: &tree)
        let attached = tree.attach(snapshot: [Data(count: 16)], to: chain[0])
        #expect(attached && tree.bytesHeld == 32 && tree.snapshotCount == 1)
        // A 32-byte chunk elsewhere only fits once the node and its snapshot go.
        let other = tree.insert(
            parent: nil, tokens: Self.tokens(4, from: 50)[...], depth: 1, bytes: 32, chunk: Self.kv(1, bytes: 16))
        #expect(other != nil)
        #expect(tree.nodeCount == 1 && tree.snapshotCount == 0 && tree.bytesHeld == 32)
    }
}

@Suite("PrefixSnapshotPlan")
struct PrefixSnapshotPlanTests {
    @Test("The snapshot goes at the last chunk boundary before the held-back token, past the start")
    func boundary() {
        // Prompt [0, 10): the last token is held back, so 9 run; the boundary under 9 is 8.
        #expect(PrefixSnapshotPlan.boundary(start: 0, promptTokens: 10, chunkTokens: 4) == 8)
        // Prompt [0, 9): 8 run, exactly to the boundary.
        #expect(PrefixSnapshotPlan.boundary(start: 0, promptTokens: 9, chunkTokens: 4) == 8)
        // Prompt [0, 8): 7 run, so the last boundary reached is 4.
        #expect(PrefixSnapshotPlan.boundary(start: 0, promptTokens: 8, chunkTokens: 4) == 4)
        // Starting on a boundary with too little to reach the next: none.
        #expect(PrefixSnapshotPlan.boundary(start: 8, promptTokens: 4, chunkTokens: 4) == nil)
        #expect(PrefixSnapshotPlan.boundary(start: 5, promptTokens: 4, chunkTokens: 4) == 8)
        #expect(PrefixSnapshotPlan.boundary(start: 0, promptTokens: 1, chunkTokens: 4) == nil)
    }

    @Test("The step crossing the target is cut there; one ending on it, or past it, is left alone")
    func split() {
        #expect(PrefixSnapshotPlan.split([6, 6, 3], from: 0, at: 8) == [6, 2, 4, 3])
        #expect(PrefixSnapshotPlan.split([6, 6, 3], from: 10, at: 16) == [6, 6, 3])
        #expect(PrefixSnapshotPlan.split([6, 6], from: 0, at: 6) == [6, 6])
        #expect(PrefixSnapshotPlan.split([6, 6], from: 0, at: 20) == [6, 6])
        #expect(PrefixSnapshotPlan.split([5], from: 3, at: 4) == [1, 4])
        #expect(PrefixSnapshotPlan.split([], from: 0, at: 4) == [])
    }
}

@Suite("KVChunk")
struct KVChunkTests {
    /// A `[layers, 1, heads, capacity, head_dim]` cache with each position padded to `rowStride`
    /// elements, every element holding its own coordinates so a misplaced copy shows.
    static func cache(
        layers: Int, heads: Int, capacity: Int, headDim: Int, rowStride: Int
    ) -> (values: [Float], layout: KVChunk.Layout) {
        let strides = [heads * capacity * rowStride, heads * capacity * rowStride, capacity * rowStride, rowStride, 1]
        var values = [Float](repeating: -1, count: layers * strides[0])
        for l in 0..<layers {
            for h in 0..<heads {
                for p in 0..<capacity {
                    for d in 0..<headDim {
                        values[l * strides[0] + h * strides[2] + p * strides[3] + d] =
                            Float(l * 1000 + h * 100 + p * 10 + d)
                    }
                }
            }
        }
        let layout = KVChunk.Layout(
            shape: [layers, 1, heads, capacity, headDim], strides: strides,
            elementBytes: MemoryLayout<Float>.size, sequenceAxis: 3)
        return (values, layout)
    }

    @Test("A chunk read from a padded cache is the slice along the sequence axis, densely, without the padding")
    func readDropsPadding() {
        let (values, layout) = Self.cache(layers: 2, heads: 2, capacity: 8, headDim: 3, rowStride: 4)
        let chunk = values.withUnsafeBytes { KVChunk.read(from: $0.baseAddress!, layout: layout, start: 2, count: 3) }
        let floats = chunk.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        var expected: [Float] = []
        for l in 0..<2 {
            for h in 0..<2 {
                for p in 2..<5 {
                    for d in 0..<3 { expected.append(Float(l * 1000 + h * 100 + p * 10 + d)) }
                }
            }
        }
        #expect(floats == expected)
    }

    @Test("A chunk goes back into a cache of another capacity at the same positions, and nowhere else")
    func writeRoundTripsAcrossCapacities() {
        let source = Self.cache(layers: 2, heads: 2, capacity: 8, headDim: 3, rowStride: 4)
        let chunk = source.values.withUnsafeBytes {
            KVChunk.read(from: $0.baseAddress!, layout: source.layout, start: 4, count: 4)
        }
        var destination = Self.cache(layers: 2, heads: 2, capacity: 16, headDim: 3, rowStride: 8)
        let before = destination.values
        destination.values.withUnsafeMutableBytes {
            KVChunk.write(chunk, to: $0.baseAddress!, layout: destination.layout, start: 4, count: 4)
        }
        let back = destination.values.withUnsafeBytes {
            KVChunk.read(from: $0.baseAddress!, layout: destination.layout, start: 4, count: 4)
        }
        #expect(back == chunk)
        // Positions outside the span and the padding are untouched.
        let strides = destination.layout.strides
        for l in 0..<2 {
            for h in 0..<2 {
                for p in 0..<16 where !(4..<8).contains(p) {
                    let base = l * strides[0] + h * strides[2] + p * strides[3]
                    #expect(Array(destination.values[base..<(base + 8)]) == Array(before[base..<(base + 8)]))
                }
            }
        }
    }

    @Test("Positions with no padding between them copy as one run per outer index, to the same result")
    func unpaddedRoundTrip() {
        let source = Self.cache(layers: 2, heads: 2, capacity: 8, headDim: 3, rowStride: 3)
        let chunk = source.values.withUnsafeBytes {
            KVChunk.read(from: $0.baseAddress!, layout: source.layout, start: 3, count: 4)
        }
        var expected: [Float] = []
        for l in 0..<2 {
            for h in 0..<2 {
                for p in 3..<7 {
                    for d in 0..<3 { expected.append(Float(l * 1000 + h * 100 + p * 10 + d)) }
                }
            }
        }
        #expect(chunk.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } == expected)
        var destination = Self.cache(layers: 2, heads: 2, capacity: 8, headDim: 3, rowStride: 3)
        destination.values = destination.values.map { _ in 0 }
        destination.values.withUnsafeMutableBytes {
            KVChunk.write(chunk, to: $0.baseAddress!, layout: destination.layout, start: 3, count: 4)
        }
        let back = destination.values.withUnsafeBytes {
            KVChunk.read(from: $0.baseAddress!, layout: destination.layout, start: 3, count: 4)
        }
        #expect(back == chunk)
        #expect(destination.values[2 * 3] == 0 && destination.values[7 * 3] == 0)
    }

    @Test("A layout whose inner axes are not dense is copied element by element, to the same result")
    func stridedInnerAxis() {
        // head_dim walked with stride 2: every other element.
        let layout = KVChunk.Layout(
            shape: [1, 1, 1, 4, 3], strides: [48, 48, 48, 8, 2], elementBytes: 4, sequenceAxis: 3)
        #expect(!layout.innerIsDense)
        var values = [Float](repeating: -1, count: 48)
        for p in 0..<4 { for d in 0..<3 { values[p * 8 + d * 2] = Float(p * 10 + d) } }
        let chunk = values.withUnsafeBytes { KVChunk.read(from: $0.baseAddress!, layout: layout, start: 1, count: 2) }
        #expect(chunk.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } == [10, 11, 12, 20, 21, 22])
    }

    @Test("Through a pipelined KV binding: the sequence axis and element size come from the binding")
    func throughABinding() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        // [layers, batch, heads, capacity, head_dim] float16, positions padded to 8 elements.
        let shape = [2, 1, 2, 6, 4]
        let strides = [96, 96, 48, 8, 1]
        let buffer = try #require(device.makeBuffer(length: 2 * 96 * 2, options: .storageModeShared))
        let halves = buffer.contents().bindMemory(to: UInt16.self, capacity: 2 * 96)
        for i in 0..<(2 * 96) { halves[i] = UInt16(i) }
        let binding = TensorBinding(metalBuffer: buffer, shape: shape, strides: strides, scalarType: .float16)
        let layout = KVChunk.Layout(binding)
        #expect(layout.sequenceAxis == 3)
        #expect(layout.elementBytes == 2)
        let chunk = KVChunk.read(from: buffer.contents(), layout: layout, start: 1, count: 2)
        #expect(chunk.count == 2 * 2 * 2 * 4 * 2)
        // Clear the span, write the chunk back, and read it again.
        for l in 0..<2 {
            for h in 0..<2 { for p in 1..<3 { for d in 0..<4 { halves[l * 96 + h * 48 + p * 8 + d] = 0 } } }
        }
        KVChunk.write(chunk, to: buffer.contents(), layout: layout, start: 1, count: 2)
        #expect(KVChunk.read(from: buffer.contents(), layout: layout, start: 1, count: 2) == chunk)
        let probe = 1 * 48 + 2 * 8 + 3
        #expect(halves[probe] == UInt16(probe))
    }
}
