// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CryptoKit
import Foundation

/// Chunks of a sequence's state, keyed by the exact token prefix that produced them.
///
/// A node covers one chunk of `chunkTokens` tokens and is named by the chained SHA-256 of every token
/// from the start of the sequence to the end of that chunk. A chunk's KV is a function of exactly that
/// prefix, so the key identifies it, and two conversations that share a system prompt share its nodes.
///
/// A node holds its chunk's KV rows as bytes, and on a hybrid model it may also hold a snapshot of the
/// recurrent state at its end. The recurrent state at one position cannot be rebuilt without replaying
/// every token before it, so on a hybrid reuse stops at the deepest snapshot at or above the match:
///
/// ```
/// attention only   reuse = matched chunks
/// hybrid           reuse = deepest snapshot at or above the match   (0 if there is none)
/// ```
///
/// Everything a node holds counts against one byte budget. When an insert would pass it, the least
/// recently used leaf goes first, so no chain is ever left without the nodes above it.
package struct PrefixTree: Sendable {
    /// A node's name: the chained SHA-256 of every token up to and including its chunk, as hex.
    package typealias Key = String

    /// Where a chain starts. Any fixed value; it is never itself a node.
    package static let rootKey: Key = "root"

    private struct Node {
        /// Chunks from the root, so this node covers tokens `[(depth - 1) * chunkTokens, depth * chunkTokens)`.
        let depth: Int
        let parent: Key?
        /// The ids this node covers, compared on a match as a guard against a chaining mistake.
        let tokens: [Int32]
        /// The chunk's KV rows, one blob per KV state, as ``KVChunk`` lays them out.
        let chunk: [Data]
        /// The recurrent state at the end of this chunk, one blob per state, or nil.
        var snapshot: [Data]?
        /// Nodes whose parent this is. Only a childless node is evictable, so no chain is stranded.
        var children = 0
        /// LRU clock reading, bumped on every match that walks through it.
        var lastUsed = 0

        var bytes: Int { chunk.reduce(0) { $0 + $1.count } + (snapshot?.reduce(0) { $0 + $1.count } ?? 0) }
    }

    private var nodes: [Key: Node] = [:]
    private var clock = 0

    /// Tokens one node covers, and the granularity of every depth in here.
    package let chunkTokens: Int

    /// Bytes the tree may hold, chunks and snapshots together.
    package let byteBudget: Int

    /// Bytes held now.
    package private(set) var bytesHeld = 0

    // Counters, for the summary.
    package private(set) var hits = 0
    package private(set) var misses = 0
    /// Tokens restored rather than prefilled, across every request.
    package private(set) var reusedTokens = 0
    /// Tokens matched in the tree but replayed for want of a snapshot at that depth.
    package private(set) var matchedButReplayedTokens = 0
    /// Nodes dropped to stay within the budget.
    package private(set) var evictedNodes = 0

    package init(chunkTokens: Int, byteBudget: Int) {
        precondition(chunkTokens > 0, "a chunk holds at least one token")
        precondition(byteBudget >= 0, "a byte budget is not negative")
        self.chunkTokens = chunkTokens
        self.byteBudget = byteBudget
    }

    /// Nodes held.
    package var nodeCount: Int { nodes.count }

    /// Snapshots held.
    package var snapshotCount: Int { nodes.values.count { $0.snapshot != nil } }

    // MARK: - Naming a prefix

    /// Extends `parent`'s key with one chunk of tokens.
    ///
    /// The parent's digest goes in first, so a chunk's name depends on every token before it.
    package static func chain(_ parent: Key, _ tokens: ArraySlice<Int32>) -> Key {
        var hasher = SHA256()
        hasher.update(data: Data(parent.utf8))
        Array(tokens).withUnsafeBytes { hasher.update(bufferPointer: $0) }
        var hex = [UInt8]()
        hex.reserveCapacity(SHA256.byteCount * 2)
        for byte in hasher.finalize() {
            hex.append(hexDigits[Int(byte >> 4)])
            hex.append(hexDigits[Int(byte & 0x0f)])
        }
        return String(decoding: hex, as: UTF8.self)
    }

    private static let hexDigits = Array("0123456789abcdef".utf8)

    // MARK: - Matching

    /// What an incoming prompt found in the tree.
    package struct Match: Sendable {
        /// The nodes on the usable path, in order. The last is where publishing resumes.
        package var keys: [Key] = []
        /// Their chunks, in order, to copy into the KV cache.
        package var chunks: [[Data]] = []
        /// The recurrent state at exactly `reusedTokens`, for a hybrid; nil otherwise.
        package var snapshot: [Data]?
        /// Prompt tokens the caller will not have to compute.
        package var reusedTokens = 0
        /// Tokens matched past `reusedTokens`, which a hybrid replays for want of a snapshot there.
        package var replayedTokens = 0

        package var isEmpty: Bool { reusedTokens == 0 }
    }

    /// Walks `tokens` down the tree and reports what is usable. Changes nothing: hand the result to
    /// ``record(_:restored:)`` once it is known whether the caller used it.
    ///
    /// - Parameters:
    ///   - limit: Tokens the caller may skip at most: `prompt.count - 1`, since at least one token has to
    ///     run to produce the first logits.
    ///   - needsSnapshot: True for a hybrid. Caps the result at the deepest snapshot on the path.
    package func match(_ tokens: [Int32], limit: Int, needsSnapshot: Bool) -> Match {
        var result = Match()
        guard !nodes.isEmpty, limit >= chunkTokens else { return result }

        var key = Self.rootKey
        var path: [Key] = []
        var snapshotDepth = 0
        var depth = 0
        while (depth + 1) * chunkTokens <= Swift.min(limit, tokens.count) {
            let chunk = tokens[(depth * chunkTokens)..<((depth + 1) * chunkTokens)]
            key = Self.chain(key, chunk)
            guard let node = nodes[key], node.tokens.elementsEqual(chunk) else { break }
            path.append(key)
            if node.snapshot != nil { snapshotDepth = path.count }
            depth += 1
        }

        let usableDepth = needsSnapshot ? snapshotDepth : path.count
        for key in path.prefix(usableDepth) {
            result.keys.append(key)
            result.chunks.append(nodes[key]!.chunk)
        }
        result.reusedTokens = usableDepth * chunkTokens
        result.replayedTokens = (path.count - usableDepth) * chunkTokens
        if needsSnapshot, let last = result.keys.last { result.snapshot = nodes[last]?.snapshot }
        return result
    }

    /// Counts a ``match(_:limit:needsSnapshot:)``, and marks its nodes used.
    ///
    /// - Parameter restored: Whether the caller restored it. One it passed over, for a history that
    ///   already held as much, still keeps its nodes warm but is neither a hit nor a miss.
    package mutating func record(_ match: Match, restored: Bool) {
        if !match.keys.isEmpty {
            clock += 1
            for key in match.keys { nodes[key]?.lastUsed = clock }
        }
        if restored {
            hits += 1
            reusedTokens += match.reusedTokens
            matchedButReplayedTokens += match.replayedTokens
        } else if match.isEmpty {
            misses += 1
            matchedButReplayedTokens += match.replayedTokens
        }
    }

    // MARK: - Inserting

    /// Publishes one full chunk as a child of `parent`.
    ///
    /// The caller supplies the parent's key, one this tree returned, rather than re-deriving the chain
    /// above, which would be quadratic in depth.
    ///
    /// - Parameters:
    ///   - parent: The node at `depth - 1`, or nil at depth 1.
    ///   - tokens: Exactly the tokens this chunk covers.
    ///   - depth: Which chunk, one-based, covering `[(depth-1)*chunkTokens, depth*chunkTokens)`.
    ///   - bytes: What `chunk` will total, so a chunk that can't fit is never read.
    ///   - chunk: The chunk's KV rows, one blob per KV state.
    /// - Returns: The node's key, whether created here or already present, or nil when the chunk can't
    ///   hang here: `depth` is not positive or `tokens` is not one whole chunk; `parent` is not the node
    ///   at `depth - 1`; a node under the same key holds other tokens; or the budget can't make room
    ///   for it beside its chain.
    @discardableResult
    package mutating func insert(
        parent: Key?, tokens: ArraySlice<Int32>, depth: Int, bytes size: Int,
        chunk: @autoclosure () -> [Data]
    ) -> Key? {
        guard depth > 0, tokens.count == chunkTokens else { return nil }
        switch parent {
        case .none:
            guard depth == 1 else { return nil }
        case .some(let parent):
            guard nodes[parent]?.depth == depth - 1 else { return nil }
        }

        let key = Self.chain(parent ?? Self.rootKey, tokens)
        if let existing = nodes[key] {
            // Another sequence published this span first; keep its chunk, but this one just used it.
            guard existing.tokens.elementsEqual(tokens) else { return nil }
            clock += 1
            nodes[key]?.lastUsed = clock
            return key
        }

        // Read the chunk only once it is known to fit, since that is a copy out of the cache.
        guard makeRoom(for: size, keeping: parent) else { return nil }
        let bytes = chunk()
        precondition(bytes.reduce(0) { $0 + $1.count } == size, "chunk is not the size it was said to be")
        clock += 1
        nodes[key] = Node(
            depth: depth, parent: parent, tokens: Array(tokens), chunk: bytes, lastUsed: clock)
        bytesHeld += size
        if let parent { nodes[parent]?.children += 1 }
        return key
    }

    /// Whether `key` is a node.
    package func contains(_ key: Key) -> Bool { nodes[key] != nil }

    /// `key`'s chunk, for a caller writing it elsewhere.
    package func chunk(of key: Key) -> [Data]? { nodes[key]?.chunk }

    /// Whether `key` is a node that already holds a snapshot, so taking another is wasted.
    package func hasSnapshot(at key: Key) -> Bool {
        nodes[key]?.snapshot != nil
    }

    /// Hangs the recurrent state at the end of `key`'s chunk on it.
    ///
    /// - Parameter tip: The deepest node of the chain `key` is on, kept with everything above it
    ///   while room is made, so the snapshot does not cost the chain it belongs to. `key` when nil.
    /// - Returns: False when `key` is gone or already has one, or the budget can't make room.
    @discardableResult
    package mutating func attach(snapshot: [Data], to key: Key, keepingChainTo tip: Key? = nil) -> Bool {
        guard let node = nodes[key], node.snapshot == nil else { return false }
        let size = snapshot.reduce(0) { $0 + $1.count }
        if bytesHeld + size > byteBudget {
            guard makeRoom(for: size, protecting: ancestors(of: key).union(ancestors(of: tip))) else { return false }
        }
        clock += 1
        nodes[key]?.snapshot = snapshot
        nodes[key]?.lastUsed = clock
        bytesHeld += size
        return true
    }

    /// Evicts least-recently-used leaves until at least `bytes` are freed, or the tree is empty, for a
    /// caller short of memory. Returns the bytes freed.
    @discardableResult
    package mutating func evict(atLeast bytes: Int) -> Int {
        var freed = 0
        while freed < bytes {
            let candidate = nodes.lazy.filter { $0.value.children == 0 }.min { $0.value.lastUsed < $1.value.lastUsed }
            guard let (victim, node) = candidate else { break }
            nodes.removeValue(forKey: victim)
            if let parent = node.parent { nodes[parent]?.children -= 1 }
            bytesHeld -= node.bytes
            evictedNodes += 1
            freed += node.bytes
        }
        return freed
    }

    // MARK: - Eviction

    /// Evicts least-recently-used leaves until `bytes` more fit, never `keeping` or its ancestors.
    ///
    /// False when they cannot fit at all.
    private mutating func makeRoom(for bytes: Int, keeping key: Key?) -> Bool {
        // The chain above is only walked when something has to go.
        guard bytesHeld + bytes > byteBudget else { return true }
        return makeRoom(for: bytes, protecting: ancestors(of: key))
    }

    /// Evicts least-recently-used leaves until `bytes` more fit, never one in `protected`, which must
    /// hold every ancestor of each of its nodes. False, having evicted nothing, when they cannot fit.
    private mutating func makeRoom(for bytes: Int, protecting protected: Set<Key>) -> Bool {
        // Every other node is evictable, leaves first, so this is exactly what can be freed.
        let protectedBytes = protected.reduce(0) { $0 + (nodes[$1]?.bytes ?? 0) }
        guard protectedBytes + bytes <= byteBudget else { return false }
        while bytesHeld + bytes > byteBudget {
            let candidate = nodes.lazy
                .filter { $0.value.children == 0 && !protected.contains($0.key) }
                .min { $0.value.lastUsed < $1.value.lastUsed }
            guard let (victim, node) = candidate else { return false }
            nodes.removeValue(forKey: victim)
            if let parent = node.parent { nodes[parent]?.children -= 1 }
            bytesHeld -= node.bytes
            evictedNodes += 1
        }
        return true
    }

    /// `key` and every node above it.
    private func ancestors(of key: Key?) -> Set<Key> {
        var out = Set<Key>()
        var next = key
        while let current = next, let node = nodes[current] {
            out.insert(current)
            next = node.parent
        }
        return out
    }

    /// One line for a log.
    package var summary: String {
        let requests = hits + misses
        let rate = requests > 0 ? Int((Double(hits) / Double(requests) * 100).rounded()) : 0
        let held = ByteCountFormatter.string(fromByteCount: Int64(bytesHeld), countStyle: .memory)
        let budget = ByteCountFormatter.string(fromByteCount: Int64(byteBudget), countStyle: .memory)
        return "\(nodeCount) node(s), \(snapshotCount) snapshot(s), \(held) of \(budget), "
            + "\(hits)/\(requests) hit (\(rate)%), \(reusedTokens) token(s) reused, "
            + "\(matchedButReplayedTokens) replayed for want of a snapshot, \(evictedNodes) node(s) evicted"
    }
}
