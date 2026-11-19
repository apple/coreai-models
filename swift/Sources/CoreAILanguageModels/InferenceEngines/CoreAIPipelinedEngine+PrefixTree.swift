// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Metal

// The pipelined engine's prefix cache: publishing what it computed into a ``PrefixTree``, restoring a
// shared prefix from it, and the recurrent-state snapshots a hybrid model needs to resume.

extension CoreAIPipelinedEngine {
    /// Tokens per prefix-cache chunk: the granularity a later prompt can resume at.
    static let prefixChunkTokens = 256

    var prefixCacheEnabled: Bool { prefixTree != nil }

    func shedPrefixCache(atLeast bytes: Int) -> Int {
        guard prefixTree != nil, tryAcquireEngine() else { return 0 }
        defer { releaseEngine() }
        // Pending snapshot copies first: until the next publish they are held outside the budget.
        var freed = engine.capturedSnapshots.reduce(0) { $0 + $1.copies.reduce(0) { $0 + $1.length } }
        engine.capturedSnapshots = []
        if freed < bytes { freed += prefixTree?.evict(atLeast: bytes - freed) ?? 0 }
        // Eviction takes leaves, so what is left of the published chain is a prefix of it.
        publishedKeys = Array(publishedKeys.prefix { prefixTree?.contains($0) == true })
        if freed > 0 {
            CLILogger.log(
                "prefix cache: gave back "
                    + ByteCountFormatter.string(fromByteCount: Int64(freed), countStyle: .memory)
                    + " for memory; " + (prefixTree?.summary ?? ""))
        }
        return freed
    }

    /// Publishes the current history's full chunks not yet in the tree, and the recurrent snapshot
    /// its last prompt saved. Call holding the engine, before anything discards the history.
    func publishHistory() async {
        guard prefixTree != nil else { return }
        let chunk = Self.prefixChunkTokens
        // The last sampled token is in the history but not yet in the KV cache.
        let valid = min(history.count, engine.processedTokenCount)
        let depth = valid / chunk
        guard depth > publishedKeys.count || !engine.capturedSnapshots.isEmpty else { return }
        await engine.computeStream.currentWorkCompleted()
        let tokens = history.tokens
        let chunkBytes = engine.kvChunkBytes(count: chunk)
        while publishedKeys.count < depth {
            let next = publishedKeys.count + 1
            let start = (next - 1) * chunk
            let parent = publishedKeys.last
            guard
                let key = prefixTree?.insert(
                    parent: parent, tokens: tokens[start..<(start + chunk)],
                    depth: next, bytes: chunkBytes,
                    chunk: self.engine.readKVChunk(start: start, count: chunk))
            else { break }
            publishedKeys.append(key)
            if let store = prefixStore, store.record(of: key) == nil, let bytes = prefixTree?.chunk(of: key) {
                store.write(chunk: bytes, for: key, parent: parent, depth: next)
            }
        }
        let snapshots = engine.capturedSnapshots
        engine.capturedSnapshots = []
        for snapshot in snapshots {
            let at = snapshot.position / chunk
            if snapshot.position % chunk == 0, at >= 1, at <= publishedKeys.count,
                snapshot.position <= valid, let states = snapshot.states()
            {
                prefixTree?.attach(
                    snapshot: states, to: publishedKeys[at - 1], keepingChainTo: publishedKeys.last)
                // On disk even when memory had no room for it.
                prefixStore?.write(snapshot: states, for: publishedKeys[at - 1])
            }
        }
    }

    /// Forgets what was published past the first `count` tokens of the history, which no longer
    /// describe it.
    func forgetPublished(beyond count: Int) {
        publishedKeys = Array(publishedKeys.prefix(count / Self.prefixChunkTokens))
        engine.capturedSnapshots.removeAll { $0.position > count }
    }

    /// Restores the longest prefix of `input` the tree holds, when it is at least a chunk longer than
    /// the `kept` tokens the engine is about to reuse anyway: less would save little prefill for the
    /// cost of the copy. Returns the tokens now in place, or nil when the engine is untouched.
    func restoreLongestPrefix(of input: [TokenId], keeping kept: Int) async -> Int? {
        let chunk = Self.prefixChunkTokens
        // At least one prompt token has to run, for the first logits.
        guard prefixTree != nil, (input.count - 1) / chunk * chunk >= kept + chunk else { return nil }
        let fromDisk = loadFromDisk(input, limit: input.count - 1, atLeast: kept + chunk)
        guard
            let match = prefixTree?.match(
                input, limit: input.count - 1, needsSnapshot: engine.hasNonTruncatableStates)
        else { return nil }
        guard match.reusedTokens >= kept + chunk else {
            prefixTree?.record(match, restored: false)
            return nil
        }
        let restored = match.reusedTokens
        await engine.resetWhenIdle()
        history.clear()
        publishedKeys = []
        do {
            try engine.restorePrefix(
                match, tokens: Array(input[..<restored]), chunkTokens: Self.prefixChunkTokens)
        } catch {
            CLILogger.log("prefix cache: could not restore \(restored) token(s): \(error)")
            engine.reset()
            return 0
        }
        prefixTree?.record(match, restored: true)
        prefixStore?.touch(match.keys)
        history.append(contentsOf: input[..<restored])
        publishedKeys = match.keys
        CLILogger.log(
            "prefix cache: restored \(restored) of \(input.count) prompt token(s)"
                + (fromDisk > 0 ? ", \(fromDisk) chunk(s) of them from disk" : "") + "; "
                + (prefixTree?.summary ?? "") + (prefixStore.map { "; " + $0.summary } ?? ""))
        return restored
    }

    /// Brings the chunks of `input` the tree lacks but the disk tier holds into the tree, so the match
    /// after finds them. Reads only up to the deepest chunk a match could use, and nothing when that is
    /// short of `minimum` tokens. Returns the chunks it brought.
    private func loadFromDisk(_ input: [TokenId], limit: Int, atLeast minimum: Int) -> Int {
        guard let store = prefixStore, prefixTree != nil else { return 0 }
        let chunk = Self.prefixChunkTokens
        let needsSnapshot = engine.hasNonTruncatableStates
        // Name the chain first, reading nothing: where it stops, and the deepest a match could use.
        var keys: [PrefixTree.Key] = []
        var usable = 0
        while (keys.count + 1) * chunk <= limit {
            let span = input[(keys.count * chunk)..<((keys.count + 1) * chunk)]
            let key = PrefixTree.chain(keys.last ?? PrefixTree.rootKey, span)
            let inMemory = prefixTree?.contains(key) ?? false
            let record = store.record(of: key)
            guard inMemory || (record?.parent == keys.last && record?.depth == keys.count + 1) else { break }
            keys.append(key)
            if !needsSnapshot || prefixTree?.hasSnapshot(at: key) == true || (record?.snapshotBytes ?? 0) > 0 {
                usable = keys.count
            }
        }
        guard usable * chunk >= minimum else { return 0 }

        var loaded = 0
        for (offset, key) in keys.prefix(usable).enumerated() {
            // A hybrid only needs the snapshot the match will stop at.
            let wantsSnapshot = needsSnapshot && offset == usable - 1 && prefixTree?.hasSnapshot(at: key) != true
            let inMemory = prefixTree?.contains(key) ?? false
            guard !inMemory || wantsSnapshot else { continue }
            guard let files = store.read(key, withSnapshot: wantsSnapshot) else { break }
            if !inMemory {
                guard engine.fitsKVChunk(files.chunk, count: chunk) else {
                    store.remove(key)
                    break
                }
                let parent = offset == 0 ? nil : keys[offset - 1]
                guard
                    prefixTree?.insert(
                        parent: parent, tokens: input[(offset * chunk)..<((offset + 1) * chunk)],
                        depth: offset + 1, bytes: files.chunk.reduce(0) { $0 + $1.count },
                        chunk: files.chunk) != nil
                else { break }
                loaded += 1
            }
            if let snapshot = files.snapshot {
                guard engine.fitsSnapshot(snapshot) else {
                    store.remove(key)
                    break
                }
                prefixTree?.attach(snapshot: snapshot, to: key, keepingChainTo: key)
            }
        }
        return loaded
    }

    /// Where prefill should snapshot a hybrid's recurrent state for `input`, of which the engine holds
    /// the first `processedTokenCount`: the ``PrefixSnapshotPlan`` boundaries it reaches, but not one
    /// whose node already has a snapshot.
    func planSnapshots(for input: [TokenId]) {
        engine.snapshotPositions = []
        guard let tree = prefixTree, engine.hasNonTruncatableStates else { return }
        let chunk = Self.prefixChunkTokens
        let start = engine.processedTokenCount
        let boundaries = PrefixSnapshotPlan.boundaries(start: start, promptTokens: input.count - start)
        // Each costs a copy of the recurrent state. The boundaries lie past what is published, so
        // their keys chain on from its last.
        var key = publishedKeys.last ?? PrefixTree.rootKey
        var depth = publishedKeys.count
        var wanted: [Int] = []
        for boundary in boundaries where boundary / chunk > publishedKeys.count {
            while depth < boundary / chunk {
                key = PrefixTree.chain(key, input[(depth * chunk)..<((depth + 1) * chunk)])
                depth += 1
            }
            if !tree.hasSnapshot(at: key) { wanted.append(boundary) }
        }
        // Until the next publish, every copy is held outside the budget, so a long prompt takes no
        // more than half the budget's worth: its deepest, for the same prompt again, then the
        // shallowest, which the most prompts share. Prefill stops at the others all the same.
        let size = engine.additionalStates?.byteCount ?? 0
        let most = size > 0 ? max(1, tree.byteBudget / 2 / size) : wanted.count
        if wanted.count > most, let last = wanted.last {
            wanted = Array(wanted.prefix(most - 1)) + [last]
        }
        engine.snapshotPositions = wanted
    }
}

extension EngineImpl {
    /// Why this engine's state can't go into the prefix cache, or nil when it can.
    var prefixCacheUnsupportedReason: String? {
        // A ring holds positions modulo its size, so its contents are no chunk of the sequence.
        if slidingRingCapacity != nil { return "its sliding-window cache is a ring" }
        // A KV pair beyond the first sits in `additionalStates` and would not be restored with it.
        if hasExtraGrowingStates { return "it has a second growing KV cache" }
        // Only `.fixed` states are snapshotted; a sliding one would be left as zeros by a restore.
        if hasSlidingStates { return "it keeps a sliding-window state outside its KV cache" }
        // A fixed query length cannot take the split a snapshot needs.
        if case .fixed(let length) = inputLayout.queryPolicy, length > 1 {
            return "its graph takes a fixed query length of \(length)"
        }
        guard
            Self.copiesWholeElements(kvCache.keyBinding.scalarType)
                && Self.copiesWholeElements(kvCache.valueBinding.scalarType)
        else { return "its KV cache's element type is not a whole number of bytes" }
        return nil
    }

    /// Whether elements of `type` take whole bytes, as ``KVChunk`` walks them; sub-byte types do not.
    private static func copiesWholeElements(_ type: NDArray.ScalarType) -> Bool {
        switch type {
        case .float16, .bfloat16, .float32, .int8, .uint8, .float8e5m2, .float8e4m3fn: true
        default: false
        }
    }

    /// What the prefix cache's files depend on, besides the model: the chunk length, the KV's layout
    /// without its capacity, and the recurrent states' sizes.
    func prefixCacheGeometry(chunkTokens: Int) -> String {
        func describe(_ binding: TensorBinding) -> String {
            var shape = binding.shape.map(String.init)
            shape[KVChunk.Layout(binding).sequenceAxis] = "*"
            return "\(shape.joined(separator: "x")) \(binding.scalarType)"
        }
        let states = additionalStates?.snapshotLayout.map { "state \($0.name) \($0.bytes)" } ?? []
        return
            ([
                "prefix cache 1", "chunk \(chunkTokens)", "key \(describe(kvCache.keyBinding))",
                "value \(describe(kvCache.valueBinding))",
            ] + states).joined(separator: "\n")
    }

    /// Whether `chunk` is the keys and values of `count` positions of this KV cache.
    func fitsKVChunk(_ chunk: [Data], count: Int) -> Bool {
        chunk.map(\.count) == [
            KVChunk.Layout(kvCache.keyBinding).chunkBytes(count: count),
            KVChunk.Layout(kvCache.valueBinding).chunkBytes(count: count),
        ]
    }

    /// Whether `snapshot` is one of these recurrent states.
    func fitsSnapshot(_ snapshot: [Data]) -> Bool {
        additionalStates?.snapshotLayout.map(\.bytes) == snapshot.map(\.count)
    }

    /// Bytes one chunk of `count` positions takes, keys and values, before any is read.
    func kvChunkBytes(count: Int) -> Int {
        KVChunk.Layout(kvCache.keyBinding).chunkBytes(count: count)
            + KVChunk.Layout(kvCache.valueBinding).chunkBytes(count: count)
    }

    /// `plan`, prefill steps of up to `stepTokens` from ``processedTokenCount``. A recurrent model takes
    /// ``PrefixSnapshotPlan/steps(from:to:stepTokens:ringCapacity:)`` over the same span instead: steps
    /// between the same absolute positions wherever it starts, so its output does not depend on what the
    /// prefix cache restored, and every snapshot position is one of them. Other models run `plan`.
    func splittingAtStops(_ plan: [Int], stepTokens: Int) -> [Int] {
        guard hasNonTruncatableStates else { return plan }
        let start = processedTokenCount
        return PrefixSnapshotPlan.steps(
            from: start, to: start + plan.reduce(0, +), stepTokens: stepTokens, ringCapacity: slidingRingCapacity)
    }

    /// Saves the recurrent state when prefill has just reached the next of ``snapshotPositions``.
    ///
    /// A copy on the GPU, so nothing is read back here. It waits for the stream first: nothing promises
    /// that the step which got here has reached the queue when its encode returns, and a copy queued
    /// ahead of it would hold the state from before it.
    mutating func snapshotIfDue() async {
        // A position prefill went past without stopping can't be taken any more.
        snapshotPositions.removeAll { $0 < processedTokenCount }
        guard snapshotPositions.first == processedTokenCount else { return }
        snapshotPositions.removeFirst()
        guard let states = additionalStates else { return }
        await computeStream.currentWorkCompleted()
        guard let commandBuffer = pipelineQueue.makeCommandBuffer(),
            let copies = states.encodeSnapshot(on: commandBuffer)
        else { return }
        commandBuffer.commit()
        capturedSnapshots.append(
            PendingSnapshot(position: processedTokenCount, copies: copies, commandBuffer: commandBuffer))
    }

    /// Positions `[start, start + count)` of the key and value caches, as dense chunks.
    /// Caller must ensure no in-flight GPU work writes them.
    func readKVChunk(start: Int, count: Int) -> [Data] {
        [kvCache.keyBinding, kvCache.valueBinding].map { binding in
            KVChunk.read(
                from: UnsafeRawPointer(binding.metalBuffer.contents()),
                layout: KVChunk.Layout(binding), start: start, count: count)
        }
    }

    /// Installs a cached prefix after ``reset()``: its chunks become the KV cache's first positions,
    /// its snapshot the recurrent state, and its tokens the token buffer's, so prefill resumes after it.
    /// Caller must ensure no in-flight GPU work references any of them.
    mutating func restorePrefix(_ match: PrefixTree.Match, tokens: [Int32], chunkTokens: Int) throws {
        let count = tokens.count
        let needed = Swift.min(count + 1, config.maxContextLength)
        if needed > kvCache.currentCapacity {
            _ = try kvCache.ensureCapacity(forContextLength: needed, queue: pipelineQueue)
            // Growth copies the old cache on this queue; it must land before these writes do.
            if let fence = pipelineQueue.makeCommandBuffer() {
                fence.commit()
                fence.waitUntilCompleted()
            }
        }
        let bindings = [kvCache.keyBinding, kvCache.valueBinding]
        for (index, chunk) in match.chunks.enumerated() {
            for (blob, binding) in zip(chunk, bindings) {
                KVChunk.write(
                    blob, to: binding.metalBuffer.contents(), layout: KVChunk.Layout(binding),
                    start: index * chunkTokens, count: chunkTokens)
            }
        }
        if let snapshot = match.snapshot { additionalStates?.restore(snapshot) }
        let ids = inputTokensBuffer.contents().bindMemory(to: Int32.self, capacity: count)
        for (index, token) in tokens.enumerated() { ids[index] = token }
        processedTokenCount = count
        step = count
    }
}

/// A copy of the recurrent states that prefill queued on the GPU at `position`, for the prefix cache.
struct PendingSnapshot {
    let position: Int
    let copies: [MTLBuffer]
    let commandBuffer: MTLCommandBuffer

    /// The copies as blobs, once the GPU has made them, or nil when it failed to. Wraps the buffers
    /// rather than copying them.
    func states() -> [Data]? {
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else { return nil }
        return copies.map { buffer in
            Data(
                bytesNoCopy: buffer.contents(), count: buffer.length,
                deallocator: .custom { _, _ in withExtendedLifetime(buffer) {} })
        }
    }
}
