// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("PrefixCacheStore")
struct PrefixCacheStoreTests {
    static let geometry = "test geometry"

    /// A chunk as two recognisable blobs; 36 bytes on disk with its header.
    static func kv(_ depth: Int) -> [Data] {
        [Data(repeating: UInt8(depth), count: 8), Data(repeating: UInt8(100 + depth), count: 8)]
    }

    static let nodeBytes = PrefixCacheStore.encodedSize(of: kv(1))

    /// A fresh directory, deleted when `body` returns.
    static func withDirectory(_ body: (URL) throws -> Void) rethrows {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PrefixCacheStoreTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    /// Writes a chain of `count` chunks, keyed as the tree would key `tokens`, and returns its keys.
    @discardableResult
    static func writeChain(_ count: Int, from start: Int32 = 0, into store: PrefixCacheStore) -> [PrefixTree.Key] {
        var keys: [PrefixTree.Key] = []
        for depth in 1...count {
            let tokens = (0..<4).map { start + Int32((depth - 1) * 4 + $0) }
            let key = PrefixTree.chain(keys.last ?? PrefixTree.rootKey, tokens[...])
            store.write(chunk: kv(depth), for: key, parent: keys.last, depth: depth)
            keys.append(key)
        }
        store.flush()
        return keys
    }

    static func files(in directory: URL, withExtension pathExtension: String) -> [URL] {
        let all = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return all.filter { $0.pathExtension == pathExtension }
    }

    @Test("A chunk and its snapshot read back as written")
    func roundTrip() throws {
        try Self.withDirectory { directory in
            let store = try #require(
                PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: Self.geometry))
            let keys = Self.writeChain(2, into: store)
            let state = [Data(repeating: 7, count: 32), Data(repeating: 9, count: 4)]
            store.write(snapshot: state, for: keys[1])
            store.flush()
            let withState = try #require(store.read(keys[1], withSnapshot: true))
            #expect(withState.chunk == Self.kv(2))
            #expect(withState.snapshot == state)
            let without = try #require(store.read(keys[1], withSnapshot: false))
            #expect(without.snapshot == nil)
            #expect(store.record(of: keys[1])?.parent == keys[0])
            #expect(store.record(of: keys[1])?.depth == 2)
            #expect(store.bytesHeld == 2 * Self.nodeBytes + PrefixCacheStore.encodedSize(of: state))
        }
    }

    @Test("A store opened again on the same directory holds what the last one wrote")
    func survivesReopening() throws {
        try Self.withDirectory { directory in
            let keys: [PrefixTree.Key]
            do {
                let store = try #require(
                    PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: Self.geometry))
                keys = Self.writeChain(3, into: store)
                store.write(snapshot: [Data(count: 16)], for: keys[2])
                store.flush()
            }
            let reopened = try #require(
                PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: Self.geometry))
            #expect(reopened.nodeCount == 3)
            let deepest = try #require(reopened.read(keys[2], withSnapshot: true))
            #expect(deepest.chunk == Self.kv(3))
            #expect(deepest.snapshot == [Data(count: 16)])
        }
    }

    @Test("A directory written for another layout is emptied")
    func otherGeometryIsDiscarded() throws {
        try Self.withDirectory { directory in
            do {
                let store = try #require(
                    PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: Self.geometry))
                Self.writeChain(2, into: store)
            }
            let other = try #require(PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: "another"))
            #expect(other.nodeCount == 0)
            #expect(Self.files(in: directory, withExtension: "kv").isEmpty)
        }
    }

    @Test("A chunk whose parent is not held, or that can't fit beside its chain, is not written")
    func refusals() throws {
        try Self.withDirectory { directory in
            // Room for three nodes.
            let store = try #require(
                PrefixCacheStore(directory: directory, byteBudget: 3 * Self.nodeBytes, geometry: Self.geometry))
            store.write(chunk: Self.kv(2), for: "orphan", parent: "nowhere", depth: 2)
            store.write(chunk: Self.kv(2), for: "deep", parent: nil, depth: 2)
            store.flush()
            #expect(store.nodeCount == 0)
            let chain = Self.writeChain(2, into: store)
            Self.writeChain(1, from: 50, into: store)
            // Twice a node's size can't sit beside the chain's two, even with the other node gone.
            store.write(chunk: [Data(count: 2 * Self.nodeBytes)], for: "big", parent: chain[1], depth: 3)
            store.flush()
            #expect(store.nodeCount == 3)
            #expect(store.record(of: "big") == nil)
        }
    }

    @Test("Past the budget the least recently used leaf goes, files and all, and that order survives a restart")
    func evictsColdestLeaf() throws {
        try Self.withDirectory { directory in
            let budget = 3 * Self.nodeBytes
            let a: [PrefixTree.Key]
            let b: [PrefixTree.Key]
            do {
                let store = try #require(
                    PrefixCacheStore(directory: directory, byteBudget: budget, geometry: Self.geometry))
                a = Self.writeChain(1, from: 0, into: store)
                b = Self.writeChain(1, from: 50, into: store)
                Self.writeChain(1, from: 90, into: store)
                // Use A, so B is now the coldest.
                store.touch(a)
                store.flush()
            }
            let store = try #require(
                PrefixCacheStore(directory: directory, byteBudget: budget, geometry: Self.geometry))
            Self.writeChain(1, from: 130, into: store)
            #expect(store.nodeCount == 3)
            #expect(store.record(of: a[0]) != nil)
            #expect(store.record(of: b[0]) == nil)
            #expect(Self.files(in: directory, withExtension: "kv").count == 3)
        }
    }

    @Test("A chain's middle is never evicted ahead of its leaf")
    func keepsChainsWhole() throws {
        try Self.withDirectory { directory in
            let store = try #require(
                PrefixCacheStore(directory: directory, byteBudget: 3 * Self.nodeBytes, geometry: Self.geometry))
            let chain = Self.writeChain(3, into: store)
            // The chain's root is its coldest node, but only the leaf may go.
            Self.writeChain(1, from: 50, into: store)
            #expect(store.record(of: chain[0]) != nil && store.record(of: chain[1]) != nil)
            #expect(store.record(of: chain[2]) == nil)
        }
    }

    @Test("A damaged file takes its node and everything below it out")
    func damagedFileDropsSubtree() throws {
        try Self.withDirectory { directory in
            let store = try #require(
                PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: Self.geometry))
            let chain = Self.writeChain(3, into: store)
            let file = directory.appendingPathComponent("\(chain[1]).kv")
            try Data([1, 2, 3]).write(to: file)
            #expect(store.read(chain[1], withSnapshot: false) == nil)
            store.flush()
            #expect(store.nodeCount == 1)
            #expect(store.record(of: chain[2]) == nil)
            let reopened = try #require(
                PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: Self.geometry))
            #expect(reopened.nodeCount == 1)
        }
    }

    @Test("Opening drops a cut-off index line, a node whose file is gone, and files the index does not name")
    func openingTidies() throws {
        try Self.withDirectory { directory in
            let chain: [PrefixTree.Key]
            do {
                let store = try #require(
                    PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: Self.geometry))
                chain = Self.writeChain(3, into: store)
            }
            let index = directory.appendingPathComponent("index.jsonl")
            let handle = try FileHandle(forWritingTo: index)
            try handle.seekToEnd()
            try handle.write(contentsOf: Data("{\"key\":\"cut".utf8))
            try handle.close()
            try FileManager.default.removeItem(at: directory.appendingPathComponent("\(chain[2]).kv"))
            let stray = directory.appendingPathComponent("stray.kv")
            try Data([0]).write(to: stray)
            let reopened = try #require(
                PrefixCacheStore(directory: directory, byteBudget: 1 << 20, geometry: Self.geometry))
            #expect(reopened.nodeCount == 2)
            #expect(!FileManager.default.fileExists(atPath: stray.path))
        }
    }

    @Test("A file's header is checked against its length")
    func decodeChecksHeaders() {
        let blobs = [Data([1, 2, 3]), Data(), Data([4])]
        let encoded = PrefixCacheStore.encode(blobs)
        #expect(encoded.count == PrefixCacheStore.encodedSize(of: blobs))
        #expect(PrefixCacheStore.decode(encoded) == blobs)
        #expect(PrefixCacheStore.decode(encoded.dropLast()) == nil)
        #expect(PrefixCacheStore.decode(encoded + Data([0])) == nil)
        var huge = encoded
        huge.replaceSubrange(4..<12, with: [UInt8](repeating: 0xff, count: 8))
        #expect(PrefixCacheStore.decode(huge) == nil)
        #expect(PrefixCacheStore.decode(Data([0, 0, 0, 0])) == nil)
    }

    @Test("A model's directory changes when one of its files does")
    func directoryFollowsTheModel() throws {
        try Self.withDirectory { model in
            try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
            let weights = model.appendingPathComponent("weights.bin")
            try Data([1, 2, 3]).write(to: weights)
            let first = try #require(PrefixCacheStore.defaultDirectory(forModelAt: model))
            #expect(PrefixCacheStore.defaultDirectory(forModelAt: model) == first)
            try Data([1, 2, 3, 4]).write(to: weights)
            #expect(PrefixCacheStore.defaultDirectory(forModelAt: model) != first)
        }
    }
}
